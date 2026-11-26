// ---- benchmark metadata ----
// benchmark=l1_width
// level=1
// component=GPU/Memory/Bandwidth
// description=EXPERIMENT ONLY (axis validation): cached_read with 4-, 8- or 16-byte loads, otherwise identical; shows how the L1 limit (bytes per clock vs. lookups per clock) depends on the access width; not a coverage point
// tags=memory-bound;bandwidth;read-only;cache;experiment
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.width=16|4, 8, 16|Bytes per load and lane (unsigned, uint2, uint4)
// param.working_set_bytes=16384|power of two, >= 16384, <= 1073741824|Size of the shared buffer; 16 KiB = L1
// param.iterations=1000|>= 1|Loop iterations per thread and launch (4 reads each)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dword;global_load_dwordx2;global_load_dwordx4
// expect_counters=
// ---- end metadata ----
//
// l1_width: the kernel of cached_read 0.2.0 (block-wise sweep, pseudo-random start per
// block, four reads per iteration, one per quarter, 32-bit byte offsets, values only fed
// to an empty inline asm), with the load width as a parameter. Experiment for the L1
// axis (project note axis_validation.md, "L1: width experiment"): per width, L1 lookups
// per instruction, lookups per clock and CU, bytes per clock and CU.
// Lives in src/experimental/; build and run with --src-dir src/experimental and
// --config config/runs_experimental.cfg.
// Source: own implementation (derived from cached_read).

#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static const int THREADS = 256;
static const int BLOCKS_PER_CU = 128;
static const int SLICES = 4;                 // reads per thread and iteration (one per quarter)
static const unsigned MAGIC = 0x9E3779B9u;

// Value of 4-byte word i of the buffer (also computed on the host)
__host__ __device__ inline unsigned word_value(unsigned i) { return (i + 1u) * 2654435761u; }

__global__ void fill_kernel(unsigned* buf, unsigned nwords) {
    const unsigned g = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nwords; i += g) buf[i] = word_value(i);
}

// Pseudo-random start offset of a block, aligned to one block step (block 0 starts at 0)
__host__ __device__ inline unsigned block_start(unsigned block, unsigned step) {
    return (block * 2654435761u) & ~(step - 1);
}

// T = unsigned, uint2 or uint4; W = sizeof(T) / 4 words
template <typename T, int W>
__global__ void l1_width(const unsigned* __restrict__ buf, unsigned quarter_bytes, int iters, unsigned* out)
{
    const unsigned step = THREADS * (unsigned)sizeof(T);       // a block moves by one block per iteration
    const char* p0 = (const char*)buf;
    const char* p1 = p0 + quarter_bytes;
    const char* p2 = p1 + quarter_bytes;
    const char* p3 = p2 + quarter_bytes;
    const unsigned mask = quarter_bytes - 1;
    const unsigned off0 = block_start(blockIdx.x, step) + threadIdx.x * (unsigned)sizeof(T);
    T v[SLICES];
    for (int s = 0; s < SLICES; ++s) v[s] = T{};

    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        const unsigned off = (off0 + (unsigned)it * step) & mask;
        v[0] = *(const T*)(p0 + off);
        v[1] = *(const T*)(p1 + off);
        v[2] = *(const T*)(p2 + off);
        v[3] = *(const T*)(p3 + off);
        // empty asm per loaded word: every read stays in the loop, no instruction for the use
#if defined(__HIP_DEVICE_COMPILE__)
        #pragma unroll
        for (int s = 0; s < SLICES; ++s) {
            const unsigned* w = (const unsigned*)&v[s];
            #pragma unroll
            for (int k = 0; k < W; ++k) asm volatile("" :: "v"(w[k]));
        }
#endif
    }
    unsigned r = 0;
    #pragma unroll
    for (int s = 0; s < SLICES; ++s) {
        const unsigned* w = (const unsigned*)&v[s];
        #pragma unroll
        for (int k = 0; k < W; ++k) r ^= w[k];
    }
    if (blockIdx.x == 0) out[threadIdx.x] = r;
    else if (r == MAGIC) out[THREADS] = r;
}

// Host reference for thread t of block 0 (start offset 0)
static unsigned reference(unsigned t, unsigned width, unsigned quarter_bytes, int iters) {
    const unsigned step = THREADS * width;
    const unsigned off = (block_start(0, step) + t * width + (unsigned)(iters - 1) * step) & (quarter_bytes - 1);
    unsigned r = 0;
    for (int s = 0; s < SLICES; ++s)
        for (unsigned k = 0; k < width / 4; ++k) r ^= word_value((s * quarter_bytes + off) / 4 + k);
    return r;
}

using mono_clock = std::chrono::steady_clock;
using unix_clock = std::chrono::system_clock;

static long long mono_ns(mono_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
static double unix_s(unix_clock::time_point t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}
static bool parse_double(const char* s, double* out) {
    char* end; errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !std::isfinite(v)) return false;
    *out = v; return true;
}
static bool parse_ull(const char* s, unsigned long long* out) {
    char* end; errno = 0;
    if (*s == '-') return false;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0') return false;
    *out = v; return true;
}

static void launch(unsigned width, unsigned blocks, const unsigned* buf, unsigned qb, int iters, unsigned* out) {
    if (width == 4)      hipLaunchKernelGGL((l1_width<unsigned, 1>), dim3(blocks), dim3(THREADS), 0, 0, buf, qb, iters, out);
    else if (width == 8) hipLaunchKernelGGL((l1_width<uint2, 2>), dim3(blocks), dim3(THREADS), 0, 0, buf, qb, iters, out);
    else                 hipLaunchKernelGGL((l1_width<uint4, 4>), dim3(blocks), dim3(THREADS), 0, 0, buf, qb, iters, out);
}

int main(int argc, char** argv) {
    double duration = 0;
    unsigned long long ws = 16384, iters_arg = 1000, width = 16;
    bool check = false, have_duration = false, have_ws = false, have_iters = false, have_width = false;

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "l1_width: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen; bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration; valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws; valid = parse_ull(val, &ws) && ws >= 16384 && ws <= (1ULL << 30) && (ws & (ws - 1)) == 0;
        } else if (!strcmp(key, "--iterations")) {
            seen = &have_iters; valid = parse_ull(val, &iters_arg) && iters_arg >= 1 && iters_arg <= 1000000000ULL;
        } else if (!strcmp(key, "--width")) {
            seen = &have_width; valid = parse_ull(val, &width) && (width == 4 || width == 8 || width == 16);
        } else {
            fprintf(stderr, "l1_width: unknown option %s\n", key); return 1;
        }
        if (*seen) { fprintf(stderr, "l1_width: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) { fprintf(stderr, "l1_width: invalid value for %s: '%s'\n", key, val); return 1; }
    }
    if (!have_duration) { fprintf(stderr, "l1_width: --duration is required\n"); return 1; }
    if (check) return 0;
    const int iters = (int)iters_arg;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned blocks = BLOCKS_PER_CU * (unsigned)prop.multiProcessorCount;
    const unsigned qb = (unsigned)(ws / SLICES);
    const double bytes_per_launch = (double)blocks * THREADS * iters * SLICES * width;
    fprintf(stderr, "l1_width: device %s (%s), width=%llu B buffer=%llu B blocks=%u iterations=%d, %.3f GB per launch\n",
            prop.name, pci, width, ws, blocks, iters, bytes_per_launch / 1e9);

    unsigned *d_buf, *d_out;
    HIP_CHECK(hipMalloc(&d_buf, ws));
    HIP_CHECK(hipMalloc(&d_out, (THREADS + 1) * sizeof(unsigned)));
    HIP_CHECK(hipMemset(d_out, 0, (THREADS + 1) * sizeof(unsigned)));
    hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(THREADS), 0, 0, d_buf, (unsigned)(ws / 4));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    launch((unsigned)width, blocks, d_buf, qb, iters, d_out);        // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        launch((unsigned)width, blocks, d_buf, qb, iters, d_out);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    std::vector<unsigned> h_out(THREADS);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, THREADS * sizeof(unsigned), hipMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned t = 0; t < (unsigned)THREADS && ok; ++t) ok = (h_out[t] == reference(t, (unsigned)width, qb, iters));
    if (!ok) fprintf(stderr, "l1_width: verification failed\n");

    const double a_gbyte = (double)launches * bytes_per_launch / 1e9;
    fprintf(stderr, "l1_width: %ld launches  %.3f s  %.1f GB/s  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_width_bytes\": %llu, \"e_working_set_bytes\": %llu, \"e_loop_iterations\": %d, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           0.0, a_gbyte, 0.0, 0.0, a_gbyte / elapsed,
           ok ? "true" : "false", width, ws, iters, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipFree(d_out));
    return ok ? 0 : 3;
}
