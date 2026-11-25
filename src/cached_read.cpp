// ---- benchmark metadata ----
// benchmark=cached_read
// level=1
// component=GPU/Memory/Bandwidth
// description=Independent 16-byte reads from a buffer shared by all blocks (global_load_dwordx4), consecutive addresses per wavefront, values only marked as used (no arithmetic); read bandwidth of the cache level the buffer fits in (L1, L2, LLC)
// tags=memory-bound;bandwidth;read-only;cache
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.working_set_bytes=16384|power of two, >= 16384, <= 1073741824|Size of the shared buffer; selects the cache level (L1 32 KiB per CU, L2 4 MiB per XCD, LLC 256 MiB)
// param.iterations=1000|>= 1|Loop iterations per thread and launch (4 reads of 16 B each); controls the launch length
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx4
// expect_counters=
// ---- end metadata ----
//
// cached_read: all blocks read the same buffer of working_set_bytes with 16-byte
// loads; per iteration every thread makes four independent reads at constant
// offsets of a quarter buffer from a start element, consecutive lanes read
// consecutive elements (coalesced), and the start element moves by one wavefront
// (1 KiB) per iteration (bit mask instead of modulo, buffer size a power of two;
// 32-bit byte offsets, 2 VALU address instructions per 4 loads).
// The loaded values only feed an empty inline asm statement, so the reads stay in
// the loop without arithmetic (same idea as shared_read --consume asm). Reuse of
// the buffer within a launch keeps it in the cache level it fits in.
// Read counterpart of write and copy for the cache levels; HBM reads: read.
// Source: own implementation. Reference for comparison: reuse (rocprof-compute).
// Details: docs/benchmarks.md#cached_read
// Interface: docs/benchmark_contract.md

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

static const int THREADS = 256;          // threads per block, as in read, write, copy
static const int BLOCKS_PER_CU = 128;    // fixed grid, as in read, write, copy
static const int SLICES = 4;             // 16-byte reads per thread and iteration
static const unsigned SHIFT = 64;        // start element moves by one wavefront per iteration
static const unsigned ELEM_BYTES = 16;   // sizeof(uint4)
static const unsigned MAGIC = 0x9E3779B9u;   // value that (almost) never occurs; keeps the loop alive

// Value of element e (also computed on the host for verification)
__host__ __device__ inline uint4 value(unsigned e) {
    return make_uint4(e * 2654435761u, e ^ 0xA5A5A5A5u, ~e, e + 0x01234567u);
}

// Fills the buffer with value(e) (setup, not part of the ROI)
__global__ void fill_kernel(uint4* buf, unsigned n) {
    const unsigned g = gridDim.x * blockDim.x;
    for (unsigned e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += g) buf[e] = value(e);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void cached_read(const uint4* __restrict__ buf, unsigned quarter, int iters, unsigned* out)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 32-bit byte offsets from four base pointers (one per quarter): the compiler
    // then uses global loads with a scalar base and a 32-bit vector offset, so the
    // address costs 2 VALU instructions per 4 loads instead of 64-bit arithmetic.
    const char* p0 = (const char*)buf;
    const char* p1 = p0 + (size_t)quarter * ELEM_BYTES;
    const char* p2 = p1 + (size_t)quarter * ELEM_BYTES;
    const char* p3 = p2 + (size_t)quarter * ELEM_BYTES;
    const unsigned mask = (quarter - 1) * ELEM_BYTES;   // quarter = n / 4, a power of two
    const unsigned off0 = tid * ELEM_BYTES;
    uint4 a = make_uint4(0, 0, 0, 0), b = a, c = a, d = a;

    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        // start element in [0, quarter): the four reads are at constant offsets of a
        // quarter buffer from it; a wavefront reads 64 consecutive elements.
        const unsigned off = (off0 + (unsigned)it * SHIFT * ELEM_BYTES) & mask;
        a = *(const uint4*)(p0 + off);
        b = *(const uint4*)(p1 + off);
        c = *(const uint4*)(p2 + off);
        d = *(const uint4*)(p3 + off);
        // empty asm that "uses" all 16 registers: the compiler must keep every read
        // in every iteration but emits no instruction for the use.
        // "v" = VGPR constraint, only valid in the device compilation.
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" :: "v"(a.x), "v"(a.y), "v"(a.z), "v"(a.w),
                           "v"(b.x), "v"(b.y), "v"(b.z), "v"(b.w),
                           "v"(c.x), "v"(c.y), "v"(c.z), "v"(c.w),
                           "v"(d.x), "v"(d.y), "v"(d.z), "v"(d.w));
#endif
    }
    // xor of the 16 words of the last iteration
    const unsigned r = a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^
                       c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    // Block 0 writes its results for the verification; all other blocks only
    // store if the result equals MAGIC, so the loop is needed in every block.
    if (blockIdx.x == 0) out[threadIdx.x] = r;
    else if (r == MAGIC) out[THREADS] = r;
}

// Host reference of r for thread tid (block 0, so tid = threadIdx.x)
static unsigned reference(unsigned tid, unsigned quarter, int iters) {
    const unsigned base = (tid + (unsigned)(iters - 1) * SHIFT) & (quarter - 1);
    unsigned r = 0;
    for (int s = 0; s < SLICES; ++s) {
        const uint4 v = value(base + s * quarter);
        r ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

using mono_clock = std::chrono::steady_clock;  // CLOCK_MONOTONIC on Linux
using unix_clock = std::chrono::system_clock;  // Unix time

static long long mono_ns(mono_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
static double unix_s(unix_clock::time_point t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static bool parse_double(const char* s, double* out) {
    char* end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

static bool parse_ull(const char* s, unsigned long long* out) {
    char* end;
    errno = 0;
    if (*s == '-') return false;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0') return false;
    *out = v;
    return true;
}

int main(int argc, char** argv) {
    double duration = 0;
    unsigned long long ws = 16384, iters_arg = 1000;
    bool check = false, have_duration = false, have_ws = false, have_iters = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "cached_read: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws;
            valid = parse_ull(val, &ws) && ws >= 16384 && ws <= (1ULL << 30) && (ws & (ws - 1)) == 0;
        } else if (!strcmp(key, "--iterations")) {
            seen = &have_iters;
            valid = parse_ull(val, &iters_arg) && iters_arg >= 1 && iters_arg <= 1000000000ULL;
        } else {
            fprintf(stderr, "cached_read: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "cached_read: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "cached_read: invalid value for %s: '%s' (see: run_suite.sh --help cached_read)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "cached_read: --duration is required\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access
    const int iters = (int)iters_arg;

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned cus = (unsigned)prop.multiProcessorCount;
    const unsigned blocks = BLOCKS_PER_CU * cus;
    const unsigned n = (unsigned)(ws / ELEM_BYTES);          // elements in the buffer
    const unsigned quarter = n / SLICES;
    // bytes read per launch: blocks x threads x iterations x 4 reads x 16 B
    const double bytes_per_launch = (double)blocks * THREADS * iters * SLICES * ELEM_BYTES;
    fprintf(stderr, "cached_read: device %s (%s), buffer=%llu B elements=%u blocks=%u threads=%d iterations=%d, %.3f GB per launch\n",
            prop.name, pci, ws, n, blocks, THREADS, iters, bytes_per_launch / 1e9);

    uint4* d_buf;
    unsigned* d_out;
    HIP_CHECK(hipMalloc(&d_buf, ws));
    HIP_CHECK(hipMalloc(&d_out, (THREADS + 1) * sizeof(unsigned)));
    HIP_CHECK(hipMemset(d_out, 0, (THREADS + 1) * sizeof(unsigned)));
    hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(THREADS), 0, 0, d_buf, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(cached_read, dim3(blocks), dim3(THREADS), 0, 0, d_buf, quarter, iters, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(cached_read, dim3(blocks), dim3(THREADS), 0, 0, d_buf, quarter, iters, d_out);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification: results of block 0 against the host reference ----
    std::vector<unsigned> h_out(THREADS);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, THREADS * sizeof(unsigned), hipMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned t = 0; t < (unsigned)THREADS && ok; ++t) ok = (h_out[t] == reference(t, quarter, iters));
    if (!ok) fprintf(stderr, "cached_read: verification failed\n");

    // ---- results; GB = 10^9 byte. Bytes = bytes loaded by the threads ----
    const double a_gbyte = (double)launches * bytes_per_launch / 1e9;
    const double a_gflop = 0.0;                     // no arithmetic
    const double a_intensity = a_gflop / a_gbyte;   // 0

    fprintf(stderr, "cached_read: %ld launches  %.3f s  %.1f GB/s  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_working_set_bytes\": %llu, \"e_loop_iterations\": %d, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false", ws, iters, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipFree(d_out));
    return ok ? 0 : 3;                          // contract section 7: 3 = verification failed
}
