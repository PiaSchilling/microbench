// ---- benchmark metadata ----
// benchmark=shared_read
// level=1
// component=GPU/Memory/LDS
// description=Independent 16-byte reads from the LDS (ds_read_b128), consecutive addresses per wavefront, no bank conflicts; LDS read bandwidth
// tags=lds;bandwidth;read-only
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.iterations=1000|>= 1|Loop iterations per thread and launch (4 reads of 16 B each); controls the launch length
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=ds_read_b128
// expect_counters=
// ---- end metadata ----
//
// shared_read: every block fills 16 KiB of LDS once and then reads it repeatedly with
// 16-byte loads; the reads are independent of each other, consecutive lanes read
// consecutive 16-byte elements (no bank conflicts), and the start address moves
// by one wavefront (1 KiB) per iteration so the loads cannot be hoisted. Measures
// the LDS read bandwidth; theoretical peak 128 B per clock and CU.
// Source: own implementation. Reference for comparison: shared_read_rocprof.
// Details: docs/benchmarks.md#shared_read
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

static const int THREADS = 256;          // threads per block, as in read and gather
static const int BLOCKS_PER_CU = 128;    // fixed grid, as in read and gather
static const int SLICES = 4;             // 16-byte reads per thread and iteration
static const int ELEMS = THREADS * SLICES;   // 1024 x 16 B = 16 KiB LDS per block (4 blocks per CU)
static const int SHIFT = 64;             // start element moves by one wavefront per iteration
static const unsigned MAGIC = 0x9E3779B9u;   // value that (almost) never occurs; keeps the loop alive

// Fill pattern of LDS element j, component c (also computed on the host for verification)
__host__ __device__ inline unsigned fill(unsigned j, unsigned c) {
    return (j * 4u + c + 1u) * 2654435761u;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void lds_read(unsigned* out, int iters)
{
    __shared__ uint4 buf[ELEMS];
    const unsigned tid = threadIdx.x;

    #pragma unroll
    for (int s = 0; s < SLICES; ++s) {
        const unsigned j = tid + s * THREADS;
        buf[j] = make_uint4(fill(j, 0), fill(j, 1), fill(j, 2), fill(j, 3));
    }
    __syncthreads();

    uint4 v = make_uint4(0, 0, 0, 0);
    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        // base in [0, 256): the four reads of an iteration are at constant offsets
        // (0, 4, 8, 12 KiB) from it; a wavefront reads 64 consecutive elements.
        const unsigned base = (tid + (unsigned)it * SHIFT) & (THREADS - 1);
        const uint4 a = buf[base];
        const uint4 b = buf[base + THREADS];
        const uint4 c = buf[base + 2 * THREADS];
        const uint4 d = buf[base + 3 * THREADS];
        // three-operand adds (v_add3_u32): 2 VALU instructions per 16-byte read
        v.x += a.x + b.x;  v.x += c.x + d.x;
        v.y += a.y + b.y;  v.y += c.y + d.y;
        v.z += a.z + b.z;  v.z += c.z + d.z;
        v.w += a.w + b.w;  v.w += c.w + d.w;
    }
    const unsigned r = v.x ^ v.y ^ v.z ^ v.w;
    // Block 0 writes its results for the verification; all other blocks only
    // store if the result equals MAGIC, so the loop is needed in every block.
    if (blockIdx.x == 0) out[tid] = r;
    else if (r == MAGIC) out[THREADS] = r;
}

// Host reference of r for thread tid (same arithmetic, unsigned wrap-around)
static unsigned reference(unsigned tid, int iters) {
    unsigned v[4] = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        const unsigned base = (tid + (unsigned)it * SHIFT) & (THREADS - 1);
        for (unsigned c = 0; c < 4; ++c)
            for (int s = 0; s < SLICES; ++s) v[c] += fill(base + s * THREADS, c);
    }
    return v[0] ^ v[1] ^ v[2] ^ v[3];
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
    unsigned long long iters_arg = 1000;
    bool check = false, have_duration = false, have_iters = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "shared_read: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--iterations")) {
            seen = &have_iters;
            valid = parse_ull(val, &iters_arg) && iters_arg >= 1 && iters_arg <= 1000000000ULL;
        } else {
            fprintf(stderr, "shared_read: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "shared_read: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "shared_read: invalid value for %s: '%s' (see: run_suite.sh --help shared_read)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "shared_read: --duration is required\n"); return 1; }
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
    // LDS bytes read per launch: blocks x threads x iterations x 4 reads x 16 B
    const double bytes_per_launch = (double)blocks * THREADS * iters * SLICES * 16.0;
    fprintf(stderr, "shared_read: device %s (%s), blocks=%u threads=%d iterations=%d lds_per_block=%d B, %.3f GB per launch\n",
            prop.name, pci, blocks, THREADS, iters, ELEMS * 16, bytes_per_launch / 1e9);

    unsigned* d_out;
    HIP_CHECK(hipMalloc(&d_out, (THREADS + 1) * sizeof(unsigned)));
    HIP_CHECK(hipMemset(d_out, 0, (THREADS + 1) * sizeof(unsigned)));

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(lds_read, dim3(blocks), dim3(THREADS), 0, 0, d_out, iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(lds_read, dim3(blocks), dim3(THREADS), 0, 0, d_out, iters);
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
    for (unsigned t = 0; t < (unsigned)THREADS && ok; ++t) ok = (h_out[t] == reference(t, iters));

    // ---- results; GB = 10^9 byte. Bytes are LDS bytes (no HBM traffic) ----
    const double a_gbyte = (double)launches * bytes_per_launch / 1e9;
    const double a_gflop = 0.0;                     // integer adds only
    const double a_intensity = a_gflop / a_gbyte;   // 0
    // Theoretical peak at the measured clock is not known here; at 2100 MHz:
    // 128 B x CUs x 2.1 GHz (for comparison in the log only)
    const double peak_2100 = 128.0 * cus * 2.1;    // GB/s

    fprintf(stderr, "shared_read: %ld launches  %.3f s  %.1f GB/s  (%.1f %% of %.0f GB/s at 2100 MHz)  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, 100.0 * a_gbyte / elapsed / peak_2100, peak_2100,
            ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_lds_bytes_per_block\": %d, \"e_loop_iterations\": %d, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false", ELEMS * 16, iters, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_out));
    return ok ? 0 : 3;                          // contract section 7: 3 = verification failed
}
