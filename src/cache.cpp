// ---- benchmark metadata ----
// benchmark=cache
// level=1
// component=GPU/Memory/Cache
// description=Every block re-reads its own chunk of an array several times per launch (double); cache bandwidth (L1, L2) depending on the working set
// tags=memory-bound;bandwidth;cache
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.working_set_bytes=3735552|>= 8 x blocks (blocks = 4 x CUs)|Total size of the array; per CU it is working_set_bytes / CUs and decides which cache level serves the reads (L1 32 KiB per CU, L2 4 MiB per XCD)
// param.passes=2000|>= 1|How often every block reads its chunk per launch; controls the kernel duration
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx2
// expect_counters=
// ---- end metadata ----
//
// Why a separate benchmark: the L2 does not keep data across kernel launches
// (validation finding 3), and stream reads every element only once per launch,
// so stream can never measure L1 or L2. Here the reuse happens inside one launch.
//
// Kernel: the array is split into one chunk per block. Every block reads its
// chunk `passes` times (coalesced: thread t reads elements t, t+256, t+512, ...)
// and adds the values into four partial sums (four independent chains, so the
// add latency does not limit). Only reads; every thread writes one result at
// the end, so the compiler cannot remove the loads. A compiler barrier after
// each pass forces the values to be loaded again instead of kept in registers.
//
// A block stays on one CU during the launch, so its chunk lives in the caches
// of that CU and XCD. With blocks = 4 x CUs, every CU holds about
// working_set_bytes / CUs:
//   - L1:  working set per CU <= 32 KiB        (default: 16 KiB per CU on 228 CUs)
//   - L2:  working set per XCD <= 4 MiB, but > L1
// Which level actually serves the reads is checked with the counters
// (L1: almost no TCP->TCC requests; L2: TCC hits, almost no reads beyond L2).
//
// Interface: docs/benchmark_contract.md.

#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define DATA_TYPE double

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static const int THREADS = 256;      // threads per block
static const int BLOCKS_PER_CU = 4;  // 16 wavefronts per CU

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void read_chunk(const DATA_TYPE* A, DATA_TYPE* B, size_t chunk, long passes) {
    const size_t begin = (size_t)blockIdx.x * chunk;   // first element of this block's chunk
    const size_t end   = begin + chunk;
    const size_t step  = blockDim.x;

    DATA_TYPE s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (long p = 0; p < passes; ++p) {
        size_t j = begin + threadIdx.x;
        for (; j + 3 * step < end; j += 4 * step) {    // four elements per step, four chains
            s0 += A[j];
            s1 += A[j + step];
            s2 += A[j + 2 * step];
            s3 += A[j + 3 * step];
        }
        for (; j < end; j += step) s0 += A[j];          // rest of the chunk
        __asm__ volatile("" ::: "memory");               // load again in the next pass
    }
    B[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s0 + s1 + s2 + s3;
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
    if (*s == '-') return false;
    char* end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0') return false;
    *out = v;
    return true;
}

// Input values: small integers (0..15), so all sums are exact in double and
// the result can be verified exactly.
static DATA_TYPE value(size_t i) { return (DATA_TYPE)(i % 16); }

int main(int argc, char** argv) {
    double duration = 0;
    unsigned long long ws_bytes = 3735552;   // 16 KiB per CU on 228 CUs
    unsigned long long passes = 2000;
    bool check = false, have_duration = false, have_ws = false, have_passes = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "cache: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "cache: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "cache: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--working_set_bytes")) {
            if (have_ws) { fprintf(stderr, "cache: --working_set_bytes given twice\n"); return 1; }
            have_ws = true;
            if (!parse_ull(val, &ws_bytes) || ws_bytes < sizeof(DATA_TYPE)) {
                fprintf(stderr, "cache: --working_set_bytes must be an integer >= %zu (got '%s')\n",
                        sizeof(DATA_TYPE), val);
                return 1;
            }
        } else if (!strcmp(key, "--passes")) {
            if (have_passes) { fprintf(stderr, "cache: --passes given twice\n"); return 1; }
            have_passes = true;
            if (!parse_ull(val, &passes) || passes < 1 || passes > 1000000000ULL) {
                fprintf(stderr, "cache: --passes must be an integer between 1 and 1e9 (got '%s')\n", val);
                return 1;
            }
        } else {
            fprintf(stderr, "cache: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "cache: --duration is required\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access
                                               // (minimum size needs the CU count, checked below)

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    const unsigned cus = (unsigned)prop.multiProcessorCount;
    const unsigned blocks = BLOCKS_PER_CU * cus;
    const size_t chunk = (size_t)(ws_bytes / sizeof(DATA_TYPE)) / blocks;   // elements per block
    if (chunk == 0) {
        fprintf(stderr, "cache: --working_set_bytes too small: at least %zu bytes (8 B x %u blocks)\n",
                sizeof(DATA_TYPE) * blocks, blocks);
        return 1;
    }
    const size_t n = chunk * blocks;                   // effective elements (truncated)
    const size_t ws_eff = n * sizeof(DATA_TYPE);
    const size_t total_threads = (size_t)blocks * THREADS;
    fprintf(stderr, "cache: device %s (%s), working set %zu B (%.1f KiB per CU), chunk %zu elements, "
            "blocks=%u threads=%d passes=%llu\n",
            prop.name, pci, ws_eff, (double)ws_eff / cus / 1024.0, chunk, blocks, THREADS, passes);

    std::vector<DATA_TYPE> h_A(n);
    for (size_t i = 0; i < n; ++i) h_A[i] = value(i);
    DATA_TYPE *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, n * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_B, total_threads * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), n * sizeof(DATA_TYPE), hipMemcpyHostToDevice));

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(read_chunk, dim3(blocks), dim3(THREADS), 0, 0, d_A, d_B, chunk, (long)passes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(read_chunk, dim3(blocks), dim3(THREADS), 0, 0, d_A, d_B, chunk, (long)passes);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification (after the ROI): every thread's sum, exact ----
    std::vector<DATA_TYPE> h_B(total_threads);
    HIP_CHECK(hipMemcpy(h_B.data(), d_B, total_threads * sizeof(DATA_TYPE), hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t t = 0; t < (size_t)THREADS; ++t) {
            double s = 0;
            for (size_t j = b * chunk + t; j < (b + 1) * chunk; j += THREADS) s += (double)h_A[j];
            const double expected = s * (double)passes;
            if ((double)h_B[b * THREADS + t] != expected) {
                if (bad == 0)
                    fprintf(stderr, "cache: verification failed at block %zu thread %zu: %g, expected %g\n",
                            b, t, (double)h_B[b * THREADS + t], expected);
                ++bad;
            }
        }
    }
    const bool ok = (bad == 0);

    // ---- results; GB = 10^9 byte ----
    // Reads: passes x working set per launch; writes: one value per thread.
    const double reads  = (double)launches * passes * ws_eff;
    const double writes = (double)launches * total_threads * sizeof(DATA_TYPE);
    const double a_gbyte = (reads + writes) / 1e9;
    const double a_gflop = (double)launches * passes * n / 1e9;     // one add per element read
    const double a_intensity = a_gflop / a_gbyte;

    fprintf(stderr, "cache: %ld launches  %.3f s  %.1f GB/s\n", launches, elapsed, a_gbyte / elapsed);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_working_set_bytes\": %zu, \"e_working_set_bytes_per_cu\": %.1f, \"e_chunk_elements\": %zu, "
           "\"e_passes\": %llu, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false",
           ws_eff, (double)ws_eff / cus, chunk, passes, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    return ok ? 0 : 3;
}
