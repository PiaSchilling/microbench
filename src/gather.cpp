// ---- benchmark metadata ----
// benchmark=gather
// level=1
// component=GPU/Memory/Bandwidth
// description=Indirect read t[idx[k]] (gather), independent accesses; strided or random order of the cache lines; bandwidth of irregular reads per memory level
// tags=memory-bound;bandwidth;irregular
// version=0.2.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.pattern=random|strided, random|Order of the accesses: strided = lines in address order, random = lines in a random permutation (no duplicates)
// param.working_set_bytes=1073741824|>= 2 x stride_bytes, <= 17179869184|Size of the table t; selects the memory level (L1, L2, LLC, HBM)
// param.stride_bytes=128|multiple of 8|Distance between the accessed elements; 128 = one access per cache line
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx2
// expect_counters=
// ---- end metadata ----
//
// gather: the threads read elements of the table t at the addresses given by an
// index array (t[idx[k]]); the accesses are independent of each other. The index
// visits every stride-th element of t once, either in address order (strided) or
// in a random order (random). Fixed grid (128 blocks per CU); small tables are read
// many times per launch, large ones once. Measures the bandwidth of irregular
// reads per memory level; the counterpart of chase (dependent accesses, latency).
// Source: own implementation (gather as in the HPC Challenge / Uwe Kuester loops).
// Details: docs/benchmarks.md#gather
// Interface: docs/benchmark_contract.md

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
static const int BLOCKS_PER_CU = 128;                        // as in read / reuse
static const unsigned long long LLC_BYTES = 64ULL << 20;    // above: every element once per launch
static const unsigned long long ACCESSES_PER_THREAD = 16;    // for tables up to LLC_BYTES

// Grid-stride loop over the accesses k = 0 .. m-1 of one launch; access k reads
// t[idx[k mod n]]. The position in the index (k mod n) is advanced incrementally,
// so no 64-bit division per access limits the throughput.
__global__ void gather(const double* t, const unsigned int* idx, double* a,
                       unsigned long n, unsigned long m) {
    const unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long g = (unsigned long)gridDim.x * blockDim.x;
    const unsigned long g_mod = g % n;
    unsigned long pos = tid % n;
    double s = 0.0;
    for (unsigned long k = tid; k < m; k += g) {
        s += t[idx[pos]];
        pos += g_mod;
        if (pos >= n) pos -= n;
    }
    a[tid] = s;                                // contiguous store, prevents dead-code elimination
}

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
    const char* pattern = "random";
    unsigned long long ws = 1ULL << 30, stride = 128;
    bool check = false, have_duration = false, have_pattern = false, have_ws = false,
         have_stride = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "gather: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--pattern")) {
            seen = &have_pattern;
            valid = !strcmp(val, "strided") || !strcmp(val, "random");
            pattern = val;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws;
            valid = parse_ull(val, &ws) && ws > 0 && ws <= (1ULL << 34);
        } else if (!strcmp(key, "--stride_bytes")) {
            seen = &have_stride;
            valid = parse_ull(val, &stride) && stride > 0 && stride % 8 == 0;
        } else {
            fprintf(stderr, "gather: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "gather: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "gather: invalid value for %s: '%s' (see: run_suite.sh --help gather)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "gather: --duration is required\n"); return 1; }
    const unsigned long long table_elems = ws / 8;
    const unsigned long long stride_elems = stride / 8;
    const unsigned long long n = table_elems / stride_elems;          // index length (lines visited)
    if (n < 2) { fprintf(stderr, "gather: --working_set_bytes must be at least 2 x --stride_bytes\n"); return 1; }
    if (table_elems > 0xFFFFFFFFull) { fprintf(stderr, "gather: table too large for 32-bit indices\n"); return 1; }
    if (check) return 0;                                              // arguments valid, no GPU access

    const bool random = !strcmp(pattern, "random");

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned blocks = BLOCKS_PER_CU * (unsigned)prop.multiProcessorCount;
    const unsigned long long g = (unsigned long long)blocks * THREADS;
    // Accesses per launch: large tables every element once; small tables (cache
    // levels) a whole number of rounds over the index, about ACCESSES_PER_THREAD per thread.
    unsigned long long m = n;
    if (ws <= LLC_BYTES) {
        const unsigned long long rounds = (g * ACCESSES_PER_THREAD + n - 1) / n;
        m = rounds * n;
    }
    fprintf(stderr, "gather: device %s (%s), pattern=%s table=%llu B stride=%llu B index=%llu accesses=%llu blocks=%u\n",
            prop.name, pci, pattern, ws, stride, n, m, blocks);

    // Index: every stride-th element once; random = the lines in a random permutation
    // (Fisher-Yates, fixed seed, reproducible).
    std::vector<unsigned int> h_idx(n);
    for (unsigned long long k = 0; k < n; ++k) h_idx[k] = (unsigned int)k;
    if (random) {
        std::mt19937_64 rng(12345);
        for (unsigned long long k = n - 1; k > 0; --k) {
            std::uniform_int_distribution<unsigned long long> d(0, k);
            std::swap(h_idx[k], h_idx[d(rng)]);
        }
    }
    for (unsigned long long k = 0; k < n; ++k) h_idx[k] = (unsigned int)(h_idx[k] * stride_elems);

    std::vector<double> h_t(table_elems);
    for (unsigned long long k = 0; k < table_elems; ++k) h_t[k] = (double)(k % 16);

    double *d_t, *d_a;
    unsigned int* d_idx;
    HIP_CHECK(hipMalloc(&d_t, table_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_idx, n * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_a, g * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_t, h_t.data(), table_elems * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_idx, h_idx.data(), n * sizeof(unsigned int), hipMemcpyHostToDevice));

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(gather, dim3(blocks), dim3(THREADS), 0, 0,
                       (const double*)d_t, (const unsigned int*)d_idx, d_a,
                       (unsigned long)n, (unsigned long)m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(gather, dim3(blocks), dim3(THREADS), 0, 0,
                           (const double*)d_t, (const unsigned int*)d_idx, d_a,
                           (unsigned long)n, (unsigned long)m);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification: a[tid] = sum of t[idx[k mod n]] over k = tid, tid+g, ... < m ----
    std::vector<double> h_a(g);
    HIP_CHECK(hipMemcpy(h_a.data(), d_a, g * sizeof(double), hipMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned long long tid = 0; tid < g && ok; tid += (g / 1000 + 1)) {
        double s = 0.0;
        for (unsigned long long k = tid; k < m; k += g) s += h_t[h_idx[k % n]];
        ok = (h_a[tid] == s);
    }

    // ---- results; GB = 10^9 byte ----
    // Nominal bytes: per access 8 B table + 4 B index; per thread and launch 8 B result.
    // The hardware moves whole cache lines; the difference shows in the counters.
    const double accesses = (double)launches * m;
    const double a_gbyte = (accesses * 12.0 + (double)launches * g * 8.0) / 1e9;
    const double a_gflop = accesses / 1e9;                 // one add per access
    const double a_intensity = a_gflop / a_gbyte;

    fprintf(stderr, "gather: %ld launches  %.3f s  %.1f GB/s  %.3f G accesses/s  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, accesses / elapsed / 1e9, ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, \"precision\": \"fp64\", \"pattern\": \"%s\", "
           "\"d_gaccess_per_s\": %.9g, "
           "\"e_working_set_bytes\": %llu, \"e_stride_bytes\": %llu, \"e_accesses\": %llu, "
           "\"e_index_length\": %llu, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false", pattern, accesses / elapsed / 1e9,
           ws, stride, m, n, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_t));
    HIP_CHECK(hipFree(d_idx));
    HIP_CHECK(hipFree(d_a));
    return ok ? 0 : 3;                          // contract section 7: 3 = verification failed
}
