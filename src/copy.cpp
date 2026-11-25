// ---- benchmark metadata ----
// benchmark=copy
// level=1
// component=GPU/Memory/Bandwidth
// description=Copy kernel B[e] = A[e] with 16-byte elements in a grid-stride loop, consecutive addresses per wavefront (coalesced), as the Copy kernel of STREAM (McCalpin) / BabelStream; read + write bandwidth per memory level
// tags=memory-bound;bandwidth;read-write
// version=0.4.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.working_set_bytes=4294967296|multiple of 32, >= 32768, <= 68719476736|Size of both arrays together (A and B, half each); selects the memory level (L2, LLC, HBM)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx4;global_store_dwordx4
// expect_counters=
// ---- end metadata ----
//
// copy: the threads copy 16-byte elements from A to B in a grid-stride loop
// (B[e] = A[e]); neighbouring lanes copy neighbouring elements (coalesced), no
// computation. Fixed grid (128 blocks per CU); working sets up to 64 MiB are copied
// several times per launch (about 16 copies per thread), larger ones once, as in
// write. Same access pattern for read and write, so it is a Level 1 benchmark.
// Source: Copy kernel of STREAM (McCalpin) / BabelStream; grid and repetition
// scheme own (as in write).
// Details: docs/benchmarks.md#copy
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

static const int THREADS = 256;
static const int BLOCKS_PER_CU = 128;                        // as in read / write / gather
static const unsigned long long LLC_BYTES = 64ULL << 20;    // above: every element once per launch
static const unsigned long long COPIES_PER_THREAD = 16;      // for working sets up to LLC_BYTES
static const unsigned ELEM_BYTES = 16;                       // sizeof(uint4)

// Value of element e of A (also computed on the host for verification)
__host__ __device__ inline uint4 value(unsigned long e) {
    const unsigned lo = (unsigned)e, hi = (unsigned)(e >> 32);
    return make_uint4(lo, hi, lo ^ 0xA5A5A5A5u, ~lo);
}

// Fills A with value(e) (setup, not part of the ROI)
__global__ void fill_kernel(uint4* a, unsigned long n) {
    const unsigned long g = (unsigned long)gridDim.x * blockDim.x;
    for (unsigned long e = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += g)
        a[e] = value(e);
}

// Grid-stride loop over the copies k = 0 .. m-1 of one launch; copy k copies
// element k mod n. The position (k mod n) is advanced incrementally, so no 64-bit
// division per copy limits the throughput (as in write). In the cache runs the same
// element is copied several times per launch.
__global__ void copy_kernel(const uint4* a, uint4* b, unsigned long n, unsigned long m) {
    const unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long g = (unsigned long)gridDim.x * blockDim.x;
    const unsigned long g_mod = g % n;
    unsigned long pos = tid % n;
    for (unsigned long k = tid; k < m; k += g) {
        b[pos] = a[pos];
        pos += g_mod;
        if (pos >= n) pos -= n;
    }
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
    unsigned long long ws = 4ULL << 30;
    bool check = false, have_duration = false, have_ws = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "copy: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws;
            valid = parse_ull(val, &ws) && ws >= 32768 && ws <= (1ULL << 36) && ws % (2 * ELEM_BYTES) == 0;
        } else {
            fprintf(stderr, "copy: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "copy: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "copy: invalid value for %s: '%s' (see: run_suite.sh --help copy)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "copy: --duration is required\n"); return 1; }
    if (check) return 0;                                              // arguments valid, no GPU access

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned blocks = BLOCKS_PER_CU * (unsigned)prop.multiProcessorCount;
    const unsigned long long g = (unsigned long long)blocks * THREADS;
    const unsigned long long n = ws / (2 * ELEM_BYTES);               // elements per array
    // Copies per launch: large working sets every element once; small ones (cache
    // levels) a whole number of rounds, about COPIES_PER_THREAD per thread.
    unsigned long long m = n;
    if (ws <= LLC_BYTES) {
        const unsigned long long rounds = (g * COPIES_PER_THREAD + n - 1) / n;
        m = rounds * n;
    }
    fprintf(stderr, "copy: device %s (%s), working_set=%llu B (2 arrays) elements=%llu copies=%llu blocks=%u\n",
            prop.name, pci, ws, n, m, blocks);

    uint4 *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_a, n * ELEM_BYTES));
    HIP_CHECK(hipMalloc(&d_b, n * ELEM_BYTES));
    hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(THREADS), 0, 0, d_a, (unsigned long)n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemset(d_b, 0, n * ELEM_BYTES));
    HIP_CHECK(hipDeviceSynchronize());

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(THREADS), 0, 0,
                       d_a, d_b, (unsigned long)n, (unsigned long)m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(THREADS), 0, 0,
                           d_a, d_b, (unsigned long)n, (unsigned long)m);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification: every element of B holds value(e) ----
    std::vector<uint4> h_b(n);
    HIP_CHECK(hipMemcpy(h_b.data(), d_b, n * ELEM_BYTES, hipMemcpyDeviceToHost));
    bool ok = true;
    unsigned long long bad = 0;
    for (unsigned long long e = 0; e < n; ++e) {
        const uint4 v = value(e), h = h_b[e];
        if (h.x != v.x || h.y != v.y || h.z != v.z || h.w != v.w) { bad = e; ok = false; break; }
    }
    if (!ok) fprintf(stderr, "copy: verification failed at element %llu\n", bad);

    // ---- results; GB = 10^9 byte. Nominal bytes = 16 B read + 16 B written per copy ----
    const double a_gbyte = (double)launches * m * 2 * ELEM_BYTES / 1e9;
    const double a_gflop = 0.0;                            // no arithmetic
    const double a_intensity = a_gflop / a_gbyte;          // 0

    fprintf(stderr, "copy: %ld launches  %.3f s  %.1f GB/s  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_working_set_bytes\": %llu, \"e_elements\": %llu, \"e_copies\": %llu, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false", ws, n, m, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    return ok ? 0 : 3;                          // contract section 7: 3 = verification failed
}
