// ---- benchmark metadata ----
// benchmark=write
// level=1
// component=GPU/Memory/Bandwidth
// description=Write-only streaming kernel: every thread stores 16-byte elements in a grid-stride loop, consecutive addresses per wavefront (coalesced), no reads; write bandwidth per memory level
// tags=memory-bound;bandwidth;write-only
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.working_set_bytes=4294967296|multiple of 16, >= 16384, <= 68719476736|Size of the written buffer; selects the memory level (L2, LLC, HBM; the L1 is write-through)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_store_dwordx4
// expect_counters=
// ---- end metadata ----
//
// write: the threads store 16-byte elements to a buffer in a grid-stride loop;
// neighbouring lanes write neighbouring elements (coalesced), nothing is read.
// Fixed grid (128 blocks per CU); buffers up to 64 MiB are written several times
// per launch (about 16 stores per thread), larger ones once. The counterpart of
// read. The L1 is write-through, so the smallest meaningful level is the L2.
// Source: own implementation.
// Details: docs/benchmarks.md#write
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
static const int BLOCKS_PER_CU = 128;                        // as in read / gather
static const unsigned long long LLC_BYTES = 64ULL << 20;    // above: every element once per launch
static const unsigned long long STORES_PER_THREAD = 16;      // for buffers up to LLC_BYTES
static const unsigned ELEM_BYTES = 16;                       // sizeof(uint4)

// Value of element e (also computed on the host for verification)
__host__ __device__ inline uint4 value(unsigned long e) {
    const unsigned lo = (unsigned)e, hi = (unsigned)(e >> 32);
    return make_uint4(lo, hi, lo ^ 0xA5A5A5A5u, ~lo);
}

// Grid-stride loop over the stores k = 0 .. m-1 of one launch; store k writes
// element k mod n. The position (k mod n) is advanced incrementally, so no 64-bit
// division per store limits the throughput. In the cache runs the same element
// is written several times per launch, always with the same value.
__global__ void write_kernel(uint4* buf, unsigned long n, unsigned long m) {
    const unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long g = (unsigned long)gridDim.x * blockDim.x;
    const unsigned long g_mod = g % n;
    unsigned long pos = tid % n;
    for (unsigned long k = tid; k < m; k += g) {
        buf[pos] = value(pos);
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
        if (i + 1 >= argc) { fprintf(stderr, "write: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws;
            valid = parse_ull(val, &ws) && ws >= 16384 && ws <= (1ULL << 36) && ws % ELEM_BYTES == 0;
        } else {
            fprintf(stderr, "write: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "write: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "write: invalid value for %s: '%s' (see: run_suite.sh --help write)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "write: --duration is required\n"); return 1; }
    if (check) return 0;                                              // arguments valid, no GPU access

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned blocks = BLOCKS_PER_CU * (unsigned)prop.multiProcessorCount;
    const unsigned long long g = (unsigned long long)blocks * THREADS;
    const unsigned long long n = ws / ELEM_BYTES;                     // elements in the buffer
    // Stores per launch: large buffers every element once; small buffers (cache
    // levels) a whole number of rounds, about STORES_PER_THREAD per thread.
    unsigned long long m = n;
    if (ws <= LLC_BYTES) {
        const unsigned long long rounds = (g * STORES_PER_THREAD + n - 1) / n;
        m = rounds * n;
    }
    fprintf(stderr, "write: device %s (%s), buffer=%llu B elements=%llu stores=%llu blocks=%u\n",
            prop.name, pci, ws, n, m, blocks);

    uint4* d_buf;
    HIP_CHECK(hipMalloc(&d_buf, ws));
    HIP_CHECK(hipMemset(d_buf, 0, ws));
    HIP_CHECK(hipDeviceSynchronize());

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(THREADS), 0, 0,
                       d_buf, (unsigned long)n, (unsigned long)m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(THREADS), 0, 0,
                           d_buf, (unsigned long)n, (unsigned long)m);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification: every element holds value(e) ----
    std::vector<uint4> h_buf(n);
    HIP_CHECK(hipMemcpy(h_buf.data(), d_buf, ws, hipMemcpyDeviceToHost));
    bool ok = true;
    unsigned long long bad = 0;
    for (unsigned long long e = 0; e < n; ++e) {
        const uint4 v = value(e), h = h_buf[e];
        if (h.x != v.x || h.y != v.y || h.z != v.z || h.w != v.w) { bad = e; ok = false; break; }
    }
    if (!ok) fprintf(stderr, "write: verification failed at element %llu\n", bad);

    // ---- results; GB = 10^9 byte. Nominal bytes = bytes stored ----
    const double a_gbyte = (double)launches * m * ELEM_BYTES / 1e9;
    const double a_gflop = 0.0;                            // no arithmetic
    const double a_intensity = a_gflop / a_gbyte;          // 0

    fprintf(stderr, "write: %ld launches  %.3f s  %.1f GB/s  verify %s\n",
            launches, elapsed, a_gbyte / elapsed, ok ? "ok" : "FAILED");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_working_set_bytes\": %llu, \"e_stores\": %llu, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false", ws, m, blocks, THREADS);
    fflush(stdout);

    HIP_CHECK(hipFree(d_buf));
    return ok ? 0 : 3;                          // contract section 7: 3 = verification failed
}
