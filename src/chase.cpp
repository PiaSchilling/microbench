// ---- benchmark metadata ----
// benchmark=chase
// level=1
// component=GPU/Memory/Latency
// description=Pointer chasing: one thread follows a random chain (j = next[j]); latency pole of the design space
// tags=memory-bound;latency;single-thread
// version=0.3.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.working_set_bytes=1048576|>= 2 x stride_bytes|Size of the chain; selects the memory level (L1, L2, LLC, HBM)
// param.stride_bytes=128|> 0, multiple of 4|Distance between nodes; at least one cache line (128 B on MI300A)
// param.accesses=1000000|>= 1|Accesses per kernel launch; should be >= nodes to visit the whole chain
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dword
// expect_counters=
// ---- end metadata ----
//
// One thread follows a random chain: j = next[j]. Every access depends on the
// previous one, so nothing can run in parallel or be prefetched.
// The chain is a random permutation with exactly one cycle through all nodes
// (Sattolo's algorithm, fixed seed).
// Interface: docs/benchmark_contract.md.

#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

__global__ void chase(const unsigned int* next, unsigned int* out, long accesses) {
    unsigned int j = 0;
    for (long i = 0; i < accesses; i++)
        j = next[j + threadIdx.x];   // threadIdx.x is always 0 here; it keeps the compiler
                                     // from treating the access as uniform and routing it
                                     // through the scalar cache (s_load)
    out[0] = j;                      // prevents dead-code elimination
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
    unsigned long long working_set = 1ULL << 20, stride = 128, accesses_arg = 1000000;
    bool check = false, have_duration = false, have_ws = false, have_stride = false, have_acc = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "chase: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--working_set_bytes")) {
            seen = &have_ws;
            valid = parse_ull(val, &working_set) && working_set > 0;
        } else if (!strcmp(key, "--stride_bytes")) {
            seen = &have_stride;
            valid = parse_ull(val, &stride) && stride > 0 && stride % 4 == 0;
        } else if (!strcmp(key, "--accesses")) {
            seen = &have_acc;
            valid = parse_ull(val, &accesses_arg) && accesses_arg > 0 && accesses_arg <= 0x7FFFFFFFFFFFFFFFull;
        } else {
            fprintf(stderr, "chase: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "chase: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "chase: invalid value for %s: '%s' (see: run_suite.sh --help chase)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "chase: --duration is required\n"); return 1; }

    const long accesses = (long)accesses_arg;
    const size_t nodes = (size_t)(working_set / stride);
    const size_t se = (size_t)(stride / 4);          // stride in elements (unsigned int)
    if (nodes < 2) { fprintf(stderr, "chase: --working_set_bytes must be at least 2 x --stride_bytes\n"); return 1; }
    if ((unsigned long long)nodes * se > 0xFFFFFFFFull) {
        fprintf(stderr, "chase: working set too large for 32-bit indices\n");
        return 1;
    }
    if (check) return 0;                              // arguments valid, no GPU access

    if ((long)nodes > accesses)
        fprintf(stderr, "chase: warning: --accesses (%ld) < nodes (%zu): the chain is not traversed completely, "
                        "the effective working set is smaller than stated (possibly the wrong cache level)\n",
                accesses, nodes);

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    // Random cycle through all nodes (Sattolo): every node is visited exactly once before
    // the chain starts again. Random, so that no prefetcher can help.
    std::vector<unsigned int> p(nodes);
    std::iota(p.begin(), p.end(), 0u);
    std::mt19937_64 rng(12345);
    for (size_t i = nodes - 1; i >= 1; --i) {
        size_t j = rng() % i;                         // j in [0, i-1]
        std::swap(p[i], p[j]);
    }
    std::vector<unsigned int> h(nodes * se, 0);
    for (size_t i = 0; i < nodes; ++i) h[i * se] = (unsigned int)(p[i] * se);

    // the chain must be a single cycle through all nodes
    size_t len = 0;
    unsigned int pos = 0;
    do { pos = h[pos]; ++len; } while (pos != 0 && len <= nodes);
    if (len != nodes) { fprintf(stderr, "chase: chain is not a single cycle through all nodes\n"); return 1; }

    fprintf(stderr, "chase: device %s (%s), working_set=%zu B stride=%llu B nodes=%zu accesses=%ld\n",
            prop.name, pci, nodes * (size_t)stride, stride, nodes, accesses);

    unsigned int *d_next, *d_out;
    HIP_CHECK(hipMalloc(&d_next, h.size() * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_next, h.data(), h.size() * sizeof(unsigned int), hipMemcpyHostToDevice));

    // Warmup (not part of the ROI): loads code and data
    hipLaunchKernelGGL(chase, dim3(1), dim3(1), 0, 0, d_next, d_out, accesses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(chase, dim3(1), dim3(1), 0, 0, d_next, d_out, accesses);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification (after the ROI): final node after `accesses` steps ----
    unsigned int expected = 0;
    for (long k = 0; k < accesses % (long)nodes; ++k) expected = h[expected];
    unsigned int got = 0;
    HIP_CHECK(hipMemcpy(&got, d_out, sizeof(unsigned int), hipMemcpyDeviceToHost));
    const bool ok = (got == expected);
    if (!ok) fprintf(stderr, "chase: verification failed: GPU=%u, expected %u\n", got, expected);

    // ---- results; GB = 10^9 byte ----
    const double total_accesses = (double)launches * accesses;
    const double a_gflop = 0.0;
    const double a_gbyte = total_accesses * sizeof(unsigned int) / 1e9;   // logical: 4 B per access
    const double a_intensity = a_gflop / a_gbyte;                         // 0
    const double d_ns_per_access = elapsed / total_accesses * 1e9;

    fprintf(stderr, "chase: working_set=%zu B  stride=%llu B  nodes=%zu  %ld launches  %.3f s  %.1f ns/access\n",
            nodes * (size_t)stride, stride, nodes, launches, elapsed, d_ns_per_access);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"a_accesses\": %.0f, \"d_ns_per_access\": %.6f, "
           "\"e_working_set_bytes\": %zu, \"e_nodes\": %zu, \"e_stride_bytes\": %llu}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false",
           total_accesses, d_ns_per_access,
           nodes * (size_t)stride, nodes, stride);
    fflush(stdout);

    HIP_CHECK(hipFree(d_next));
    HIP_CHECK(hipFree(d_out));
    return ok ? 0 : 3;
}
