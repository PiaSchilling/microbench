// ---- benchmark metadata ----
// benchmark=stream
// level=1
// component=GPU/Memory/Bandwidth
// description=Copy kernel B[i] = A[i] (double); bandwidth pole of the design space
// tags=memory-bound;bandwidth
// version=0.3.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.elements=134217728|> 0, multiple of 256|Elements per array (8 B each); working set = 16 B x elements
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx2;global_store_dwordx2
// expect_counters=
// ---- end metadata ----
//
// One thread per element copies A to B. 100 launches per synchronization, so that
// the launch overhead does not dominate for small arrays. Interface:
// docs/benchmark_contract.md.

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

__global__ void copy(const DATA_TYPE* A, DATA_TYPE* B) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    B[i] = A[i];
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
    const int threads = 256;
    double duration = 0;
    unsigned long long elements = 1ULL << 27;
    bool check = false, have_duration = false, have_elements = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "stream: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "stream: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "stream: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--elements")) {
            if (have_elements) { fprintf(stderr, "stream: --elements given twice\n"); return 1; }
            have_elements = true;
            if (!parse_ull(val, &elements) || elements == 0 || elements % threads != 0) {
                fprintf(stderr, "stream: --elements must be an integer > 0 and divisible by %d (got '%s')\n",
                        threads, val);
                return 1;
            }
        } else {
            fprintf(stderr, "stream: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "stream: --duration is required\n"); return 1; }
    if (elements / threads > 0xFFFFFFFFull) { fprintf(stderr, "stream: --elements too large for the grid\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    const size_t n = (size_t)elements;
    const unsigned blocks = (unsigned)(n / threads);   // one element per thread
    const int batch = 100;                             // launches per synchronization
    fprintf(stderr, "stream: device %s (%s), elements=%zu blocks=%u threads=%d\n",
            prop.name, pci, n, blocks, threads);

    DATA_TYPE *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, n * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_B, n * sizeof(DATA_TYPE)));
    std::vector<DATA_TYPE> h_A(n);
    for (size_t i = 0; i < n; ++i) h_A[i] = (DATA_TYPE)i;
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), n * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B, 0, n * sizeof(DATA_TYPE)));   // B starts at 0

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(copy, dim3(blocks), dim3(threads), 0, 0, d_A, d_B);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: batches until duration is reached, at least one ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        for (int b = 0; b < batch; ++b)
            hipLaunchKernelGGL(copy, dim3(blocks), dim3(threads), 0, 0, d_A, d_B);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        launches += batch;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- verification (after the ROI): sample, B[idx] must equal A[idx] ----
    const size_t samples = (n < 1024) ? n : 1024;
    bool ok = true;
    for (size_t k = 0; k < samples && ok; ++k) {
        size_t idx = samples > 1 ? (n - 1) / (samples - 1) * k : 0;
        DATA_TYPE got;
        HIP_CHECK(hipMemcpy(&got, d_B + idx, sizeof(DATA_TYPE), hipMemcpyDeviceToHost));
        if (got != h_A[idx]) {
            fprintf(stderr, "stream: verification failed at element %zu: B=%g, expected %g\n",
                    idx, (double)got, (double)h_A[idx]);
            ok = false;
        }
    }

    // ---- results; GB = 10^9 byte ----
    const double a_gflop = 0.0;                                                 // no arithmetic
    const double a_gbyte = (double)launches * n * 2 * sizeof(DATA_TYPE) / 1e9;  // 1 load + 1 store
    const double a_intensity = a_gflop / a_gbyte;                               // 0

    fprintf(stderr, "stream: elements=%zu  %ld launches  %.3f s  %.1f GB/s\n",
            n, launches, elapsed, a_gbyte / elapsed);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"e_elements\": %zu, \"e_working_set_bytes\": %zu, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           ok ? "true" : "false",
           n, n * 2 * sizeof(DATA_TYPE), blocks, threads);
    fflush(stdout);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    return ok ? 0 : 3;
}
