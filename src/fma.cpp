// ---- benchmark metadata ----
// benchmark=fma
// level=1
// component=GPU/Compute/VALU
// description=Four independent FMA chains per thread with bounded, changing values; compute pole of the design space
// tags=compute-bound;valu
// version=0.5.0
// precisions=fp64,fp32,fp16,int64,int32,int16,int8
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.ops_per_element=1000000|>= 1|Multiply-adds per chain and thread; controls arithmetic intensity
// param.precision=fp64|see precisions|Data type of the computation (integers are unsigned)
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps)
// # before they are enabled: the compiler may emit v_fmac_* instead of v_fma_*.
// # expect_isa.fp64=v_fma_f64
// # expect_isa.fp32=v_fma_f32
// expect_counters=
// ---- end metadata ----
//
// Kernel: based on Guerreiro et al. (HPCA 2018, computeKernel.cu), ported to HIP.
// Each thread loads one value, runs four independent chains of ops_per_element
// multiply-adds and stores the sum. Interface: docs/benchmark_contract.md.
//
// Floating point (since 0.5.0): the original chain r = r*r + r overflows to inf
// after 7 (fp16), 10 (fp32) or 13 (fp64) iterations; afterwards the same value
// repeats, hardly any bits toggle and the power draw is far too low although the
// VALU is fully busy (docs: validation findings 6 and 12). Now each chain
// alternates between two FMAs that undo each other:
//     r = r * a  + b        (a = 1.3, b = 0.7)
//     r = r * ai + bi       (ai = 1/a, bi = -b/a)
// The values stay bounded and change in every step (they jump between two
// values per chain). a, b, ai, bi are kernel arguments, so the compiler cannot
// fold the two FMAs at compile time.
//
// Integer types: unchanged, r = r*r + r. They are unsigned, so overflow is
// defined (wrap-around) and the values keep changing.
// bf16 is not included (no native VALU arithmetic on gfx942).

#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Multiply-add a*b+c (with -ffp-contract=fast this becomes an FMA for floats).
template <typename T>
__device__ __forceinline__ T mad(T a, T b, T c) { return a * b + c; }

// 16-bit integers are promoted to (signed) int in C++; 65535*65535 does not fit
// into int -> undefined behaviour. Therefore compute explicitly in unsigned.
template <>
__device__ __forceinline__ uint16_t mad<uint16_t>(uint16_t a, uint16_t b, uint16_t c) {
    return (uint16_t)((unsigned)a * (unsigned)b + (unsigned)c);
}

// N multiply-adds per chain in total, as before (accounting: 8 operations per
// iteration and thread). a, b, ai, bi are only used for floating-point types.
template <typename T>
__global__ void arith(const T* A, T* B, long N, T a, T b, T ai, T bi) {
    size_t threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    T r0, r1, r2, r3;
    r0 = A[threadId];
    r1 = r0 + (T)1;
    r2 = r0 + (T)2;
    r3 = r0 + (T)3;

    if constexpr (std::is_integral<T>::value) {
#pragma unroll 4
        for (long i = 0; i < N; i++) {
            r0 = mad(r0, r0, r0);
            r1 = mad(r1, r1, r1);
            r2 = mad(r2, r2, r2);
            r3 = mad(r3, r3, r3);
        }
    } else {
        // pairs of FMAs that undo each other: values stay bounded and change
#pragma unroll 2
        for (long i = 0; i + 1 < N; i += 2) {
            r0 = mad(r0, a, b);   r1 = mad(r1, a, b);   r2 = mad(r2, a, b);   r3 = mad(r3, a, b);
            r0 = mad(r0, ai, bi); r1 = mad(r1, ai, bi); r2 = mad(r2, ai, bi); r3 = mad(r3, ai, bi);
        }
        if (N & 1) {              // odd N: one more step, so the total stays N
            r0 = mad(r0, a, b);   r1 = mad(r1, a, b);   r2 = mad(r2, a, b);   r3 = mad(r3, a, b);
        }
    }

    B[threadId] = r0 + r1 + r2 + r3;
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

// Input values. Floats: 0.5 ... 1.5. Integers: pseudo-random values
// (casting 0.5...1.5 to an integer would give only 0 and 1).
template <typename T>
static T make_value(size_t i) {
    if constexpr (std::is_integral<T>::value) {
        uint64_t h = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 32;
        return (T)h;
    } else {
        return (T)(0.5 + (double)(i % 1000) / 1000.0);
    }
}

template <typename T>
static int run(double duration, long N, const char* precision) {
    constexpr bool is_int = std::is_integral<T>::value;
    const char* op_type = is_int ? "iop" : "flop";
    const char* unit    = is_int ? "GIop/s" : "GFlop/s";

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    const int threads = 256;
    const int blocks = 8 * prop.multiProcessorCount;
    const size_t total = (size_t)blocks * threads;
    fprintf(stderr, "fma: device %s (%s), precision=%s blocks=%d threads=%d\n",
            prop.name, pci, precision, blocks, threads);

    T* h_A = (T*)malloc(total * sizeof(T));
    if (!h_A) { fprintf(stderr, "host memory allocation failed\n"); return 1; }
    for (size_t i = 0; i < total; ++i) h_A[i] = make_value<T>(i);

    T *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, total * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_B, total * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_A, h_A, total * sizeof(T), hipMemcpyHostToDevice));

    // Constants of the FMA pair (floating point only), computed in double and
    // converted to T; passed as kernel arguments (see header).
    const double a_d = 1.3, b_d = 0.7;
    const T a = (T)a_d, b = (T)b_d, ai = (T)(1.0 / a_d), bi = (T)(-b_d / a_d);

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(arith<T>, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, N, a, b, ai, bi);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Timed loop = ROI: launch until duration is reached, at least once.
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(arith<T>, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, N, a, b, ai, bi);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // Verification (after the ROI): floating point results must be finite.
    // Integers: no verification (wrap-around is intended).
    const char* verify = "null";
    if constexpr (!is_int) {
        T* h_B = (T*)malloc(total * sizeof(T));
        if (!h_B) { fprintf(stderr, "host memory allocation failed\n"); return 1; }
        HIP_CHECK(hipMemcpy(h_B, d_B, total * sizeof(T), hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < total; ++i) bad += !std::isfinite((double)h_B[i]);
        free(h_B);
        verify = bad ? "false" : "true";
        if (bad) fprintf(stderr, "fma: verification failed: %zu of %zu results not finite\n", bad, total);
    }

    // 4 multiply-adds per iteration, 2 operations each (mul + add).
    // For integers these are nominal operations, not necessarily 2 instructions.
    const double a_gflop = (double)launches * total * N * 8 / 1e9;
    const double a_gbyte = (double)launches * total * 2 * sizeof(T) / 1e9;   // 1 load + 1 store
    const double a_intensity = a_gflop / a_gbyte;

    fprintf(stderr, "fma: precision=%s ops_per_element=%ld  %ld launches  %.3f s  %.1f %s\n",
            precision, N, launches, elapsed, a_gflop / elapsed, unit);

    // Exactly one line on stdout (contract section 5)
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": %s, "
           "\"precision\": \"%s\", \"op_type\": \"%s\", "
           "\"e_blocks\": %d, \"e_threads\": %d, \"e_ops_per_element\": %ld}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           verify, precision, op_type, blocks, threads, N);
    fflush(stdout);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_A);
    return strcmp(verify, "false") ? 0 : 3;    // contract section 7: 3 = verification failed
}

// ---------------------------------------------------------------------------
// Arguments (contract section 4)
// ---------------------------------------------------------------------------

static const char* const PRECISIONS[] = {"fp64", "fp32", "fp16", "int64", "int32", "int16", "int8"};

static bool parse_double(const char* s, double* out) {
    char* end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

static bool parse_long(const char* s, long* out) {
    char* end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0') return false;
    *out = v;
    return true;
}

int main(int argc, char** argv) {
    double duration = 0;
    long N = 1000000;                       // = ops_per_element
    const char* precision = "fp64";
    bool check = false, have_duration = false, have_ops = false, have_prec = false;

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "fma: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "fma: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "fma: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--ops_per_element")) {
            if (have_ops) { fprintf(stderr, "fma: --ops_per_element given twice\n"); return 1; }
            have_ops = true;
            if (!parse_long(val, &N) || N < 1) {
                fprintf(stderr, "fma: --ops_per_element must be an integer >= 1 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--precision")) {
            if (have_prec) { fprintf(stderr, "fma: --precision given twice\n"); return 1; }
            have_prec = true;
            precision = val;
        } else {
            fprintf(stderr, "fma: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "fma: --duration is required\n"); return 1; }

    bool known = false;
    for (const char* p : PRECISIONS) known = known || !strcmp(precision, p);
    if (!known) {
        fprintf(stderr, "fma: --precision '%s' is not valid (fp64, fp32, fp16, int64, int32, int16, int8)\n",
                precision);
        return 1;
    }

    if (check) return 0;                    // arguments valid, no GPU access

    if (!strcmp(precision, "fp64"))  return run<double>(duration, N, "fp64");
    if (!strcmp(precision, "fp32"))  return run<float>(duration, N, "fp32");
    if (!strcmp(precision, "fp16"))  return run<_Float16>(duration, N, "fp16");
    if (!strcmp(precision, "int64")) return run<uint64_t>(duration, N, "int64");
    if (!strcmp(precision, "int32")) return run<uint32_t>(duration, N, "int32");
    if (!strcmp(precision, "int16")) return run<uint16_t>(duration, N, "int16");
    return run<uint8_t>(duration, N, "int8");
}
