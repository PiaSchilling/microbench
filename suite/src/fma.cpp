// fma.cpp -- Guerreiros Kernel, 1:1 nach HIP, mit Zeitschleife und waehlbarem Datentyp
//
// Bauen:   hipcc -O3 -std=c++17 -ffp-contract=fast --offload-arch=gfx942 fma.cpp -o fma
// Starten: ./fma --duration 30 --ops_per_element 1000000 --precision fp64
//
// --precision: fp64 | fp32 | fp16 | int64 | int32 | int16 | int8   (Standard: fp64)
//   Integer-Typen sind vorzeichenlos (unsigned): Ueberlauf ist dann definiert (Wrap-around).
//   bf16 ist bewusst nicht enthalten (siehe Doku: keine native VALU-Arithmetik auf gfx942).

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP-Error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Multiply-Add. Allgemein a*b+c (mit -ffp-contract=fast wird daraus bei Floats eine FMA).
template <typename T>
__device__ __forceinline__ T mad(T a, T b, T c) { return a * b + c; }

// 16-Bit-Integer werden in C++ zu (vorzeichenbehaftetem) int promoviert. 65535*65535
// passt nicht in int -> undefiniertes Verhalten. Deshalb explizit in unsigned rechnen.
template <>
__device__ __forceinline__ uint16_t mad<uint16_t>(uint16_t a, uint16_t b, uint16_t c) {
    return (uint16_t)((unsigned)a * (unsigned)b + (unsigned)c);
}

template <typename T>
__global__ void arith(const T* A, T* B, long N) {
    size_t threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    T r0, r1, r2, r3;
    r0 = A[threadId];
    r1 = r0 + (T)1;
    r2 = r0 + (T)2;
    r3 = r0 + (T)3;

#pragma unroll 4
    for (long i = 0; i < N; i++) {
        r0 = mad(r0, r0, r0);
        r1 = mad(r1, r1, r1);
        r2 = mad(r2, r2, r2);
        r3 = mad(r3, r3, r3);
    }

    B[threadId] = r0 + r1 + r2 + r3;
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Eingabewerte. Floats: 0.5 ... 1.5. Integer: pseudozufaellige Werte
// (ein Cast von 0.5...1.5 auf Integer ergaebe nur 0 und 1).
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
    const int threads = 256;
    const int blocks = 8 * prop.multiProcessorCount;
    const size_t total = (size_t)blocks * threads;
    fprintf(stderr, "precision=%s blocks=%d threads=%d\n", precision, blocks, threads);

    T* h_A = (T*)malloc(total * sizeof(T));
    if (!h_A) { fprintf(stderr, "Host memory allocation failed\n"); return 1; }
    for (size_t i = 0; i < total; ++i) h_A[i] = make_value<T>(i);

    T *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, total * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_B, total * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_A, h_A, total * sizeof(T), hipMemcpyHostToDevice));

    // Warmup (nicht gezaehlt)
    arith<T><<<blocks, threads>>>(d_A, d_B, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Zeitschleife: Kernel wiederholt starten, bis duration erreicht ist
    long launches = 0;
    double t0 = now_s(), elapsed = 0;
    do {
        arith<T><<<blocks, threads>>>(d_A, d_B, N);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        elapsed = now_s() - t0;
    } while (elapsed < duration);

    // 4 Multiply-Adds pro Durchlauf, je 2 Operationen (mul + add).
    // Bei Integer sind das nominale Operationen, nicht zwingend 2 Instruktionen.
    const double a_gflop = (double)launches * total * N * 8 / 1e9;
    const double a_gbyte = (double)launches * total * 2 * sizeof(T) / 1e9;   // 1 Load + 1 Store
    const double a_intensity = a_gflop / a_gbyte;
    const double d_gflop_per_s = a_gflop / elapsed;
    const double d_gbyte_per_s = a_gbyte / elapsed;

    fprintf(stderr, "precision=%s N=%ld  %ld Starts  %.2f s  %.1f %s\n",
            precision, N, launches, elapsed, d_gflop_per_s, unit);
    printf("{\"runtime_s\": %.6f, \"iterations\": %ld, "
           "\"a_gflop\": %.6f, \"a_gbyte\": %.9f, \"a_intensity\": %.6f, "
           "\"d_gflop_per_s\": %.3f, \"d_gbyte_per_s\": %.6f, \"verify_ok\": null, "
           "\"precision\": \"%s\", \"op_type\": \"%s\"}\n",
           elapsed, launches, a_gflop, a_gbyte, a_intensity,
           d_gflop_per_s, d_gbyte_per_s, precision, op_type);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_A);
    return 0;
}

int main(int argc, char** argv) {
    double duration = 30.0;
    long N = 1000000;                           // = ops_per_element
    const char* precision = "fp64";
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "Argument without value: %s\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "--duration"))        duration = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--ops_per_element")) N = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--precision"))       precision = argv[i + 1];
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }
    if (duration <= 0) { fprintf(stderr, "--duration must be > 0 \n"); return 1; }
    if (N < 1)         { fprintf(stderr, "--ops_per_element must be >= 1 \n"); return 1; }

    if      (!strcmp(precision, "fp64"))  return run<double>(duration, N, "fp64");
    else if (!strcmp(precision, "fp32"))  return run<float>(duration, N, "fp32");
    else if (!strcmp(precision, "fp16"))  return run<_Float16>(duration, N, "fp16");
    else if (!strcmp(precision, "int64")) return run<uint64_t>(duration, N, "int64");
    else if (!strcmp(precision, "int32")) return run<uint32_t>(duration, N, "int32");
    else if (!strcmp(precision, "int16")) return run<uint16_t>(duration, N, "int16");
    else if (!strcmp(precision, "int8"))  return run<uint8_t>(duration, N, "int8");
    fprintf(stderr, "--precision '%s' not supported (fp64, fp32, fp16, int64, int32, int16, int8)\n",
            precision);
    return 1;
}
