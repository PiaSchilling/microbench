// gemm.cpp -- naive Matrixmultiplikation C = A * B (Proxy-Kernel GEMM)
//
// Bauen:   hipcc -O3 -std=c++17 --offload-arch=gfx942 gemm.cpp -o gemm
// Starten: ./gemm --duration 30 --precision fp64 --size 4096
//          ./gemm --duration 30 --precision fp16_acc32 --m 8192 --n 8192 --k 512
//
// Argumente (alle optional):
//   --duration <s>       Ziel-Laufzeit der Zeitschleife                  (30)
//   --precision <p>      Datentypen, siehe Tabelle                       (fp64)
//   --size <n>           setzt M = N = K = n                             (4096)
//   --m / --n / --k <n>  einzelne Dimensionen, C[MxN] = A[MxK] * B[KxN]
//
//   precision    Eingabe   Akkumulation   Ausgabe
//   fp64         fp64      fp64           fp64
//   fp32         fp32      fp32           fp32
//   fp16         fp16      fp16           fp16
//   fp16_acc32   fp16      fp32           fp16
//   int8         int8      int32          int32
//   int16        int16     int32          int32
//   int32        int32     int32          int32
//   int64        int64     int64          int64
//
// Ein Thread pro Element von C, row-major, ohne Tiling oder LDS. Ueber die Form (M, N, K)
// laesst sich die arithmetische Intensitaet variieren.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

constexpr int THREADS = 256;

// T = Eingabetyp (A, B), AccT = Akkumulationstyp, CT = Ausgabetyp (C)
template <typename T, typename AccT, typename CT>
__global__ void matmul(const T* A, const T* B, CT* C, int M, int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * N) return;
    size_t row = idx / N;
    size_t col = idx % N;

    AccT sum = 0;
    for (int k = 0; k < K; k++)
        sum += (AccT)A[row * K + k] * (AccT)B[(size_t)k * N + col];
    C[idx] = (CT)sum;
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Zufallswerte mit festem Seed. Floats: [-1, 1]. Integer: [-8, 8], damit auch bei grossem K
// kein Ueberlauf im Akkumulator entsteht (|Summe| <= 64*K).
template <typename T>
static void init_random(std::vector<T>& v, unsigned seed) {
    std::mt19937 gen(seed);
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<int> d(-8, 8);
        for (auto& x : v) x = (T)d(gen);
    } else {
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (auto& x : v) x = (T)d(gen);
    }
}

// Rundungsfehlereinheit u = eps/2 (0 fuer Integer: dort muss das Ergebnis exakt sein)
template <typename T>
static constexpr double unit_roundoff() {
    if constexpr (std::is_integral<T>::value)        return 0.0;
    else if constexpr (std::is_same<T, double>::value) return 0x1p-53;
    else if constexpr (std::is_same<T, float>::value)  return 0x1p-24;
    else                                               return 0x1p-11;   // _Float16
}

// Stichprobe von C gegen eine fp64-Referenz (auf den gespeicherten Eingaben gerechnet).
// Fehler je Element: |C_gpu - C_ref| / sum_k |a_ik * b_kj|. Die Ecken (0,0) und (M-1,N-1)
// sind immer dabei, um Indexfehler am Rand zu finden. NaN/Inf ergibt +Inf.
template <typename T, typename CT>
static double verify(const std::vector<T>& A, const std::vector<T>& B, const std::vector<CT>& C,
                     int M, int N, int K, int samples = 64) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> ri(0, M - 1), ci(0, N - 1);
    double max_err = 0.0;
    for (int s = 0; s < samples; s++) {
        size_t i, j;
        if      (s == 0) { i = 0;     j = 0; }
        else if (s == 1) { i = M - 1; j = N - 1; }
        else             { i = ri(gen); j = ci(gen); }

        double ref = 0.0, abs_sum = 0.0;
        for (int k = 0; k < K; k++) {
            double p = (double)A[i * K + k] * (double)B[(size_t)k * N + j];
            ref += p;
            abs_sum += std::fabs(p);
        }
        double err = std::fabs((double)C[i * N + j] - ref) / std::max(abs_sum, 1e-30);
        if (!std::isfinite(err)) return INFINITY;
        max_err = std::max(max_err, err);
    }
    return max_err;
}

// Zahl als JSON (nicht-endliche Werte sind in JSON nicht erlaubt -> null)
static void json_num(char* buf, size_t n, double v, const char* fmt) {
    if (std::isfinite(v)) snprintf(buf, n, fmt, v);
    else                  snprintf(buf, n, "null");
}

template <typename T, typename AccT, typename CT>
static int run(double duration, int M, int N, int K,
               const char* precision, const char* in_s, const char* acc_s, const char* out_s) {
    constexpr bool is_int = std::is_integral<AccT>::value;
    const char* op_type = is_int ? "iop" : "flop";
    const char* unit    = is_int ? "GIop/s" : "GFlop/s";

    const size_t MN    = (size_t)M * N;
    const size_t a_el  = (size_t)M * K;
    const size_t b_el  = (size_t)K * N;
    const size_t blocks = (MN + THREADS - 1) / THREADS;
    if (blocks > UINT_MAX) { fprintf(stderr, "M*N too big for the grid\n"); return 1; }
    fprintf(stderr, "precision=%s M=%d N=%d K=%d blocks=%zu threads=%d\n",
            precision, M, N, K, blocks, THREADS);

    std::vector<T>  h_A(a_el), h_B(b_el);
    std::vector<CT> h_C(MN);
    init_random(h_A, 1);
    init_random(h_B, 2);

    T *d_A, *d_B;
    CT* d_C;
    HIP_CHECK(hipMalloc(&d_A, a_el * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_B, b_el * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_C, MN * sizeof(CT)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), a_el * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), b_el * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0xFF, MN * sizeof(CT)));   // ungeschriebene Elemente fallen auf

    auto launch = [&]() {
        hipLaunchKernelGGL((matmul<T, AccT, CT>), dim3((unsigned)blocks), dim3(THREADS), 0, 0,
                           d_A, d_B, d_C, M, N, K);
    };

    // Warmup (nicht gezaehlt), danach ein Start zur Kalibrierung: so viele Starts pro
    // Synchronisation, dass ein Batch ca. 0.1 s dauert (wichtig bei kleinen Matrizen)
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    double t_one = now_s();
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    t_one = now_s() - t_one;
    const long batch = std::clamp((long)(0.1 / std::max(t_one, 1e-9)), 1L, 1000L);

    // Zeitschleife: Batches starten, bis duration erreicht ist
    long launches = 0;
    double t0 = now_s(), elapsed = 0;
    do {
        for (long b = 0; b < batch; ++b) launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        launches += batch;
        elapsed = now_s() - t0;
    } while (elapsed < duration);

    // Verifikation (nach der Messung)
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, MN * sizeof(CT), hipMemcpyDeviceToHost));
    const double max_rel_err = verify(h_A, h_B, h_C, M, N, K);
    // Toleranz: statistische Schranke fuer die Akkumulation ueber K Terme plus Rundung der
    // Ausgabe. Integer muessen exakt sein.
    const double tol = 4.0 * unit_roundoff<AccT>() * std::sqrt((double)K) + unit_roundoff<CT>();
    const bool ok = (max_rel_err <= tol);
    if (!ok) fprintf(stderr, "Verifikation fehlgeschlagen: max_rel_err=%.3e > Toleranz %.3e\n",
                     max_rel_err, tol);

    // Analytische Groessen; GB = 10^9 Byte. a_gbyte ist der minimale Verkehr (A, B, C je
    // einmal). Der naive Kernel liest A und B tatsaechlich mehrfach (ueber Caches).
    const double flop_per_launch = 2.0 * M * (double)N * K;
    const double byte_per_launch = (double)(a_el + b_el) * sizeof(T) + (double)MN * sizeof(CT);
    const double a_gflop = (double)launches * flop_per_launch / 1e9;
    const double a_gbyte = (double)launches * byte_per_launch / 1e9;
    const double a_intensity = flop_per_launch / byte_per_launch;
    const double d_gflop_per_s = a_gflop / elapsed;
    const double d_gbyte_per_s = a_gbyte / elapsed;

    fprintf(stderr, "precision=%s M=%d N=%d K=%d  %ld Starts  %.2f s  %.1f %s  max_rel_err=%.2e\n",
            precision, M, N, K, launches, elapsed, d_gflop_per_s, unit, max_rel_err);

    char err_s[32], tol_s[32];
    json_num(err_s, sizeof err_s, max_rel_err, "%.3e");
    json_num(tol_s, sizeof tol_s, tol, "%.3e");
    printf("{\"runtime_s\": %.6f, \"iterations\": %ld, "
           "\"a_gflop\": %.6f, \"a_gbyte\": %.6f, \"a_intensity\": %.6f, "
           "\"d_gflop_per_s\": %.3f, \"d_gbyte_per_s\": %.3f, \"verify_ok\": %s, "
           "\"precision\": \"%s\", \"op_type\": \"%s\", "
           "\"input_precision\": \"%s\", \"acc_precision\": \"%s\", \"output_precision\": \"%s\", "
           "\"m\": %d, \"n\": %d, \"k\": %d, \"max_rel_err\": %s, \"tolerance\": %s}\n",
           elapsed, launches, a_gflop, a_gbyte, a_intensity,
           d_gflop_per_s, d_gbyte_per_s, ok ? "true" : "false",
           precision, op_type, in_s, acc_s, out_s, M, N, K, err_s, tol_s);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    return ok ? 0 : 3;
}

// Dimension parsen: ganze Zahl in [1, INT_MAX], sonst Abbruch
static int parse_dim(const char* name, const char* s) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 1 || v > INT_MAX) {
        fprintf(stderr, "%s must be an integer in [1, %d]: %s\n", name, INT_MAX, s);
        exit(1);
    }
    return (int)v;
}

int main(int argc, char** argv) {
    double duration = 30.0;
    const char* precision = "fp64";
    int M = 4096, N = 4096, K = 4096;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "Argument without value: %s\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "--duration"))  duration = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--precision")) precision = argv[i + 1];
        else if (!strcmp(argv[i], "--size"))      M = N = K = parse_dim("--size", argv[i + 1]);
        else if (!strcmp(argv[i], "--m"))         M = parse_dim("--m", argv[i + 1]);
        else if (!strcmp(argv[i], "--n"))         N = parse_dim("--n", argv[i + 1]);
        else if (!strcmp(argv[i], "--k"))         K = parse_dim("--k", argv[i + 1]);
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }
    if (duration <= 0) { fprintf(stderr, "--duration must be > 0 \n"); return 1; }

#define RUN(T, ACC, CT, IN_S, ACC_S, OUT_S) \
    return run<T, ACC, CT>(duration, M, N, K, precision, IN_S, ACC_S, OUT_S)

    if      (!strcmp(precision, "fp64"))       RUN(double,   double,   double,   "fp64",  "fp64",  "fp64");
    else if (!strcmp(precision, "fp32"))       RUN(float,    float,    float,    "fp32",  "fp32",  "fp32");
    else if (!strcmp(precision, "fp16"))       RUN(_Float16, _Float16, _Float16, "fp16",  "fp16",  "fp16");
    else if (!strcmp(precision, "fp16_acc32")) RUN(_Float16, float,    _Float16, "fp16",  "fp32",  "fp16");
    else if (!strcmp(precision, "int8"))       RUN(int8_t,   int32_t,  int32_t,  "int8",  "int32", "int32");
    else if (!strcmp(precision, "int16"))      RUN(int16_t,  int32_t,  int32_t,  "int16", "int32", "int32");
    else if (!strcmp(precision, "int32"))      RUN(int32_t,  int32_t,  int32_t,  "int32", "int32", "int32");
    else if (!strcmp(precision, "int64"))      RUN(int64_t,  int64_t,  int64_t,  "int64", "int64", "int64");
#undef RUN

    fprintf(stderr, "--precision '%s' not supported "
                    "(fp64, fp32, fp16, fp16_acc32, int8, int16, int32, int64)\n", precision);
    return 1;
}
