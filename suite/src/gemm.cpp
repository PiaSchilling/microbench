// gemm.cpp -- Matrixmultiplikation C = A * B (Proxy-Kernel GEMM), geblockt oder naiv
//
// Bauen:   hipcc -O3 -std=c++17 --offload-arch=gfx942 gemm.cpp -o gemm
// Starten: ./gemm --duration 30 --precision fp64 --m 8192 --n 8192 --k 256
//          ./gemm --duration 30 --precision fp32 --size 4096 --kernel naive
//
// Argumente (alle optional):
//   --duration <s>       Ziel-Laufzeit der Zeitschleife                       (30)
//   --kernel <k>         blocked | naive                                      (blocked)
//   --precision <p>      Datentypen, siehe Tabelle                            (fp64)
//   --size <n>           setzt M = N = K = n                                  (4096)
//   --m / --n / --k <n>  einzelne Dimensionen, C[MxN] = A[MxK] * B[KxN]
//                        blocked: M und N muessen Vielfache von 128 sein, K ist beliebig
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
// --kernel blocked (Standard): Jede Workgroup (256 Threads) berechnet eine TILE x TILE-Kachel von C. Sie laedt
// dafuer ihren Zeilenstreifen von A (TILE x K) und Spaltenstreifen von B (K x TILE) in
// Stuecken von je einer Cache-Line Tiefe ins LDS, jeder Thread akkumuliert RT x RT Elemente in
// Registern. Jedes geladene Element wird so TILE-mal wiederverwendet. Keine Matrix Cores.
//
// Verkehrsmodell "kernel_level" (exakt das, was der Kernel aus dem globalen Speicher laedt/schreibt):
//   Bytes = s_in * (M*K * N/TILE + K*N * M/TILE) + s_out * M*N
//   Flop  = 2 * M*N*K
//   AI    = 2K / (s_in * 2K/TILE + s_out)          (unabhaengig von M und N)
//   fp64:  AI = 16 K / (K + 64)   -> K steuert die AI von ~0.25 (K=1) bis gegen 16
// Jede geladene Cache-Line wird voll genutzt, sofern K <= 128/s_in oder K ein Vielfaches
// davon ist (sonst liegen die Zeilen von A nicht auf Line-Grenzen). Das Modell entspricht damit dem Verkehr
// zwischen Kernel und L1/L2. Der HBM-Verkehr kann niedriger sein, wenn sich Workgroups
// Streifen von A/B ueber L2 oder Infinity Cache teilen.
//
// --kernel naive: Ein Thread pro Element von C, ohne Kacheln und LDS. Jeder Thread liest seine
// ganze Zeile von A und Spalte von B, die Wiederverwendung haengt allein von den Caches ab.
// Verkehrsmodell "minimal": A, B und C je genau einmal. Das ist eine Untergrenze fuer den
// echten Verkehr (Obergrenze fuer die AI); der tatsaechliche liegt um Groessenordnungen hoeher.

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

constexpr int TILE = 128;          // Kachelgroesse von C pro Workgroup (TILE x TILE)
constexpr int LINE = 128;          // Cache-Line in Byte (MI300A)
constexpr int TDIM = 16;           // Threads pro Workgroup: TDIM x TDIM
constexpr int TPB  = TDIM * TDIM;  // = 256
constexpr int RT   = TILE / TDIM;  // jeder Thread: RT x RT Elemente von C (= 8 x 8)

// T = Eingabetyp (A, B), AccT = Akkumulationstyp, CT = Ausgabetyp (C)
// KT = Tiefe in K pro LDS-Stueck, so gewaehlt, dass eine Zeile von A pro Stueck genau eine
// Cache-Line fuellt (fp64: 16, fp32: 32, fp16: 64, int8: 128). LDS-Bedarf: 2*LINE*TILE = 32 KiB.
template <typename T, typename AccT, typename CT>
__global__ __launch_bounds__(TPB) void gemm_blocked(const T* A, const T* B, CT* C, int M, int N, int K) {
    constexpr int KT = LINE / (int)sizeof(T);
    __shared__ T As[KT][TILE];     // A-Stueck, transponiert abgelegt: As[k][m]
    __shared__ T Bs[KT][TILE];     // B-Stueck: Bs[k][n]

    const int tid  = threadIdx.x;
    const int tx   = tid % TDIM;
    const int ty   = tid / TDIM;
    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;

    AccT acc[RT][RT];
    for (int i = 0; i < RT; ++i)
        for (int j = 0; j < RT; ++j) acc[i][j] = (AccT)0;

    for (int k0 = 0; k0 < K; k0 += KT) {
        // Jedes Element des Streifens wird genau einmal geladen; jenseits von K mit 0 auffuellen
        for (int e = tid; e < TILE * KT; e += TPB) {
            const int m = e / KT, kk = e % KT, gk = k0 + kk;
            As[kk][m] = (gk < K) ? A[(size_t)(row0 + m) * K + gk] : (T)0;
        }
        for (int e = tid; e < KT * TILE; e += TPB) {
            const int kk = e / TILE, n = e % TILE, gk = k0 + kk;
            Bs[kk][n] = (gk < K) ? B[(size_t)gk * N + col0 + n] : (T)0;
        }
        __syncthreads();

        const int kmax = (K - k0 < KT) ? (K - k0) : KT;   // kein Rechnen auf Fuellnullen
        for (int kk = 0; kk < kmax; ++kk) {
            AccT a[RT], b[RT];
            for (int i = 0; i < RT; ++i) a[i] = (AccT)As[kk][ty + TDIM * i];
            for (int j = 0; j < RT; ++j) b[j] = (AccT)Bs[kk][tx + TDIM * j];
            for (int i = 0; i < RT; ++i)
                for (int j = 0; j < RT; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < RT; ++i)
        for (int j = 0; j < RT; ++j)
            C[(size_t)(row0 + ty + TDIM * i) * N + col0 + tx + TDIM * j] = (CT)acc[i][j];
}

// Naiv: ein Thread pro Element von C, row-major
template <typename T, typename AccT, typename CT>
__global__ __launch_bounds__(TPB) void gemm_naive(const T* A, const T* B, CT* C, int M, int N, int K) {
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
static int run(double duration, bool naive, int M, int N, int K,
               const char* precision, const char* in_s, const char* acc_s, const char* out_s) {
    constexpr bool is_int = std::is_integral<AccT>::value;
    const char* op_type = is_int ? "iop" : "flop";
    const char* unit    = is_int ? "GIop/s" : "GFlop/s";

    const size_t MN   = (size_t)M * N;
    const size_t a_el = (size_t)M * K;
    const size_t b_el = (size_t)K * N;
    const char* kernel = naive ? "naive" : "blocked";
    const char* model  = naive ? "minimal" : "kernel_level";
    const dim3 grid = naive ? dim3((unsigned)((MN + TPB - 1) / TPB)) : dim3(N / TILE, M / TILE);
    fprintf(stderr, "kernel=%s precision=%s M=%d N=%d K=%d grid=%ux%u threads=%d\n",
            kernel, precision, M, N, K, grid.x, grid.y, TPB);

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
        if (naive) hipLaunchKernelGGL((gemm_naive<T, AccT, CT>),   grid, dim3(TPB), 0, 0, d_A, d_B, d_C, M, N, K);
        else       hipLaunchKernelGGL((gemm_blocked<T, AccT, CT>), grid, dim3(TPB), 0, 0, d_A, d_B, d_C, M, N, K);
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

    // Analytische Groessen nach dem jeweiligen Verkehrsmodell (siehe oben); GB = 10^9 Byte
    const double flop_per_launch = 2.0 * M * (double)N * K;
    const double reads_per_launch = naive
        ? (double)sizeof(T) * ((double)a_el + (double)b_el)                                  // minimal
        : (double)sizeof(T) * ((double)a_el * (N / TILE) + (double)b_el * (M / TILE));       // kernel_level
    const double byte_per_launch = reads_per_launch + (double)sizeof(CT) * (double)MN;
    const double a_gflop = (double)launches * flop_per_launch / 1e9;
    const double a_gbyte = (double)launches * byte_per_launch / 1e9;
    const double a_intensity = flop_per_launch / byte_per_launch;
    const double d_gflop_per_s = a_gflop / elapsed;
    const double d_gbyte_per_s = a_gbyte / elapsed;

    fprintf(stderr, "kernel=%s precision=%s M=%d N=%d K=%d  %ld Starts  %.2f s  %.1f %s  AI=%.2f  max_rel_err=%.2e\n",
            kernel, precision, M, N, K, launches, elapsed, d_gflop_per_s, unit, a_intensity, max_rel_err);

    char err_s[32], tol_s[32];
    json_num(err_s, sizeof err_s, max_rel_err, "%.3e");
    json_num(tol_s, sizeof tol_s, tol, "%.3e");
    printf("{\"runtime_s\": %.6f, \"iterations\": %ld, "
           "\"a_gflop\": %.6f, \"a_gbyte\": %.6f, \"a_intensity\": %.6f, "
           "\"d_gflop_per_s\": %.3f, \"d_gbyte_per_s\": %.3f, \"verify_ok\": %s, "
           "\"precision\": \"%s\", \"op_type\": \"%s\", \"kernel\": \"%s\", \"traffic_model\": \"%s\", "
           "\"input_precision\": \"%s\", \"acc_precision\": \"%s\", \"output_precision\": \"%s\", "
           "\"m\": %d, \"n\": %d, \"k\": %d, \"tile\": %d, \"max_rel_err\": %s, \"tolerance\": %s}\n",
           elapsed, launches, a_gflop, a_gbyte, a_intensity,
           d_gflop_per_s, d_gbyte_per_s, ok ? "true" : "false",
           precision, op_type, kernel, model, in_s, acc_s, out_s, M, N, K, naive ? 1 : TILE, err_s, tol_s);

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
    const char* kernel = "blocked";
    int M = 4096, N = 4096, K = 4096;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "Argument without value: %s\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "--duration"))  duration = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--precision")) precision = argv[i + 1];
        else if (!strcmp(argv[i], "--kernel"))    kernel = argv[i + 1];
        else if (!strcmp(argv[i], "--size"))      M = N = K = parse_dim("--size", argv[i + 1]);
        else if (!strcmp(argv[i], "--m"))         M = parse_dim("--m", argv[i + 1]);
        else if (!strcmp(argv[i], "--n"))         N = parse_dim("--n", argv[i + 1]);
        else if (!strcmp(argv[i], "--k"))         K = parse_dim("--k", argv[i + 1]);
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }
    if (duration <= 0) { fprintf(stderr, "--duration must be > 0 \n"); return 1; }
    bool naive;
    if      (!strcmp(kernel, "blocked")) naive = false;
    else if (!strcmp(kernel, "naive"))   naive = true;
    else { fprintf(stderr, "--kernel '%s' not supported (blocked, naive)\n", kernel); return 1; }
    if (naive) {
        if (((size_t)M * N + TPB - 1) / TPB > UINT_MAX) { fprintf(stderr, "M*N too big for the grid\n"); return 1; }
    } else {
        if (M % TILE != 0 || N % TILE != 0) {
            fprintf(stderr, "--m and --n must be multiples of %d for --kernel blocked (got M=%d, N=%d)\n",
                    TILE, M, N);
            return 1;
        }
        if (M / TILE > 65535) { fprintf(stderr, "--m too big (max %d)\n", 65535 * TILE); return 1; }
    }

#define RUN(T, ACC, CT, IN_S, ACC_S, OUT_S) \
    return run<T, ACC, CT>(duration, naive, M, N, K, precision, IN_S, ACC_S, OUT_S)

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
