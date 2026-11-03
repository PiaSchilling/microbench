// ---- benchmark metadata ----
// benchmark=gemm
// level=3
// component=GPU/Compute/VALU
// description=Matrix multiplication C = A * B as proxy kernel (blocked with LDS tiles, or naive); no matrix cores
// tags=proxy;gemm;lds
// version=0.3.0
// precisions=fp64,fp32,fp16,int8,int16,int32,int64
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.kernel=blocked|blocked, naive|blocked: 128x128 tiles in LDS; naive: one thread per element of C
// param.precision=fp64|see precisions|Input type of A and B (accumulation and output: see table in the source)
// param.accumulate=native|native, fp32 (only with fp16)|Accumulation type; native: as in the table
// param.size=4096|>= 1|Sets m = n = k (explicit --m/--n/--k take precedence)
// param.m=4096|>= 1; blocked: multiple of 128|Rows of A and C
// param.n=4096|>= 1; blocked: multiple of 128|Columns of B and C
// param.k=4096|>= 1|Columns of A, rows of B; controls the arithmetic intensity of the blocked kernel
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa.fp64=v_fma_f64
// expect_counters=
// ---- end metadata ----
//
// gemm.cpp -- matrix multiplication C = A * B (proxy kernel GEMM), blocked or naive
// Interface: docs/benchmark_contract.md.
//
//   --precision   --accumulate   input   accumulation   output
//   fp64          native         fp64    fp64           fp64
//   fp32          native         fp32    fp32           fp32
//   fp16          native         fp16    fp16           fp16
//   fp16          fp32           fp16    fp32           fp16
//   int8          native         int8    int32          int32
//   int16         native         int16   int32          int32
//   int32         native         int32   int32          int32
//   int64         native         int64   int64          int64
//
//   blocked: M and N must be multiples of 128, K is arbitrary
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
#include <cerrno>
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

using mono_clock = std::chrono::steady_clock;  // CLOCK_MONOTONIC on Linux
using unix_clock = std::chrono::system_clock;  // Unix time

static long long mono_ns(mono_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
static double unix_s(unix_clock::time_point t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
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

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    fprintf(stderr, "gemm: device %s (%s), kernel=%s precision=%s acc=%s M=%d N=%d K=%d grid=%ux%u threads=%d\n",
            prop.name, pci, kernel, precision, acc_s, M, N, K, grid.x, grid.y, TPB);

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

    // Timed loop = ROI: batches until duration is reached, at least one
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        for (long b = 0; b < batch; ++b) launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        launches += batch;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

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

    fprintf(stderr, "gemm: kernel=%s precision=%s acc=%s M=%d N=%d K=%d  %ld launches  %.3f s  %.1f %s  AI=%.2f  max_rel_err=%.2e\n",
            kernel, precision, acc_s, M, N, K, launches, elapsed, d_gflop_per_s, unit, a_intensity, max_rel_err);

    char err_s[32], tol_s[32];
    json_num(err_s, sizeof err_s, max_rel_err, "%.3e");
    json_num(tol_s, sizeof tol_s, tol, "%.3e");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, \"verify_ok\": %s, "
           "\"precision\": \"%s\", \"op_type\": \"%s\", \"kernel\": \"%s\", \"traffic_model\": \"%s\", "
           "\"input_precision\": \"%s\", \"acc_precision\": \"%s\", \"output_precision\": \"%s\", "
           "\"m\": %d, \"n\": %d, \"k\": %d, \"tile\": %d, \"max_rel_err\": %s, \"tolerance\": %s, "
           "\"e_batch\": %ld, \"e_grid_x\": %u, \"e_grid_y\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity,
           d_gflop_per_s, d_gbyte_per_s, ok ? "true" : "false",
           precision, op_type, kernel, model, in_s, acc_s, out_s, M, N, K, naive ? 1 : TILE, err_s, tol_s,
           batch, grid.x, grid.y, TPB);
    fflush(stdout);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    return ok ? 0 : 3;
}

// ---------------------------------------------------------------------------
// Arguments (contract section 4)
// ---------------------------------------------------------------------------

static bool parse_double(const char* s, double* out) {
    char* end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

// dimension: integer in [1, INT_MAX]
static bool parse_dim(const char* s, int* out) {
    char* end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < 1 || v > INT_MAX) return false;
    *out = (int)v;
    return true;
}

int main(int argc, char** argv) {
    double duration = 0;
    const char* precision = "fp64";
    const char* kernel = "blocked";
    const char* accumulate = "native";
    int size = 4096, M = 0, N = 0, K = 0;              // 0: not given, use size
    bool check = false;
    bool seen[8] = {false};
    static const char* const keys[8] = {"--duration", "--precision", "--kernel", "--accumulate",
                                        "--size", "--m", "--n", "--k"};

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        int which = -1;
        for (int w = 0; w < 8; ++w) if (!strcmp(key, keys[w])) which = w;
        if (which < 0) { fprintf(stderr, "gemm: unknown option %s\n", key); return 1; }
        if (i + 1 >= argc) { fprintf(stderr, "gemm: option %s requires a value\n", key); return 1; }
        if (seen[which]) { fprintf(stderr, "gemm: %s given twice\n", key); return 1; }
        seen[which] = true;
        const char* val = argv[++i];
        bool valid = true;
        switch (which) {
            case 0: valid = parse_double(val, &duration) && duration > 0; break;
            case 1: precision = val; break;
            case 2: kernel = val; break;
            case 3: accumulate = val; break;
            case 4: valid = parse_dim(val, &size); break;
            case 5: valid = parse_dim(val, &M); break;
            case 6: valid = parse_dim(val, &N); break;
            case 7: valid = parse_dim(val, &K); break;
        }
        if (!valid) {
            fprintf(stderr, "gemm: invalid value for %s: '%s' (see: run_suite.sh --help gemm)\n", key, val);
            return 1;
        }
    }
    if (!seen[0]) { fprintf(stderr, "gemm: --duration is required\n"); return 1; }
    if (M == 0) M = size;
    if (N == 0) N = size;
    if (K == 0) K = size;

    bool naive;
    if      (!strcmp(kernel, "blocked")) naive = false;
    else if (!strcmp(kernel, "naive"))   naive = true;
    else { fprintf(stderr, "gemm: --kernel '%s' is not valid (blocked, naive)\n", kernel); return 1; }
    if (naive) {
        if (((size_t)M * N + TPB - 1) / TPB > UINT_MAX) { fprintf(stderr, "gemm: M*N too large for the grid\n"); return 1; }
    } else {
        if (M % TILE != 0 || N % TILE != 0) {
            fprintf(stderr, "gemm: --m and --n must be multiples of %d for --kernel blocked (got M=%d, N=%d)\n",
                    TILE, M, N);
            return 1;
        }
        if (M / TILE > 65535) { fprintf(stderr, "gemm: --m too large (max %d)\n", 65535 * TILE); return 1; }
    }

    const bool acc32 = !strcmp(accumulate, "fp32");
    if (!acc32 && strcmp(accumulate, "native")) {
        fprintf(stderr, "gemm: --accumulate '%s' is not valid (native, fp32)\n", accumulate);
        return 1;
    }
    if (acc32 && strcmp(precision, "fp16")) {
        fprintf(stderr, "gemm: --accumulate fp32 is only available with --precision fp16\n");
        return 1;
    }

    static const char* const precisions[] = {"fp64", "fp32", "fp16", "int8", "int16", "int32", "int64"};
    bool known = false;
    for (const char* p : precisions) known = known || !strcmp(precision, p);
    if (!known) {
        fprintf(stderr, "gemm: --precision '%s' is not valid (fp64, fp32, fp16, int8, int16, int32, int64)\n",
                precision);
        return 1;
    }
    if (check) return 0;                       // arguments valid, no GPU access

#define RUN(T, ACC, CT, IN_S, ACC_S, OUT_S) \
    return run<T, ACC, CT>(duration, naive, M, N, K, precision, IN_S, ACC_S, OUT_S)

    if      (!strcmp(precision, "fp64"))  RUN(double,   double,   double,   "fp64",  "fp64",  "fp64");
    else if (!strcmp(precision, "fp32"))  RUN(float,    float,    float,    "fp32",  "fp32",  "fp32");
    else if (!strcmp(precision, "fp16") && acc32)
                                          RUN(_Float16, float,    _Float16, "fp16",  "fp32",  "fp16");
    else if (!strcmp(precision, "fp16"))  RUN(_Float16, _Float16, _Float16, "fp16",  "fp16",  "fp16");
    else if (!strcmp(precision, "int8"))  RUN(int8_t,   int32_t,  int32_t,  "int8",  "int32", "int32");
    else if (!strcmp(precision, "int16")) RUN(int16_t,  int32_t,  int32_t,  "int16", "int32", "int32");
    else if (!strcmp(precision, "int32")) RUN(int32_t,  int32_t,  int32_t,  "int32", "int32", "int32");
    else                                  RUN(int64_t,  int64_t,  int64_t,  "int64", "int64", "int64");
#undef RUN
}
