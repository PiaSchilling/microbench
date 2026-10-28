// stream.hip -- minimaler Stream-Kernel (Copy: B[i] = A[i])
//
// Bauen:   hipcc -O3 --offload-arch=gfx942 stream.hip -o stream
// Starten: ./stream --duration 30 --elements 134217728
//          (elements muss durch 256 teilbar sein; 2^27 doubles = 1 GiB pro Array)

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define DATA_TYPE double

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP-Error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__global__ void copy(const DATA_TYPE* A, DATA_TYPE* B) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    B[i] = A[i];
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int threads = 256;
    double duration = 30.0;
    size_t elements = 1UL << 27;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "Argument without value: %s\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "--duration")) duration = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--elements")) elements = (size_t)atoll(argv[i + 1]);
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }
    if (duration <= 0) { fprintf(stderr, "--duration must be > 0 \n"); return 1; }
    if (elements == 0 || elements % threads != 0) {
        fprintf(stderr, "--elements must be > 0 and divisible by %d \n", threads);
        return 1;
    }

    const size_t blocks = elements / threads;   // ein Element pro Thread
    const int batch = 100;                      // Starts pro Synchronisation

    DATA_TYPE *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, elements * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_B, elements * sizeof(DATA_TYPE)));
    
    std::vector<DATA_TYPE> h_A(elements);
    for (size_t i = 0; i < elements; ++i) h_A[i] = (DATA_TYPE)i;
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), elements * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B, 0, elements * sizeof(DATA_TYPE)));   // B startet bei 0

    // Warmup (nicht gezaehlt)
    copy<<<blocks, threads>>>(d_A, d_B);
    HIP_CHECK(hipDeviceSynchronize());

    long launches = 0;
    double t0 = now_s(), elapsed = 0;
    do {
        for (int b = 0; b < batch; ++b) copy<<<blocks, threads>>>(d_A, d_B);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        launches += batch;
        elapsed = now_s() - t0;
    } while (elapsed < duration);

    // Verifikation (nach der Messung): Stichprobe, B[idx] muss A[idx] entsprechen
    const size_t samples = (elements < 1024) ? elements : 1024;
    bool ok = true;
    for (size_t k = 0; k < samples && ok; ++k) {
        size_t idx = (elements - 1) / (samples - 1) * k;
        DATA_TYPE got;
        HIP_CHECK(hipMemcpy(&got, d_B + idx, sizeof(DATA_TYPE), hipMemcpyDeviceToHost));
        if (got != h_A[idx]) {
            fprintf(stderr, "Verifikation fehlgeschlagen bei Element %zu: B=%g, erwartet %g\n",
                    idx, (double)got, (double)h_A[idx]);
            ok = false;
        }
    }

    // Analytische Groessen (aus Formeln, nicht gemessen); GB = 10^9 Byte
    double a_gflop = 0.0;                                              // Stream rechnet nicht
    double a_gbyte = (double)launches * elements * 2 * sizeof(DATA_TYPE) / 1e9;  // 1 Load + 1 Store
    double a_intensity = a_gflop / a_gbyte;                            // Flop/Byte (hier 0)

    // Abgeleitete Raten
    double d_gflop_per_s = a_gflop / elapsed;
    double d_gbyte_per_s = a_gbyte / elapsed;

    fprintf(stderr, "elements=%zu  %ld Starts  %.2f s  %.1f GB/s\n",
            elements, launches, elapsed, d_gbyte_per_s);
    printf("{\"runtime_s\": %.6f, \"iterations\": %ld, "
           "\"a_gflop\": %.6f, \"a_gbyte\": %.6f, \"a_intensity\": %.6f, "
           "\"d_gflop_per_s\": %.3f, \"d_gbyte_per_s\": %.3f, \"verify_ok\": %s}\n",
           elapsed, launches, a_gflop, a_gbyte, a_intensity, d_gflop_per_s, d_gbyte_per_s, ok ? "true" : "false");
    return ok ? 0 : 3;
}
