// chase.cpp -- minimaler Pointer-Chasing-Kernel (Latenz)
//
// Ein Thread folgt einer zufaelligen Kette: j = next[j]. Jeder Zugriff haengt vom
// vorigen ab, es kann also nichts parallel oder vorausgeladen werden.
//
// Bauen:   hipcc -O3 -std=c++17 --offload-arch=gfx942 chase.cpp -o chase
// Starten: ./chase --duration 10 --working_set_bytes 1048576 --accesses 1000000
// ISA:     hipcc -O3 --offload-arch=gfx942 --save-temps -c chase.cpp
//          grep -E "global_load|s_load" chase-hip-amdgcn-amd-amdhsa-gfx942.s
//
// Argumente (alle optional):
//   --duration <s>             Ziel-Laufzeit                       (30)
//   --working_set_bytes <b>    Groesse der Kette in Byte           (1048576)
//   --stride_bytes <b>         Abstand zwischen Knoten, durch 4 teilbar (128)
//   --accesses <n>             Zugriffe pro Kernel-Start           (1000000)

#include <hip/hip_runtime.h>
#include <chrono>
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
            fprintf(stderr, "HIP-Error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__global__ void chase(const unsigned int* next, unsigned int* out, long accesses) {
    unsigned int j = 0;
    for (long i = 0; i < accesses; i++)
        j = next[j + threadIdx.x];   // threadIdx.x ist hier immer 0; es verhindert,
                                     // dass der Compiler den Zugriff als "uniform"
                                     // erkennt und ueber den Skalar-Cache (s_load) leitet
    out[0] = j;                      // verhindert Dead-Code-Elimination
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    double duration = 30.0;
    size_t working_set = 1UL << 20;
    size_t stride = 128;
    long accesses = 1000000;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "Argument without value: %s\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "--duration"))           duration = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--working_set_bytes"))  working_set = (size_t)atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "--stride_bytes"))       stride = (size_t)atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "--accesses"))           accesses = atol(argv[i + 1]);
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return 1; }
    }

    if (duration <= 0)  { fprintf(stderr, "--duration must be > 0 \n"); return 1; }
    if (accesses <= 0)  { fprintf(stderr, "--accesses must be > 0 \n"); return 1; }
    if (stride == 0 || stride % 4 != 0) {
        fprintf(stderr, "--stride_bytes must be > 0 and divisible by 4 \n"); return 1;
    }
    const size_t nodes = working_set / stride;
    const size_t se = stride / 4;
    if (nodes < 2) { fprintf(stderr, "working_set_bytes must at least be 2 * stride_bytes\n"); return 1; }
    if ((unsigned long long)nodes * se > 0xFFFFFFFFull) {
        fprintf(stderr, "Working Set too big for 32-Bit-Indices\n"); return 1;
    }
    if ((long)nodes > accesses)
        fprintf(stderr, "Warning: --accesses (%ld) < Nodes (%zu): the chain will not be run through completely, "
                        "the effective working set is smaller than stated (possibly the wrong cache level is targeted)\n", accesses, nodes);

    // Zufaelliger Zyklus ueber alle Knoten (Sattolo): jeder Knoten wird genau einmal besucht,
    // bevor die Kette wieder von vorn beginnt. Zufaellig, damit kein Prefetcher helfen kann.
    std::vector<unsigned int> p(nodes);
    std::iota(p.begin(), p.end(), 0u);
    std::mt19937_64 rng(12345);
    for (size_t i = nodes - 1; i >= 1; --i) {
        size_t j = rng() % i;                    // j in [0, i-1]
        std::swap(p[i], p[j]);
    }
    std::vector<unsigned int> h(nodes * se, 0);
    for (size_t i = 0; i < nodes; ++i) h[i * se] = (unsigned int)(p[i] * se);

    // Verification: Ein Zyklus über alle Knoten?
    size_t len = 0;
    unsigned int pos = 0;
    do { pos = h[pos]; ++len; } while (pos != 0 && len <= nodes);
    const bool chain_ok = (len == nodes);
    if (!chain_ok) { fprintf(stderr, "Kette ist kein einzelner Zyklus über alle Knoten\n"); return 1; }

    unsigned int *d_next, *d_out;
    HIP_CHECK(hipMalloc(&d_next, h.size() * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_next, h.data(), h.size() * sizeof(unsigned int), hipMemcpyHostToDevice));

    // Warmup (nicht gezaehlt): laedt Code und Daten
    chase<<<1, 1>>>(d_next, d_out, accesses);
    HIP_CHECK(hipDeviceSynchronize());

    long launches = 0;
    double t0 = now_s(), elapsed = 0;
    do {
        chase<<<1, 1>>>(d_next, d_out, accesses);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        elapsed = now_s() - t0;
    } while (elapsed < duration);

    // Verification
    unsigned int expected = 0;
    for (long k = 0; k < accesses % (long)nodes; ++k) expected = h[expected];
    unsigned int got = 0;
    HIP_CHECK(hipMemcpy(&got, d_out, sizeof(unsigned int), hipMemcpyDeviceToHost));
    const bool ok = (got == expected);
    if (!ok) fprintf(stderr, "Verifikation fehlgeschlagen: GPU=%u, erwartet %u\n", got, expected);

    const double total_accesses = (double)launches * accesses;
    const double a_gflop = 0.0;
    const double a_gbyte = total_accesses * sizeof(unsigned int) / 1e9;   // logisch: 4 Byte je Zugriff
    const double a_intensity = 0.0;
    const double d_gflop_per_s = 0.0;
    const double d_gbyte_per_s = a_gbyte / elapsed;
    const double d_ns_per_access = elapsed / total_accesses * 1e9;

    fprintf(stderr, "working_set=%zu B  stride=%zu B  nodes=%zu  %ld Starts  %.2f s  %.1f ns/Zugriff\n",
            nodes * stride, stride, nodes, launches, elapsed, d_ns_per_access);
    printf("{\"runtime_s\": %.6f, \"iterations\": %ld, "
           "\"a_gflop\": %.6f, \"a_gbyte\": %.9f, \"a_intensity\": %.6f, "
           "\"d_gflop_per_s\": %.3f, \"d_gbyte_per_s\": %.6f, \"verify_ok\": %s, "
           "\"a_accesses\": %.0f, \"d_ns_per_access\": %.3f}\n",
           elapsed, launches, a_gflop, a_gbyte, a_intensity,
           d_gflop_per_s, d_gbyte_per_s, ok ? "true" : "false",
           total_accesses, d_ns_per_access);

    return ok ? 0 : 3;
}
