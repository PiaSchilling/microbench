// ---- benchmark metadata ----
// benchmark=smfma
// level=1
// component=GPU/Compute/MFMA
// description=Sparse matrix-core instructions (v_smfmac, 2:4 structured sparsity in A): one dependent chain per wavefront with constant operands; compute pole of the sparse matrix cores
// tags=compute-bound;mfma;matrix-core;sparse
// version=0.1.0
// precisions=fp16,bf16,int8,fp8
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.precision=fp16|see precisions|Input type of the sparse MFMA instruction (fp16, bf16, fp8: accumulation in fp32; int8: in int32)
// param.iters=2000|>= 1|Sparse MFMA instructions per wavefront and launch (as in mfma)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa.fp16=v_smfmac_f32_32x32x16_f16
// # expect_isa.bf16=v_smfmac_f32_32x32x16_bf16
// # expect_isa.int8=v_smfmac_i32_32x32x32_i8
// # expect_isa.fp8=v_smfmac_f32_32x32x32_fp8_fp8
// expect_counters=
// ---- end metadata ----
//
// smfma: like mfma, but with the sparse matrix instructions of CDNA3 (AMD names
// them v_smfmac, "sparse matrix fused multiply-accumulate"; the benchmark is called
// smfma for consistency with mfma, it does the same D = A*B + C). A is 2:4
// structured sparse along K: of every 4 values 2 are zero; A stores only the 2
// non-zero values per group, and an index operand says where they belong. The
// hardware skips the zeros, so one instruction covers twice the K of the dense
// instruction (e.g. 32x32x16 f16 instead of 32x32x8). Every wavefront runs one
// dependent chain result = A*B + result with constant operands. Measures peak
// sparse Matrix Core throughput per precision (AMD: 2 x the dense peak).
// Source: own implementation, same pattern and launch configuration as mfma
// (rocprof-compute has no sparse kernel).
// Details: docs/benchmarks.md#smfma
// Interface: docs/benchmark_contract.md

#include <hip/hip_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s (%s:%d)\n",                         \
                    hipGetErrorString(e_), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static const int WORKGROUP_SIZE = 256;   // as in mfma / rocprof-compute
static const int BLOCKS_PER_CU = 128;    // as in mfma / rocprof-compute
static const int WAVEFRONT_SIZE = 64;

// Sparsity index (operand 4 of the builtins): for every group of 4 values of A
// along K, two 2-bit indices say which 2 positions hold the stored values.
// 0x4 = indices (0, 1): the first two of each group are non-zero. The same
// pattern for every group and lane; any valid pattern costs the same.
static const int SPARSE_INDEX = 0x44444444;

template<typename T, int Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec2 = vecT<T, 2>;
template<typename T> using vec4 = vecT<T, 4>;
template<typename T> using vec8 = vecT<T, 8>;
template<typename T> using vec16 = vecT<T, 16>;

// ---------------------------------------------------------------------------
// Kernels. Per lane: A holds half of the K values (sparse), B all of them.
//   f16/bf16 32x32x16: A = 4 values, B = 8 values per lane
//   i8/fp8   32x32x32: A = 8 values (2 ints), B = 16 values (4 ints) per lane
// ---------------------------------------------------------------------------

__global__ void smfma_f16(int iter, float *dummy)
{
    vec16<float> result = {0};
    vec4<_Float16> a;
    vec8<_Float16> b;
    for (int k = 0; k < 4; ++k) a[k] = threadIdx.x;
    for (int k = 0; k < 8; ++k) b[k] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_smfmac_f32_32x32x16_f16(
            a, b, result, SPARSE_INDEX, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void smfma_bf16(int iter, float *dummy)
{
    vec16<float> result = {0};
    vec4<short> a;
    vec8<short> b;
    for (int k = 0; k < 4; ++k) a[k] = threadIdx.x;
    for (int k = 0; k < 8; ++k) b[k] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(
            a, b, result, SPARSE_INDEX, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void smfma_i8(int iter, float *dummy)
{
    vec16<int> result = {0};
    vec2<int> a;
    vec4<int> b;
    for (int k = 0; k < 2; ++k) a[k] = threadIdx.x;
    for (int k = 0; k < 4; ++k) b[k] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_smfmac_i32_32x32x32_i8(
            a, b, result, SPARSE_INDEX, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void smfma_f8(int iter, float *dummy)
{
    // gfx942 uses the fnuz fp8 format (as in mfma)
    vec16<float> result = {0};
    vec2<int> a;
    vec4<int> b;
    for (int k = 0; k < 2; ++k) a[k] = threadIdx.x;
    for (int k = 0; k < 4; ++k) b[k] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(
            a, b, result, SPARSE_INDEX, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

// ---------------------------------------------------------------------------
// Precision -> kernel, instruction, operations per instruction
// Dense-equivalent count as in AMD's sparse peak figures: 2 * M * N * K, with K
// the K of the instruction name (the full K, of which A stores only half).
// ---------------------------------------------------------------------------

typedef void (*smfma_kernel_t)(int, float*);

struct SmfmaType {
    const char* precision;
    smfma_kernel_t kernel;
    const char* instruction;
    long ops_per_instruction;
};

static const SmfmaType SMFMA_TYPES[] = {
    {"fp16", smfma_f16,  "v_smfmac_f32_32x32x16_f16",      32768},   // 2*32*32*16
    {"bf16", smfma_bf16, "v_smfmac_f32_32x32x16_bf16",     32768},   // 2*32*32*16
    {"int8", smfma_i8,   "v_smfmac_i32_32x32x32_i8",       65536},   // 2*32*32*32
    {"fp8",  smfma_f8,   "v_smfmac_f32_32x32x32_fp8_fp8",  65536},   // 2*32*32*32
};

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
    const char* precision = "fp64";
    long iters = 2000;
    bool check = false, have_duration = false, have_prec = false, have_iters = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "smfma: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "smfma: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "smfma: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--precision")) {
            if (have_prec) { fprintf(stderr, "smfma: --precision given twice\n"); return 1; }
            have_prec = true;
            precision = val;
        } else if (!strcmp(key, "--iters")) {
            if (have_iters) { fprintf(stderr, "smfma: --iters given twice\n"); return 1; }
            have_iters = true;
            if (!parse_long(val, &iters) || iters < 1 || iters > 1000000000L) {
                fprintf(stderr, "smfma: --iters must be an integer between 1 and 1000000000 (got '%s')\n", val);
                return 1;
            }
        } else {
            fprintf(stderr, "smfma: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "smfma: --duration is required\n"); return 1; }

    const SmfmaType* type = nullptr;
    for (const SmfmaType& t : SMFMA_TYPES)
        if (!strcmp(precision, t.precision)) type = &t;
    if (!type) {
        fprintf(stderr, "smfma: --precision '%s' is not valid (fp16, bf16, int8, fp8)\n", precision);
        return 1;
    }
    if (check) return 0;                       // arguments valid, no GPU access

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    const unsigned cus = (unsigned)prop.multiProcessorCount;
    const unsigned blocks = BLOCKS_PER_CU * cus;
    fprintf(stderr, "smfma: device %s (%s), precision=%s (%s), blocks=%u threads=%d iters=%ld\n",
            prop.name, pci, precision, type->instruction, blocks, WORKGROUP_SIZE, iters);

    float* d_dummy;
    HIP_CHECK(hipMalloc(&d_dummy, 64 * sizeof(float)));   // as in rocprof-compute

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(type->kernel, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, (int)iters, d_dummy);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(type->kernel, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, (int)iters, d_dummy);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- results ----
    // Dense-equivalent FLOP (for int8: integer operations, op_type "iop"), as in the
    // sparse peak figures of AMD (2 x the dense rate).
    // Memory traffic: at most one float store per thread and launch (upper
    // bound; in fact all threads write dummy[0]). Counted so that a_gbyte > 0
    // and a_intensity is defined (contract); the intensity is not meaningful.
    const double waves_per_block = (double)WORKGROUP_SIZE / WAVEFRONT_SIZE;
    const double a_gflop = (double)launches * blocks * waves_per_block * iters
                           * type->ops_per_instruction / 1e9;
    const double a_gbyte = (double)launches * blocks * WORKGROUP_SIZE * sizeof(float) / 1e9;
    const double a_intensity = a_gflop / a_gbyte;
    const char* op_type = strcmp(precision, "int8") ? "flop" : "iop";

    fprintf(stderr, "smfma: precision=%s  %ld launches  %.3f s  %.1f %s\n",
            precision, launches, elapsed, a_gflop / elapsed,
            strcmp(precision, "int8") ? "GFLOP/s" : "GOP/s");
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": null, "
           "\"precision\": \"%s\", \"op_type\": \"%s\", "
           "\"e_instruction\": \"%s\", \"e_ops_per_instruction\": %ld, "
           "\"e_iters\": %ld, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           precision, op_type, type->instruction, type->ops_per_instruction,
           iters, blocks, WORKGROUP_SIZE);
    fflush(stdout);

    HIP_CHECK(hipFree(d_dummy));
    return 0;
}
