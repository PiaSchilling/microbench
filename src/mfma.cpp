// ---- benchmark metadata ----
// benchmark=mfma
// level=1
// component=GPU/Compute/MFMA
// description=Matrix-core (MFMA) kernel of the rocprof-compute roofline: one MFMA chain per wavefront with constant operands; compute pole of the matrix cores
// tags=compute-bound;mfma;matrix-core
// version=0.1.0
// precisions=fp64,fp32,fp16,bf16,int8,fp8
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.precision=fp64|see precisions|Input type of the MFMA instruction (fp16, bf16, fp8: accumulation in fp32; int8: in int32)
// param.iters=2000|>= 1|MFMA instructions per wavefront and launch (2000 as in rocprof-compute)
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa.fp64=v_mfma_f64_16x16x4_f64
// # expect_isa.fp32=v_mfma_f32_32x32x2_f32
// # expect_isa.fp16=v_mfma_f32_32x32x8_f16
// # expect_isa.bf16=v_mfma_f32_32x32x8_bf16
// # expect_isa.int8=v_mfma_i32_32x32x16_i8
// # expect_isa.fp8=v_mfma_f32_32x32x16_fp8_fp8
// expect_counters=
// ---- end metadata ----
//
// Kernels mfma_*: taken unchanged from the roofline benchmark of
// rocprofiler-compute (ROCm Compute Profiler), github.com/ROCm/rocm-systems,
// branch develop, commit 06cf5fb, projects/rocprofiler-compute/src/roofline/
// benchmark/gfx9/benchmark_gfx9_base.py (fp32, fp64) and
// gfx9/benchmark_gfx942.py (fp16, bf16, int8, fp8); launch configuration and
// FLOP count from benchmark_base.py (matrix_bench): 256 threads per block,
// 128 blocks per CU, 2000 iterations,
//   FLOP = blocks * (256 / 64) * iters * ops_per_instruction,
//   ops_per_instruction = 2 * M * N * K of the instruction (table MATRIX_OPS).
//
// Each wavefront runs ONE dependent chain result = A*B + result. The operands
// are constant (threadIdx.x) for the whole run; only the accumulator changes.
// So few bits toggle in the inputs, and the power draw is probably lower than
// with changing data (validation finding 12). Kept as in rocprof-compute for
// now, to compare with its roofline peak.
//
// Changes for this suite (contract docs/benchmark_contract.md): compiled
// ahead of time instead of hiprtc; kernel names without extern "C"; time loop
// over --duration with ROI time stamps and JSON output; argument check; no
// verification (the store happens only if result[0] != 2*result[0], i.e. it
// only prevents dead-code elimination; verify_ok = null). fp8 on gfx942 is
// the "fnuz" variant of fp8 (AMD comment in benchmark_gfx942.py).
//
// ---------------------------------------------------------------------------
// License of the kernels mfma_* (rocprofiler-compute):
//
// MIT License
//
// Copyright (C) Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ---------------------------------------------------------------------------

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

static const int WORKGROUP_SIZE = 256;   // as in rocprof-compute
static const int BLOCKS_PER_CU = 128;    // as in rocprof-compute
static const int WAVEFRONT_SIZE = 64;

// ---------------------------------------------------------------------------
// Kernels (rocprofiler-compute, MIT license, see header) -- unchanged
// ---------------------------------------------------------------------------

template<typename T, int Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec2 = vecT<T, 2>;
template<typename T> using vec4 = vecT<T, 4>;
template<typename T> using vec8 = vecT<T, 8>;
template<typename T> using vec16 = vecT<T, 16>;

__global__ void mfma_f32(int iter, float *dummy)
{
    float a = threadIdx.x;
    vec16<float> result = {0};

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x2f32(
            a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void mfma_f64(int iter, float *dummy)
{
    double a =  threadIdx.x;

    vec4<double> result = {0};

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f64_16x16x4f64(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void mfma_f8(int iter, float *dummy)
{
    // MI300 series only - note gfx940/gfx941/gfx942 only uses fnuz f8
    long a =  threadIdx.x;

    vec16<float> result = {0};

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
            a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void mfma_f16(int iter, float *dummy)
{
    vec16<float> result = {0};
    vec4<__fp16> a;
    a[3] = a[2] = a[1] = a[0] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void mfma_bf16(int iter, float *dummy)
{
    vec16<float> result = {0};
    vec4<short> a;
    a[3] = a[2] = a[1] = a[0] = threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

__global__ void mfma_i8(int iter, float *dummy)
{
    vec16<int> result = {0};
    long a =  threadIdx.x;

    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}

// ---------------------------------------------------------------------------
// Precision -> kernel, instruction, operations per instruction
// (matrix_ops in benchmark_gfx942.py: 2 * M * N * K)
// ---------------------------------------------------------------------------

typedef void (*mfma_kernel_t)(int, float*);

struct MfmaType {
    const char* precision;
    mfma_kernel_t kernel;
    const char* instruction;
    long ops_per_instruction;
};

static const MfmaType MFMA_TYPES[] = {
    {"fp64", mfma_f64,  "v_mfma_f64_16x16x4_f64",      2048},   // 2*16*16*4
    {"fp32", mfma_f32,  "v_mfma_f32_32x32x2_f32",      4096},   // 2*32*32*2
    {"fp16", mfma_f16,  "v_mfma_f32_32x32x8_f16",     16384},   // 2*32*32*8
    {"bf16", mfma_bf16, "v_mfma_f32_32x32x8_bf16",    16384},   // 2*32*32*8
    {"int8", mfma_i8,   "v_mfma_i32_32x32x16_i8",     32768},   // 2*32*32*16
    {"fp8",  mfma_f8,   "v_mfma_f32_32x32x16_fp8_fp8", 32768},  // 2*32*32*16
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
        if (i + 1 >= argc) { fprintf(stderr, "mfma: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "mfma: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "mfma: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--precision")) {
            if (have_prec) { fprintf(stderr, "mfma: --precision given twice\n"); return 1; }
            have_prec = true;
            precision = val;
        } else if (!strcmp(key, "--iters")) {
            if (have_iters) { fprintf(stderr, "mfma: --iters given twice\n"); return 1; }
            have_iters = true;
            if (!parse_long(val, &iters) || iters < 1 || iters > 1000000000L) {
                fprintf(stderr, "mfma: --iters must be an integer between 1 and 1000000000 (got '%s')\n", val);
                return 1;
            }
        } else {
            fprintf(stderr, "mfma: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "mfma: --duration is required\n"); return 1; }

    const MfmaType* type = nullptr;
    for (const MfmaType& t : MFMA_TYPES)
        if (!strcmp(precision, t.precision)) type = &t;
    if (!type) {
        fprintf(stderr, "mfma: --precision '%s' is not valid (fp64, fp32, fp16, bf16, int8, fp8)\n", precision);
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
    fprintf(stderr, "mfma: device %s (%s), precision=%s (%s), blocks=%u threads=%d iters=%ld\n",
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
    // FLOP as in rocprof-compute (for int8: integer operations, op_type "iop").
    // Memory traffic: at most one float store per thread and launch (upper
    // bound; in fact all threads write dummy[0]). Counted so that a_gbyte > 0
    // and a_intensity is defined (contract); the intensity is not meaningful.
    const double waves_per_block = (double)WORKGROUP_SIZE / WAVEFRONT_SIZE;
    const double a_gflop = (double)launches * blocks * waves_per_block * iters
                           * type->ops_per_instruction / 1e9;
    const double a_gbyte = (double)launches * blocks * WORKGROUP_SIZE * sizeof(float) / 1e9;
    const double a_intensity = a_gflop / a_gbyte;
    const char* op_type = strcmp(precision, "int8") ? "flop" : "iop";

    fprintf(stderr, "mfma: precision=%s  %ld launches  %.3f s  %.1f %s\n",
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
