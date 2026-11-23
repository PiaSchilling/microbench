// ---- benchmark metadata ----
// benchmark=shared_read_rocprof
// level=1
// component=GPU/Memory/LDS
// description=REFERENCE ONLY: LDS kernel of the rocprof-compute roofline (LDS_bw), dependent 1-byte reads (pointer chase in 64 B of LDS); bandwidth as counted by rocprof-compute (4 B per read); comparison for shared_read, not a coverage point
// tags=lds;reference
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.iterations=2000|>= 1|Chase steps per thread and launch (2000 as in rocprof-compute)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=ds_read_u8
// expect_counters=
// ---- end metadata ----
//
// shared_read_rocprof: REFERENCE KERNEL, used only to compare the own shared_read benchmark with
// the LDS peak that rocprof-compute reports. Not part of the coverage analysis.
// Every thread follows a chain index = shmem[index] through 64 bytes of LDS; each
// read depends on the previous one (latency x parallelism rather than bandwidth),
// reads 1 byte, and is counted as 4 bytes as in rocprof-compute. Threads with
// tid >= 64 start outside the 64-byte array, as in the original.
// Source: LDS_bw kernel of the rocprof-compute roofline (MIT, see below).
// Details: docs/benchmarks.md#shared_read_rocprof
// Interface: docs/benchmark_contract.md
//
// ---------------------------------------------------------------------------
// License of the kernel LDS_bw (rocprofiler-compute):
//
// MIT License
//
// Copyright (c) Advanced Micro Devices, Inc.
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

static const int WORKGROUP_SIZE = 256;   // as in rocprof-compute (DEFAULT_WORKGROUP_SIZE)
static const int BLOCKS_PER_CU = 128;    // as in rocprof-compute
static const int COUNTED_BYTES = 4;      // rocprof-compute counts sizeof(float) per read

// ---------------------------------------------------------------------------
// Kernel (rocprofiler-compute, MIT license, see header) -- unchanged except
// for the removed extern "C"
// ---------------------------------------------------------------------------

__global__ void LDS_bw(int numIter, float *dummy)
{
    const int tid = threadIdx.x;
    __shared__ unsigned char shmem[64];


    if (tid == 0)
    {
        #pragma unroll
        for (int i=0;i<63;i++)
            shmem[i] = i+1;

        shmem[63] = 0;
    }

    __syncthreads();

    int index = tid;
    #pragma unroll 64
    for(int iter = 0; iter < numIter; iter++)
        index = shmem[index];

    dummy[tid] = (float )index;
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
    double duration = 0;
    unsigned long long iters_arg = 2000;
    bool check = false, have_duration = false, have_iters = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "shared_read_rocprof: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        bool* seen;
        bool valid;
        if (!strcmp(key, "--duration")) {
            seen = &have_duration;
            valid = parse_double(val, &duration) && duration > 0;
        } else if (!strcmp(key, "--iterations")) {
            seen = &have_iters;
            valid = parse_ull(val, &iters_arg) && iters_arg >= 1 && iters_arg <= 1000000000ULL;
        } else {
            fprintf(stderr, "shared_read_rocprof: unknown option %s\n", key);
            return 1;
        }
        if (*seen) { fprintf(stderr, "shared_read_rocprof: %s given twice\n", key); return 1; }
        *seen = true;
        if (!valid) {
            fprintf(stderr, "shared_read_rocprof: invalid value for %s: '%s' (see: run_suite.sh --help shared_read_rocprof)\n", key, val);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "shared_read_rocprof: --duration is required\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access
    const int iters = (int)iters_arg;

    // ---- setup (sizes as in rocprof-compute lds_bw_benchmark) ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);
    const unsigned blocks = BLOCKS_PER_CU * (unsigned)prop.multiProcessorCount;
    const double reads_per_launch = (double)blocks * WORKGROUP_SIZE * iters;
    fprintf(stderr, "shared_read_rocprof: device %s (%s), blocks=%u threads=%d iterations=%d\n",
            prop.name, pci, blocks, WORKGROUP_SIZE, iters);

    float* d_dummy;
    HIP_CHECK(hipMalloc(&d_dummy, WORKGROUP_SIZE * sizeof(float)));

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(LDS_bw, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, iters, d_dummy);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(LDS_bw, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, iters, d_dummy);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- results; GB = 10^9 byte ----
    // a_gbyte as counted by rocprof-compute (4 B per read); the kernel actually
    // reads 1 B per read, see d_gbyte_per_s_actual.
    const double reads = (double)launches * reads_per_launch;
    const double a_gbyte = reads * COUNTED_BYTES / 1e9;
    const double a_gflop = 0.0;
    const double a_intensity = a_gflop / a_gbyte;

    fprintf(stderr, "shared_read_rocprof: %ld launches  %.3f s  %.1f GB/s as counted by rocprof-compute  (%.1f GB/s actually read, %.1f G reads/s)\n",
            launches, elapsed, a_gbyte / elapsed, reads / elapsed / 1e9, reads / elapsed / 1e9);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": null, "
           "\"d_gbyte_per_s_actual\": %.9g, \"d_gread_per_s\": %.9g, "
           "\"e_loop_iterations\": %d, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           reads / elapsed / 1e9, reads / elapsed / 1e9,
           iters, blocks, WORKGROUP_SIZE);
    fflush(stdout);

    HIP_CHECK(hipFree(d_dummy));
    return 0;
}
