// ---- benchmark metadata ----
// benchmark=read
// level=1
// component=GPU/Memory/Bandwidth
// description=Read-only streaming kernel of the rocprof-compute roofline (HBM_bw): every thread reads its own 16-byte elements once per launch, non-temporal loads; HBM read bandwidth
// tags=memory-bound;bandwidth;read-only
// version=0.1.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.bytes=4294967296|>= 1048576|Data read per launch; rounded up to whole steps (blocks x 256 threads x 16 loads x 16 B); must be much larger than the Infinity Cache (256 MB)
//
// # Expectations (profile mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dwordx4
// expect_counters=
// ---- end metadata ----
//
// read: every thread reads its own 16-byte elements in a grid-stride loop, each
// byte once per launch, no reuse and no writes. Measures HBM read bandwidth.
// Source: HBM_bw kernel of the rocprof-compute roofline (MIT, see below).
// Details: docs/benchmarks.md#read
// Interface: docs/benchmark_contract.md
//
// ---------------------------------------------------------------------------
// License of the kernel HBM_bw (rocprofiler-compute):
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

static const int WORKGROUP_SIZE = 256;   // as in rocprof-compute
static const int BLOCKS_PER_CU = 128;    // as in rocprof-compute
static const int UNROLL = 16;            // loads per thread and step (kernel below)
static const int ELEM_SIZE = 16;         // sizeof(__uint128_t)

// ---------------------------------------------------------------------------
// Kernel (rocprofiler-compute, MIT license, see header) -- unchanged except
// for the removed extern "C"
// ---------------------------------------------------------------------------

__global__ void HBM_bw(__uint128_t *src, long numSteps)
{
    unsigned long offset = (unsigned long)blockIdx.x * blockDim.x
                           + threadIdx.x;
    const unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
    __uint128_t v = 0;

    #pragma unroll 1
    for (long step = 0; step < numSteps; step++)
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            v |= __builtin_nontemporal_load(&src[offset]);
            offset += stride;
        }
    }
    if (v == 0) src[0] = v;
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
    unsigned long long bytes = 4ULL * 1024 * 1024 * 1024;   // 4 GiB as in rocprof-compute
    bool check = false, have_duration = false, have_bytes = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "read: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "read: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "read: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--bytes")) {
            if (have_bytes) { fprintf(stderr, "read: --bytes given twice\n"); return 1; }
            have_bytes = true;
            if (!parse_ull(val, &bytes) || bytes < (1ULL << 20) || bytes > (1ULL << 40)) {
                fprintf(stderr, "read: --bytes must be an integer between 1048576 and 2^40 (got '%s')\n", val);
                return 1;
            }
        } else {
            fprintf(stderr, "read: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "read: --duration is required\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    // Sizes as in rocprof-compute (device_memory_bw_benchmark): whole steps of
    // blocks x threads x UNROLL elements of 16 B; the data set is rounded up.
    const unsigned cus = (unsigned)prop.multiProcessorCount;
    const unsigned blocks = BLOCKS_PER_CU * cus;
    const unsigned long long elems_per_step = (unsigned long long)blocks * WORKGROUP_SIZE * UNROLL;
    const unsigned long long want_elems = bytes / ELEM_SIZE;
    const long num_steps = (long)((want_elems + elems_per_step - 1) / elems_per_step);
    const unsigned long long total_elems = (unsigned long long)num_steps * elems_per_step;
    const size_t total_bytes = (size_t)(total_elems * ELEM_SIZE);
    fprintf(stderr, "read: device %s (%s), bytes=%zu (requested %llu), blocks=%u threads=%d steps=%ld\n",
            prop.name, pci, total_bytes, bytes, blocks, WORKGROUP_SIZE, num_steps);

    __uint128_t* d_src;
    HIP_CHECK(hipMalloc(&d_src, total_bytes));
    // Non-zero fill: the kernel only stores (to src[0]) if all loaded values are
    // zero; rocprof-compute reads uninitialized memory instead.
    HIP_CHECK(hipMemset(d_src, 0xA5, total_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(HBM_bw, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, d_src, num_steps);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(HBM_bw, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0, d_src, num_steps);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- results; GB = 10^9 byte ----
    const double a_gbyte = (double)launches * total_bytes / 1e9;
    const double a_gflop = 0.0;                     // no floating-point arithmetic (bitwise OR only)
    const double a_intensity = a_gflop / a_gbyte;   // 0

    fprintf(stderr, "read: %ld launches  %.3f s  %.1f GB/s\n", launches, elapsed, a_gbyte / elapsed);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": null, "
           "\"e_buffer_bytes\": %zu, \"e_steps\": %ld, \"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           total_bytes, num_steps, blocks, WORKGROUP_SIZE);
    fflush(stdout);

    HIP_CHECK(hipFree(d_src));
    return 0;
}
