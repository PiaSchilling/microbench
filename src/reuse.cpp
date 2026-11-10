// ---- benchmark metadata ----
// benchmark=reuse
// level=1
// component=GPU/Memory/Cache
// description=Reuse within a kernel: all blocks re-read one buffer of cache size (cache bandwidth kernel of the rocprof-compute roofline); L1, L2 or LLC (Infinity Cache)
// tags=memory-bound;bandwidth;cache
// version=0.2.0
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.level=l1|l1, l2, llc|Cache level: buffer of 32 KiB (L1), 4 MiB (L2) or 64 MiB (LLC = Infinity Cache), sizes for gfx942
// param.passes=auto|auto or >= 1|How often every block reads the whole buffer per launch; auto = 100 (l1), 10 (l2), 1 (llc) as in rocprof-compute
//
// # Expectations (validation mode). To be confirmed with a disassembly (--save-temps).
// # expect_isa=global_load_dword
// expect_counters=
// ---- end metadata ----
//
// Kernel Cache_bw: taken unchanged from the roofline benchmark of
// rocprofiler-compute (ROCm Compute Profiler), github.com/ROCm/rocm-systems,
// branch develop, commit 06cf5fb, projects/rocprofiler-compute/src/roofline/
// benchmark/benchmark_base.py (kernel source there as a string, compiled with
// hiprtc). Launch configuration as there: 256 threads per block, 128 blocks
// per CU, float loads. Every block reads the WHOLE buffer `passes` times, so
// the buffer (one cache size) is shared by all blocks of a CU/XCD.
//
// Changes for this suite (contract docs/benchmark_contract.md): compiled
// ahead of time instead of hiprtc; time loop over --duration with ROI time
// stamps and JSON output; argument check; the buffer is initialized (in
// rocprof-compute it is summed uninitialized); no verification (all blocks
// write the same result array, verify_ok = null). The last level cache is
// called llc here, as in the rest of the suite (component config llc_bytes);
// AMD calls it MALL (Memory Attached Last Level cache).
//
// Named "cache" until 2026-11-10 (renamed: too similar to "chase", and the
// only benchmark named after its component; "reuse" names the access pattern,
// the difference to copy, formerly stream). Kernel and behaviour unchanged.
//
// Replaces cache 0.1.0 (own kernel: every block read its own chunk, 4 blocks
// per CU, double): it reached only ~14 TB/s for both L1 and L2, probably too
// few blocks per CU (validation finding 14).
//
// ---------------------------------------------------------------------------
// License of the kernel Cache_bw (rocprofiler-compute):
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
#include <vector>

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

// Buffer sizes for gfx942 (rocprof-compute: L1 32 KiB, L2 4 MiB, MALL = LLC 64 MiB)
static const int L1_SIZE   = 32 * 1024;
static const int L2_SIZE   = 4 * 1024 * 1024;
static const int LLC_SIZE  = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Kernel (rocprofiler-compute, MIT license, see header) -- unchanged
// ---------------------------------------------------------------------------

template <typename T, int cacheSize, int workgroup_size>
__global__ void Cache_bw(const T *memBlock, T *dummy, int numIter)
{
const int thread_id = threadIdx.x;
constexpr int cache_count = cacheSize / sizeof(T);

T sink;

sink = 0;
for (int iter = 0; iter < numIter; ++iter)
{
#pragma unroll 32
    for (int i = 0; i < cache_count; i += workgroup_size)
    {
    // if the size of the memory block is small (e.g., the size
    // of L1), then we need a slightly more complicated index
    // calculation. Otherwise, the compiler holds all the loads
    // in the inner loop in registers upon the first pass of the
    // outer loop, and it doesn't do the loads upon subsequent
    // passes of the outer loop.
    // OTOH, if the size of the memory block is larger (such as L2
    // size), experimentation showed that the overhead of the more
    // complicated index calculation has a noticeable effect on BW,
    // so we use a simpler index expression instead. This works since
    // for larger memory blocks, the compiler cannot hold the loads
    // of the inner loop in registers anymore, as it can with L1-sized
    // buffers.
    if constexpr (cache_count / workgroup_size <= 32)
    {
        sink += memBlock[(thread_id + i + iter) % cache_count];
    }
    else
    {
        sink += memBlock[thread_id + i];
    }
    }
}

dummy[thread_id] = sink;
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

static bool parse_long(const char* s, long* out) {
    char* end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0') return false;
    *out = v;
    return true;
}

typedef void (*reuse_kernel_t)(const float*, float*, int);

int main(int argc, char** argv) {
    double duration = 0;
    const char* level = "l1";
    long passes = 0;                         // 0 = auto
    bool check = false, have_duration = false, have_level = false, have_passes = false;

    // ---- arguments (contract section 4) ----
    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        if (!strcmp(key, "--check")) { check = true; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "reuse: option %s requires a value\n", key); return 1; }
        const char* val = argv[++i];
        if (!strcmp(key, "--duration")) {
            if (have_duration) { fprintf(stderr, "reuse: --duration given twice\n"); return 1; }
            have_duration = true;
            if (!parse_double(val, &duration) || duration <= 0) {
                fprintf(stderr, "reuse: --duration must be a number > 0 (got '%s')\n", val);
                return 1;
            }
        } else if (!strcmp(key, "--level")) {
            if (have_level) { fprintf(stderr, "reuse: --level given twice\n"); return 1; }
            have_level = true;
            if (strcmp(val, "l1") && strcmp(val, "l2") && strcmp(val, "llc")) {
                fprintf(stderr, "reuse: --level must be l1, l2 or llc (got '%s')\n", val);
                return 1;
            }
            level = val;
        } else if (!strcmp(key, "--passes")) {
            if (have_passes) { fprintf(stderr, "reuse: --passes given twice\n"); return 1; }
            have_passes = true;
            if (!strcmp(val, "auto")) {
                passes = 0;
            } else if (!parse_long(val, &passes) || passes < 1 || passes > 1000000) {
                fprintf(stderr, "reuse: --passes must be auto or an integer between 1 and 1000000 (got '%s')\n", val);
                return 1;
            }
        } else {
            fprintf(stderr, "reuse: unknown option %s\n", key);
            return 1;
        }
    }
    if (!have_duration) { fprintf(stderr, "reuse: --duration is required\n"); return 1; }
    if (check) return 0;                       // arguments valid, no GPU access

    // ---- level -> kernel, buffer size, default passes (as in rocprof-compute) ----
    reuse_kernel_t kernel;
    size_t cache_size;
    long auto_passes;
    if (!strcmp(level, "l1")) {
        kernel = Cache_bw<float, L1_SIZE, WORKGROUP_SIZE>;   cache_size = L1_SIZE;   auto_passes = 100;
    } else if (!strcmp(level, "l2")) {
        kernel = Cache_bw<float, L2_SIZE, WORKGROUP_SIZE>;   cache_size = L2_SIZE;   auto_passes = 10;
    } else {
        kernel = Cache_bw<float, LLC_SIZE, WORKGROUP_SIZE>;  cache_size = LLC_SIZE;  auto_passes = 1;
    }
    if (passes == 0) passes = auto_passes;

    // ---- setup ----
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char pci[32];
    snprintf(pci, sizeof pci, "%04x:%02x:%02x.0",
             (unsigned)prop.pciDomainID, (unsigned)prop.pciBusID, (unsigned)prop.pciDeviceID);

    const unsigned cus = (unsigned)prop.multiProcessorCount;
    const unsigned blocks = BLOCKS_PER_CU * cus;
    fprintf(stderr, "reuse: device %s (%s), level=%s buffer=%zu B, blocks=%u threads=%d passes=%ld\n",
            prop.name, pci, level, cache_size, blocks, WORKGROUP_SIZE, passes);

    // Buffer initialized with small values (rocprof-compute sums uninitialized memory)
    const size_t n = cache_size / sizeof(float);
    std::vector<float> h_A(n);
    for (size_t i = 0; i < n; ++i) h_A[i] = (float)(i % 16);
    float *d_A, *d_dummy;
    HIP_CHECK(hipMalloc(&d_A, cache_size));
    HIP_CHECK(hipMalloc(&d_dummy, WORKGROUP_SIZE * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), cache_size, hipMemcpyHostToDevice));

    // Warmup (not part of the ROI)
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0,
                       (const float*)d_A, d_dummy, (int)passes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // ---- timed loop = ROI: launch until duration is reached, at least once ----
    long launches = 0;
    const unix_clock::time_point u0 = unix_clock::now();
    const mono_clock::time_point t0 = mono_clock::now();
    mono_clock::time_point t1;
    double elapsed;
    do {
        hipLaunchKernelGGL(kernel, dim3(blocks), dim3(WORKGROUP_SIZE), 0, 0,
                           (const float*)d_A, d_dummy, (int)passes);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        t1 = mono_clock::now();
        elapsed = std::chrono::duration<double>(t1 - t0).count();
    } while (elapsed < duration);
    const unix_clock::time_point u1 = unix_clock::now();

    // ---- results; GB = 10^9 byte ----
    // As in rocprof-compute: every block reads the whole buffer `passes` times
    // (the final store of 256 floats per block is negligible and not counted).
    const double a_gbyte = (double)launches * blocks * passes * cache_size / 1e9;
    const double a_gflop = (double)launches * blocks * passes * n / 1e9;    // one add per float read
    const double a_intensity = a_gflop / a_gbyte;

    fprintf(stderr, "reuse: level=%s  %ld launches  %.3f s  %.1f GB/s\n",
            level, launches, elapsed, a_gbyte / elapsed);
    printf("{\"runtime_s\": %.9f, \"iterations\": %ld, "
           "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, "
           "\"roi_start_mono_ns\": %lld, \"roi_end_mono_ns\": %lld, "
           "\"e_pci_bus_id\": \"%s\", "
           "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, "
           "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, "
           "\"verify_ok\": null, "
           "\"cache_level\": \"%s\", \"e_buffer_bytes\": %zu, \"e_passes\": %ld, "
           "\"e_blocks\": %u, \"e_threads\": %d}\n",
           elapsed, launches, unix_s(u0), unix_s(u1), mono_ns(t0), mono_ns(t1), pci,
           a_gflop, a_gbyte, a_intensity, a_gflop / elapsed, a_gbyte / elapsed,
           level, cache_size, passes, blocks, WORKGROUP_SIZE);
    fflush(stdout);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_dummy));
    return 0;
}
