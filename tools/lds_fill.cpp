// lds_fill.cpp - fill the LDS of every CU with zeros or pseudo-random words, then exit.
//
// Helper for the LDS_bw finding (docs/benchmarks.md, shared_read_rocprof_prefill):
// run it directly before an external tool (e.g. rocprof-compute --roof-only) to
// control what the next process finds in the LDS.
//
// Build: hipcc --offload-arch=gfx942 -O3 -std=c++17 tools/lds_fill.cpp -o build/lds_fill
// Usage: build/lds_fill zero|random [device]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "lds_fill: %s (%s:%d)\n", hipGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static const int LDS_BYTES = 65536;          // LDS per CU on gfx942

// Same kernel as lds_prefill in src/shared_read_rocprof_prefill.cpp.
__global__ void lds_fill(int mode, unsigned *sink)
{
    extern __shared__ unsigned lds_words[];
    const unsigned tid = threadIdx.x;
    for (unsigned w = tid; w < LDS_BYTES / 4; w += blockDim.x) {
        unsigned v = 0;
        if (mode == 1) {
            unsigned h = (w + 1) * 2654435761u ^ (blockIdx.x * 40503u);
            h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
            v = h;
        }
        lds_words[w] = v;
    }
    __syncthreads();
    if (lds_words[tid] == 0x12345678u) sink[0] = tid;   // keeps the stores alive
}

int main(int argc, char **argv)
{
    if (argc < 2 || (std::strcmp(argv[1], "zero") != 0 && std::strcmp(argv[1], "random") != 0)) {
        std::fprintf(stderr, "usage: %s zero|random [device]\n", argv[0]);
        return 2;
    }
    const int mode = std::strcmp(argv[1], "random") == 0 ? 1 : 0;
    const int dev = argc > 2 ? std::atoi(argv[2]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    unsigned *d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, sizeof(unsigned)));
    // 64 KiB per block -> one block per CU at a time; 16 blocks per CU cover every CU.
    hipLaunchKernelGGL(lds_fill, dim3(16 * prop.multiProcessorCount), dim3(256), LDS_BYTES, 0, mode, d_sink);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_sink));
    std::printf("lds_fill: device %d, %d CUs, mode %s\n", dev, prop.multiProcessorCount, argv[1]);
    return 0;
}
