# Benchmarks

Background, history and implementation notes of the benchmarks in `src/`. Each source file
only carries its metadata (contract Section 3) and a short description; the interface of all
benchmarks is described in `docs/benchmark_contract.md`.

| Benchmark | Level | Component | Measures |
|---|---|---|---|
| [fma](#fma) | 1 | VALU | peak vector throughput per precision |
| [mfma](#mfma) | 1 | Matrix Cores | peak matrix throughput per precision |
| [copy](#copy) | 1 | HBM | bandwidth (read + write) |
| [reuse](#reuse) | 1 | L1, L2, LLC | read bandwidth per cache level |
| [chase](#chase) | 1 | L1, L2, LLC, HBM | load latency per memory level |
| [gemm](#gemm) | 2 | VALU, LDS, memory | load shifted between memory and compute via K |

## fma

**Kernel.** Based on Guerreiro et al. (HPCA 2018, `computeKernel.cu`), ported to HIP. Each
thread loads one value, runs four independent chains of `ops_per_element` multiply-adds and
stores the sum. Accounting: 8 operations per iteration and thread.

**Floating point (since 0.5.0).** The original chain `r = r*r + r` overflows to inf after 7
(fp16), 10 (fp32) or 13 (fp64) iterations; afterwards the same value repeats, hardly any bits
toggle and the power draw is far too low although the VALU is fully busy (validation findings 6
and 12). Now each chain alternates between two FMAs that undo each other:

    r = r * a  + b        (a = 1.3, b = 0.7)
    r = r * ai + bi       (ai = 1/a, bi = -b/a)

The values stay bounded and change in every step (they jump between two values per chain).
`a, b, ai, bi` are kernel arguments, so the compiler cannot fold the two FMAs at compile time.

**Integer types.** Unchanged, `r = r*r + r`. They are unsigned, so overflow is defined
(wrap-around) and the values keep changing. 16-bit integers are computed explicitly in unsigned,
because C++ promotes them to signed `int` (65535 × 65535 would overflow → undefined behaviour).

**Not included.** bf16 (no native VALU arithmetic on gfx942).

**Known behaviour.** fp32 and fp16 use packed instructions (two values per lane). int64 has no
native multiply-add; the compiler emits three instructions, and the VALU utilization counter
exceeds 100 %.

## mfma

**Kernel.** The `mfma_*` kernels are taken unchanged from the roofline benchmark of
rocprofiler-compute (ROCm Compute Profiler), github.com/ROCm/rocm-systems, branch develop,
commit 06cf5fb, `projects/rocprofiler-compute/src/roofline/benchmark/gfx9/benchmark_gfx9_base.py`
(fp32, fp64) and `gfx9/benchmark_gfx942.py` (fp16, bf16, int8, fp8). Launch configuration and
FLOP count from `benchmark_base.py` (`matrix_bench`): 256 threads per block, 128 blocks per CU,
2000 iterations,

    FLOP = blocks * (256 / 64) * iters * ops_per_instruction
    ops_per_instruction = 2 * M * N * K of the instruction

**Data.** Each wavefront runs one dependent chain `result = A*B + result`. The operands are
constant (`threadIdx.x`) for the whole run; only the accumulator changes. Few bits toggle in the
inputs, so the power draw is probably lower than with changing data.
Kept as in rocprof-compute for now, to compare with its roofline peak.

**Changes for this suite.** Compiled ahead of time instead of hiprtc; kernel names without
`extern "C"`; time loop over `--duration` with ROI time stamps and JSON output; argument check;
no verification (the store only prevents dead-code elimination; `verify_ok = null`). fp8 on
gfx942 is the "fnuz" variant (AMD comment in `benchmark_gfx942.py`). License: MIT (notice in
the source file).

## copy

**Kernel.** Copy kernel of STREAM (McCalpin) / BabelStream: one thread per element copies A to
B (`B[i] = A[i]`, double), no computation. 100 launches per synchronization, so that the launch
overhead does not dominate for small arrays.

**History.** Named `stream` until 2026-11-10; renamed because it is only the Copy kernel of
STREAM, not the full benchmark with Copy, Scale, Add and Triad. Behaviour unchanged.

## reuse

**Kernel.** `Cache_bw`, taken unchanged from the roofline benchmark of rocprofiler-compute,
github.com/ROCm/rocm-systems, branch develop, commit 06cf5fb,
`projects/rocprofiler-compute/src/roofline/benchmark/benchmark_base.py` (kernel source there as a
string, compiled with hiprtc). Launch configuration as there: 256 threads per block, 128 blocks
per CU, float loads. Every block reads the whole buffer `passes` times, so the buffer (one cache
size) is shared by all blocks of a CU/XCD.

**Buffer sizes (gfx942).** L1 32 KiB (100 passes), L2 4 MiB (10 passes), LLC 64 MiB (1 pass).
LLC = Infinity Cache; AMD calls it MALL (Memory Attached Last Level cache). 64 MiB is larger than
all L2 caches together (6 × 4 MiB) and smaller than the Infinity Cache (256 MB).

**Changes for this suite.** Compiled ahead of time instead of hiprtc; time loop over
`--duration` with ROI time stamps and JSON output; argument check; the buffer is initialized
(in rocprof-compute it is summed uninitialized); no verification (all blocks write the same
result array, `verify_ok = null`). License: MIT (notice in the source file).

**Known behaviour.** For L1 the index calculation limits the throughput (VALU ≈ 97 %), not the
cache.

**History.** Named `cache` until 2026-11-10 (renamed: too similar to `chase`, and the only
benchmark named after its component; `reuse` names the access pattern). It replaces cache 0.1.0
(own kernel: every block read its own chunk, 4 blocks per CU, double), which reached only
~14 TB/s for both L1 and L2, probably because of too few blocks per CU (validation finding 14).

## chase

**Kernel.** Pointer chasing as in Mei & Chu (2017): one thread follows a random chain,
`j = next[j]`. Every access depends on the previous one, so nothing can run in parallel or be
prefetched. The chain is a random permutation with exactly one cycle through all nodes
(Sattolo's algorithm, fixed seed). The buffer size selects the memory level. Loads go through
the vector L1 (not the scalar cache).

## gemm

**Kernel.** Matrix multiplication `C = A * B` on the VALU (no Matrix Cores), blocked or naive.
Level 2: K shifts the load between memory (small K) and compute (large K).

| `--precision` | `--accumulate` | input | accumulation | output |
|---|---|---|---|---|
| fp64 | native | fp64 | fp64 | fp64 |
| fp32 | native | fp32 | fp32 | fp32 |
| fp16 | native | fp16 | fp16 | fp16 |
| fp16 | fp32 | fp16 | fp32 | fp16 |
| int8 | native | int8 | int32 | int32 |
| int16 | native | int16 | int32 | int32 |
| int32 | native | int32 | int32 | int32 |
| int64 | native | int64 | int64 | int64 |

**`--kernel blocked` (default).** Each workgroup (256 threads) computes one TILE × TILE tile of
C. It loads its row strip of A (TILE × K) and column strip of B (K × TILE) into the LDS in chunks
of one cache line depth; each thread accumulates RT × RT elements in registers. Every loaded
element is reused TILE times. M and N must be multiples of 128; K is arbitrary.

Traffic model `kernel_level` (exactly what the kernel loads from and writes to global memory):

    Bytes = s_in * (M*K * N/TILE + K*N * M/TILE) + s_out * M*N
    FLOP  = 2 * M*N*K
    AI    = 2K / (s_in * 2K/TILE + s_out)          (independent of M and N)
    fp64: AI = 16 K / (K + 64)   -> K sets the AI from ~0.25 (K = 1) up to almost 16

Every loaded cache line is fully used as long as K ≤ 128 / s_in or K is a multiple of it
(otherwise the rows of A do not start on line boundaries). The model therefore describes the
traffic between kernel and L1/L2. HBM traffic can be lower when workgroups share strips of A/B
via L2 or Infinity Cache.

**`--kernel naive`.** One thread per element of C, without tiles and LDS. Each thread reads its
whole row of A and column of B; reuse depends only on the caches. Traffic model `minimal`: A, B
and C exactly once. This is a lower bound for the real traffic (upper bound for the AI); the
actual traffic is orders of magnitude higher.
