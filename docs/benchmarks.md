# Benchmarks

Background, history and implementation notes of the benchmarks in `src/`. Each source file
only carries its metadata (contract Section 3) and a short description; the interface of all
benchmarks is described in `docs/benchmark_contract.md`.

| Benchmark | Level | Component | Measures |
|---|---|---|---|
| [fma](#fma) | 1 | VALU | peak vector throughput per precision |
| [mfma](#mfma) | 1 | Matrix Cores | peak matrix throughput per precision |
| [smfma](#smfma) | 1 | Matrix Cores (sparse) | peak sparse matrix throughput per precision |
| [copy](#copy) | 1 | L2, LLC, HBM | bandwidth (read + write, same pattern) |
| [read](#read) | 1 | HBM | read bandwidth |
| [write](#write) | 1 | L2, LLC, HBM | write bandwidth |
| [cached_read](#cached_read) | 1 | L1, L2, LLC | read bandwidth per cache level (own kernel, 16-byte loads, no arithmetic) |
| [chase](#chase) | 1 | L1, L2, LLC, HBM | load latency per memory level |
| [gather](#gather) | 2 | L1, L2, LLC, HBM | bandwidth of irregular reads (random order) |
| [scatter](#scatter) | 2 | L2, LLC, HBM | bandwidth of irregular writes (random order) |
| [shared_read](#shared_read) | 1 | LDS | LDS read bandwidth |
| [shared_read_rocprof_prefill](#shared_read_rocprof_prefill) | test | LDS | test only: LDS_bw after filling the LDS with a pattern |
| [reuse](#reuse) | ref | L1, L2, LLC | reference only: rocprof-compute cache kernel, for comparison with cached_read |
| [shared_read_rocprof](#shared_read_rocprof) | ref | LDS | reference only: rocprof-compute LDS kernel, for comparison with shared_read |
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

**tf32 (own addition, version 0.2.0).** rocprof-compute has no tf32 kernel. `mfma_xf32` follows the
same pattern as `mfma_f32` with `__builtin_amdgcn_mfma_f32_32x32x4_xf32` (expected
`v_mfma_f32_32x32x4_xf32`): on gfx942 the tf32 format is called xf32 (fp32 inputs with reduced
internal precision, accumulation in fp32). A and B hold 2 floats per lane because K = 4;
ops_per_instruction = 2 × 32 × 32 × 4 = 8192. Expected peak: 4 × the fp32 matrix rate (AMD lists
490.3 TFLOP/s tf32 matrix for the MI300A at 2100 MHz). Note: xf32 exists on gfx940–942 only; later
architectures (gfx950) emulate tf32 in software.

## smfma

**Name.** AMD calls the instructions `v_smfmac` (sparse matrix fused multiply-accumulate); the
benchmark is called `smfma` for consistency with `mfma`, since it does the same `D = A*B + C`.

**Kernel.** Own implementation (rocprof-compute has no sparse kernel), same pattern and launch
configuration as `mfma`: every wavefront runs one dependent chain `result = A*B + result` with
constant operands, 256 threads per block, 128 blocks per CU, 2000 iterations. Instructions
(`__builtin_amdgcn_smfmac_*`): fp16 `v_smfmac_f32_32x32x16_f16`, bf16 `v_smfmac_f32_32x32x16_bf16`,
int8 `v_smfmac_i32_32x32x32_i8`, fp8 `v_smfmac_f32_32x32x32_fp8_fp8` (fnuz on gfx942).

**Sparsity.** A is 2:4 structured sparse along K: of every 4 values 2 are zero. A stores only the 2
non-zero values per group (half of K per lane), B all values; an index operand (2 bits per stored
value) says where the stored values belong. The hardware skips the zeros, so one instruction covers
twice the K of the dense instruction (32x32x16 f16 vs. 32x32x8). The index is the same constant for
all groups and lanes (`0x44444444`: positions 0 and 1); any valid pattern costs the same.

**FLOP.** Dense-equivalent, as in AMD's sparse peak figures: `2 * M * N * K` with the K of the
instruction name (fp16/bf16: 32768, int8/fp8: 65536 per instruction). Expected: about 2 × the dense
`mfma` rate of the same precision.

**Local ISA check (2026-11-24).** The four kernels, rewritten in plain C with the same builtins and
compiled with clang 18 for gfx942, emit the expected `v_smfmac_*` instructions in the loop (23 VGPRs,
no scratch, occupancy 8 waves per SIMD). To be confirmed with hipcc on the node.

## copy

**Kernel.** Copy kernel as in STREAM (McCalpin) / BabelStream: `B[e] = A[e]`, no computation.
Since 0.4.0 with 16-byte elements (`uint4`, expected `global_load_dwordx4` / `global_store_dwordx4`),
the widest access per lane as in all Level 1 memory benchmarks. Same access pattern for read and
write (sequential), so it is Level 1 (see the Level 1/2 boundary).

**Grid and repetitions.** As in `write`: fixed grid (128 blocks per CU × 256 threads), grid-stride
loop over the copies of one launch; copy `k` copies element `k mod n`, the position is advanced
incrementally. `--working_set_bytes` is the size of both arrays together (half each) and selects
the level: `copy_l2` 2 MiB, `copy_llc` 64 MiB, `copy_hbm` 4 GiB. Up to 64 MiB a launch passes over
the arrays several times (about 16 copies per thread), otherwise a launch of a few microseconds
would mostly measure the launch overhead; above, every element is copied once per launch, as in
STREAM. Within a pass neighbouring lanes copy neighbouring elements (coalesced), also after the
wrap-around. No L1 run: the L1 is assumed write-through (to be checked, measurement queue item 5).

**Difference to STREAM Copy.** Element type `uint4` instead of `double` (STREAM makes the type
configurable); the cache-level runs deliberately break the STREAM rule that the arrays must be
much larger than all caches.

**Address arithmetic (since 0.5.0).** The position is a 32-bit byte offset from the array base
pointers, so the compiler uses loads and stores with a scalar base and a 32-bit vector offset:
6 VALU instructions per copy (offset update, wrap-around, loop counter) instead of about 11 with
64-bit indices in 0.4.0 (local clang check), where `copy_l2` showed 21 % VALU although copy
computes nothing. Limit: at most 2 GiB per array (`--working_set_bytes` ≤ 4 GiB).

**Bytes.** 16 B read + 16 B written per copy (`a_gbyte`).

**Verification.** A is filled with `value(e)` by a setup kernel, B with zeros; after the timed loop
every element of B is compared with `value(e)`.

**History.** Named `stream` until 2026-11-10; renamed because it is only the Copy kernel of
STREAM, not the full benchmark with Copy, Scale, Add and Triad. 0.4.0 (2026-11-25): `uint4`
elements, fixed grid with grid-stride loop and repetitions per launch, `--working_set_bytes`
instead of `--elements`, runs for L2 and LLC, full verification. 0.5.0 (2026-11-25): 32-bit byte offsets instead of 64-bit indices. Up to 0.3.0: one `double` per
thread, 100 launches per synchronization, HBM only (copy_hbm 3.76 TB/s, 2 GiB).

## read

**Kernel.** `HBM_bw`, taken from the roofline benchmark of rocprofiler-compute,
github.com/ROCm/rocm-systems, branch develop, commit 787f6e7 (2026-11-17),
`projects/rocprofiler-compute/src/roofline/benchmark/gfx9/benchmark_gfx9_base.py`. Every thread
reads its own 16-byte elements (`__uint128_t`) in a grid-stride loop with
`__builtin_nontemporal_load`, 16 loads per step; the results are combined with a bitwise OR, so
there is no floating-point work (`a_gflop = 0`). Each byte is read once per launch: no reuse, no
writes. Launch configuration as there: 256 threads per block, 128 blocks per CU, 4 GiB requested,
rounded up to whole steps (blocks × 256 threads × 16 loads × 16 B, ≈ 1.9 GB per step on the
MI300A, so 4 GiB become ≈ 5.7 GB).

**Why not `reuse` with an HBM-sized buffer.** In `reuse` every block reads the same buffer; with a
buffer larger than the Infinity Cache, thousands of blocks would read the same addresses at about
the same time and hit in L2 or the Infinity Cache instead of HBM.

**Changes for this suite.** Compiled ahead of time instead of hiprtc; kernel without
`extern "C"`; time loop over `--duration` with ROI time stamps and JSON output; argument check;
the buffer is filled with a non-zero pattern (the kernel stores to `src[0]` only if all loaded
values are zero; rocprof-compute reads uninitialized memory); no verification (`verify_ok = null`).
License: MIT (notice in the source file).

## write

**Kernel.** Own implementation, the counterpart of `read`. The threads store 16-byte elements
(`uint4`, expected `global_store_dwordx4`) in a grid-stride loop; neighbouring lanes write
neighbouring elements (coalesced). Nothing is read and there is no arithmetic (`a_gflop = 0`).
Normal stores (not non-temporal), so the memory level is selected by the buffer size.

**Grid and memory level.** Fixed grid as in `read` and `gather`: 128 blocks per CU × 256 threads.
`--working_set_bytes` selects the level, as in `gather`: buffers up to 64 MiB are written in a whole
number of rounds per launch (about 16 stores per thread, always the same value per element), larger
buffers once per launch. No L1 run: the L1 is write-through, stores go on to the L2 (confirmed
2026-11-25: `scatter` L1 runs ≈ L2 runs). With 2 MiB every XCD writes the whole buffer into its
own L2; the write-back at the end of each launch is small compared with the bytes per launch.

**Addresses (since 0.2.0).** 32-bit byte offsets from the buffer base pointer (helper `at(p, off)`),
as in `copy` 0.5.0: the compiler uses stores with a scalar base and a 32-bit vector offset instead
of 64-bit address arithmetic. 0.1.0 had 22 % VALU in `write_l2` without any value computation
besides `value()`. Inner loop (local clang 18 check, gfx942): 11 → 8 VALU instructions per store
(`value()` itself needs 2; copy 11 → 6 without a value). Limit: buffer ≤ 2 GiB (`write_hbm` therefore 2 GiB instead of 4 GiB; for `copy`
2 GiB and 4 GiB gave the same bandwidth).

**Bytes.** Nominal = bytes stored (16 B per store).

**Verification.** Every element holds a value derived from its byte offset (`value(off)`, since 0.2.0;
0.1.0: from the element index); after the timed loop the whole buffer is compared on the host.

## cached_read

**Kernel.** Own implementation, read counterpart of `write` and `copy` for the cache levels. All
blocks read the same buffer of `--working_set_bytes` (a power of two) with 16-byte loads (`uint4`,
expected `global_load_dwordx4`). Per iteration every thread makes four independent reads, one in
each quarter of the buffer at the same offset; within a wavefront the 64 lanes read 64 consecutive
elements (coalesced). Every block sweeps its quarters in steps of one block (256 elements = 4 KiB)
per iteration, starting at a pseudo-random, 4 KiB-aligned position per block (since 0.2.0), so
blocks that run at the same time read unrelated parts of the buffer and an element is only re-read
after a long reuse distance. Addresses are 32-bit byte offsets from four base pointers with a bit
mask, so the compiler uses loads with a scalar base and a 32-bit vector offset: 2 VALU instructions
per 4 loads (local clang check). The loaded values only feed an empty `asm volatile` statement (as
`shared_read --consume asm`): the compiler must perform every read, but no arithmetic follows.

**History.** 0.1.0 (2026-11-25): start element `tid + 64 · iteration`; neighbouring wavefronts
re-read each other's data one iteration later, so `cached_read_llc` (64 MiB) was served mostly by
the L2 (20.6 TB/s, L2 68 %, beyond L2 69 %) instead of the Infinity Cache. 0.2.0 (2026-11-26):
block-wise sweep with pseudo-random start positions.

**Why not `reuse`.** `reuse` (rocprof-compute) computes a modulo index for every 4-byte load and is
therefore VALU-bound for the L1 (VALU 97 %, L1 axis 34 %; AMD's own comment notes the cost);
see validation finding 15.

**Grid and levels.** Fixed grid (128 blocks per CU × 256 threads), `--iterations` (default 1000)
sets the launch length (blocks × 256 × iterations × 4 × 16 B per launch). Runs: `cached_read_l1`
16 KiB (L1: 32 KiB per CU), `cached_read_l2` 2 MiB (L2: 4 MiB per XCD), `cached_read_llc` 64 MiB
(Infinity Cache). HBM reads: `read`.

**Verification.** The buffer is filled with `value(e)` by a setup kernel. Block 0 writes per thread
the XOR of the 16 words of the last iteration, the host recomputes it; all other blocks store only
if their result equals a magic value, so the loop cannot be removed in any block.

## reuse

*Source in `src/experimental/`, not built by default: move it to `src/` and uncomment its runs in `config/runs.cfg` to run it.*

**Reference only, not part of the coverage analysis** (since 2026-11-26). The runs in `runs.cfg`
are named `ref_reuse_l1`, `ref_reuse_l2`, `ref_reuse_llc`. `cached_read` reaches the same
bandwidth on all three levels with VALU ≈ 0 (L1: 260 W less) and replaces it as the Level 1
benchmark for the cache levels.

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

## gather

**Level 2 since 2026-11-26.** Rule: Level 1 drives one path with the ideal access to its maximum;
deliberate deviations from the ideal (irregular order, partial lines, expensive operations,
conflicts) are Level 2. gather adds no new corner (on L1/L2 it loads the same axes as
`cached_read`/`copy`, with less useful data) and is the base of the irregular Level 2 family
(`scatter_add`, `triad_gather`, …). Until then Level 1 (rule of 2026-11-23: widest access per lane).

**Kernel.** Own implementation. The threads read elements of the table `t` at the addresses
given by an index array, `t[idx[k]]`, and sum them; every thread stores its sum once
(contiguous). The accesses are independent of each other; this is the counterpart of `chase`,
where every access depends on the previous one. Gather/scatter as an access type is standard
(vector machines, HPC Challenge RandomAccess, Uwe Küster's loops); no code was taken over (the
HPCC RandomAccess code has no license).

**Index.** The index visits every `stride`-th element of `t` exactly once (default stride 128 B =
one access per cache line), in a random permutation (Fisher–Yates, fixed seed), without
duplicates. Until 0.3.0 there was also `--pattern strided` (the same lines in address order);
removed in 0.4.0, see *Strided order (removed)* below.

**Grid and memory level.** Fixed grid as in `read` and `reuse`: 128 blocks per CU × 256 threads,
independent of the table size. The threads run a grid-stride loop over the accesses `k` of one
launch; access `k` reads `t[idx[k mod n]]` (`n` = index length; the position is advanced
incrementally, without a division per access). The memory level is selected by
`--working_set_bytes` (size of `t`), as in `chase`:
- tables up to 64 MiB (L1, L2, LLC): a whole number of rounds over the index, about 16 accesses
  per thread; all CUs read the whole table many times, so it stays in the target cache (the same
  principle as `reuse`). Without this, a 2 MiB table with one access per line would give only
  16 384 accesses, i.e. 64 blocks for 228 CUs (fixed in version 0.2.0);
- larger tables (HBM): every element once per launch.

**Element width.** 16-byte elements (`uint4`, expected `global_load_dwordx4`), the widest load per
lane, as in all Level 1 memory benchmarks (`read`, `write`, `shared_read`): Level 1 drives the path
to its maximum. For irregular accesses the limit is the number of requests (one cache line per
access), not the data width; 16 B per request gives the most useful data at that request rate. The
sums are integer adds (`a_gflop = 0`). Until version 0.2.0: 8-byte `double` elements (changed
2026-11-23; the measurement of 18.11. used 0.2.0).

**Bytes.** Nominal: 16 B table + 4 B index per access, plus 16 B result per thread and launch.
The hardware moves whole cache lines; with one access per line (stride 128 B) and a table larger
than the caches, the bytes moved from HBM are up to 8 times the nominal table bytes. The
counters show the difference (an example for the roofline question: nominal intensity and actual
traffic diverge). With few accesses per launch (large stride on a large table) the result stores
dominate the nominal bytes; compare such runs by `d_gaccess_per_s`. The extra field
`d_gaccess_per_s` gives the accesses per second (comparable to GUPS).

**Verification.** The sums of all threads are recomputed on the host (about as many additions as
accesses per launch). A sample with a fixed step is not enough: with `g mod n = 0` every thread
visits only one index position, so a sample can miss whole positions.

**Related benchmarks (planned).** Level 1, each as its own small benchmark: `scatter`
(irregular writes, built 2026-11-23) and `atomic` (atomic read-modify-write,
executed in the L2). Level 2, from Uwe's loops: combinations and realistic variants such as
updates with duplicates and a conflict rate (`random_accumulate` vs. `_fake`, HPCC RandomAccess
style XOR updates), `random_axpy`, several indirect streams.

**Strided order (removed in 0.4.0).** Measured 2026-11-25 (0.3.0, stride 128 B): strided and
random differ by < 5 % on every level, for gather and scatter. Stride sweep 2026-11-26 (gather, L2
2 MiB and HBM 4 GiB, strided / random):

| stride | L2 [TB/s] | HBM [TB/s] |
|---|---|---|
| 16 B | 18.5 / 3.47 | 3.62 / 0.58 |
| 64 B | 6.57 / 3.40 | 1.14 / 0.58 |
| 128 B | 3.54 / 3.36 | 0.62 / 0.60 |
| 256 B | 3.61 / 3.42 | 0.69 / 0.67 |
| 4096 B | 3.66 / 3.34 | (invalid: 1 M accesses per launch, result stores dominate, 128 MiB of lines fit the Infinity Cache) |

Below 128 B strided accesses share cache lines (factor ≈ accesses per line: 16 B coalesced like
`read`, 64 B two per line); from 128 B on every access touches its own line in both orders and the
order does not matter, in the caches and in HBM; 4096 B on the L2 shows no channel or set conflict
effect. Strided therefore adds no Level 1 point: coalesced it duplicates `read`/`cached_read`,
partial-line use is an inefficiency (Level 2 candidate, `level2_catalog`).

## scatter

**Kernel.** Own implementation, the mirror image of `gather`. The threads write 16-byte elements
(`uint4`, the widest store per lane, see `gather`) to the table `t` at the address given by the
index: `t[idx[k]] = value(k)`. The value is computed from `k`, so apart from the index nothing is
read: 16 B irregular write + 4 B index per access, as `gather` has 16 B irregular read + 4 B index.
The accesses are independent of each other; there is no arithmetic (`a_gflop = 0`).
Uwe Küster's form `a(index(i)) = b(i)` adds a sequential read stream `b`; that combines two access
patterns and therefore belongs to Level 2 (version 0.2.0 still read `b`, changed 2026-11-23).

**Index, grid, memory level.** Exactly as in `gather`: the index visits every `stride`-th element
of `t` once, as a random permutation (same seed; strided order removed in 0.4.0, see `gather`); fixed
grid of 128 blocks per CU × 256 threads; tables up to 64 MiB are written in a whole number of
rounds per launch (about 16 accesses per thread), larger ones once per launch. In the cache runs
the same element of `t` is therefore written several times per launch, always with the same value
`value(k)`; concurrent writes of the same value are harmless.

**Bytes.** Nominal: 16 B written to `t` + 4 B index per access. With one
access per line (stride 128 B) every write touches its own cache line and writes only 16 B of it;
what the hardware moves (partial-line writes, possibly read-for-ownership) shows in the counters.
`d_gaccess_per_s` gives the accesses per second.

**Verification.** Every word of `t` is initialised with `0xFFFFFFFF`. After the timed loop the
whole table is compared with the expected one: `t[idx[k]] == value(k)` for every `k`, and every other element still
untouched. This also detects writes to wrong positions.

## shared_read

**Kernel.** Own implementation. Every block (256 threads) fills 16 KiB of LDS (`__shared__`,
1024 elements of 16 B) once and then reads it in a loop: per iteration every thread makes four
independent 16-byte reads (`uint4`, expected `ds_read_b128`) at constant offsets 0, 4, 8 and
12 KiB from a start element `base`. Within a wavefront the 64 lanes read 64 consecutive
16-byte elements, so the reads are free of bank conflicts. `base` moves by one wavefront
(64 elements = 1 KiB) per iteration; without this shift the compiler could hoist the reads out
of the loop.

**`--consume asm|sum`** (since 0.2.0). How the loaded values are used, so that the compiler
cannot remove the reads:
- `asm` (default): the 16 registers of an iteration feed an empty `asm volatile` statement.
  The compiler must perform every read in every iteration but emits no instruction for the use,
  so the VALU stays close to 0 and the run is a pure LDS point.
- `sum`: the values are summed with three-operand adds (expected `v_add3_u32`, two VALU
  instructions per read; measured VALU ≈ 40 %), as in version 0.1.0. Run `test_shared_read_sum`;
  compared with `asm` it separates the power of the VALU from that of the LDS reads.

Why not a dependent chain like `shared_read_rocprof` (`index = buf[index]`), only with 16 B: then
every wavefront has only one read in flight, and whether the LDS is saturated depends on the
latency and the occupancy; that is the principle of `chase`, not of a bandwidth benchmark.

**Grid.** Fixed grid as in `read` and `gather`: 128 blocks per CU × 256 threads. With 16 KiB per
block, 4 blocks (16 wavefronts) fit on a CU at the same time (64 KiB LDS per CU).
`--iterations` (default 1000) sets the launch length: per launch
blocks × 256 × iterations × 4 × 16 B (≈ 0.48 TB on the MI300A, a few milliseconds).

**Bytes.** `a_gbyte` / `d_gbyte_per_s` are LDS bytes, not memory traffic. Theoretical peak:
128 B per clock and CU (32 banks × 4 B), ≈ 61 TB/s at 2100 MHz with 228 CUs; the log line gives
the share of this value. The SoL counter formula in the counter wishlist uses the same peak.

**Verification.** Block 0 writes the result of every thread; the host recomputes all 256 values
(same fill pattern and arithmetic; for `asm` the XOR of the 16 words of the last iteration,
for `sum` the XOR of the four sums). All other blocks store only if their result equals a magic
value, so the loop cannot be removed in any block.

**Why not the rocprof-compute kernel.** See `shared_read_rocprof`: it reads single bytes in a dependent
chain and counts 4 B per read, so it measures the LDS access rate rather than the data bandwidth;
the original also reads out of bounds (result depends on leftover LDS contents, see
`shared_read_rocprof_prefill`). It is kept as a reference run for comparison.

## shared_read_rocprof_prefill

*Source in `src/experimental/`, not built by default: move it to `src/` and uncomment its runs in `config/runs.cfg` to run it.*

**Test only, not part of the coverage analysis.** Runs the unchanged `LDS_bw` kernel (see
`shared_read_rocprof`) after a prefill kernel has written the whole LDS of every CU
(`--prefill zero|random`, or `none` = no prefill). The threads 64–255 of `LDS_bw` start outside
`shmem[64]` and read bytes the kernel never writes, so they read whatever earlier kernels left in
the LDS.

**Motivation (2026-11-24).** The LDS peak reported by rocprof-compute differs between runs: about
57.5 TB/s in older roofline results, 53.2 TB/s in a recent run, 42.0 TB/s in a run together with
`gemm` (which uses the LDS heavily). Hypothesis: the out-of-range reads make the result depend on
leftover LDS contents (zeros → threads join the 64-byte chain, no bank conflicts; other data →
random jumps, bank conflicts). If `zero` gives about the `--variant fixed` value and `random` a lower
one, the AMD reference value is not reproducible.

**Prefill kernel.** Own code: `16 × CUs` blocks of 256 threads, each with 64 KiB dynamic LDS (the
whole LDS of a CU), write zeros or pseudo-random words and synchronise. It runs once before the
warm-up; `LDS_bw` never writes the bytes in question, so the pattern stays for all launches.
Whether reads beyond the workgroup's LDS allocation return the prefilled data or 0 depends on the
hardware; if all three variants give the same result, the reads stay inside the allocation and are
zero/garbage-independent.

## shared_read_rocprof

*Source in `src/experimental/`, not built by default: move it to `src/` and uncomment its runs in `config/runs.cfg` to run it.*

**Reference only, not part of the coverage analysis.** The run in `runs.cfg` is named
`ref_shared_read_rocprof`.

**Kernel.** `LDS_bw`, taken unchanged (except `extern "C"`) from the roofline benchmark of
rocprofiler-compute, github.com/ROCm/rocm-systems, branch develop, commit 6f44164,
`projects/rocprofiler-compute/src/roofline/benchmark/benchmark_base.py`. Thread 0 fills 64 bytes
of LDS as a chain (`shmem[i] = i+1`, `shmem[63] = 0`); every thread then follows
`index = shmem[index]` for 2000 steps. Each read depends on the previous one, reads 1 byte
(`ds_read_u8` expected) and is counted as 4 bytes (`sizeof(float)`) by rocprof-compute. Threads
with `tid >= 64` start outside the 64-byte array (as in the original). Launch configuration as
there: 256 threads, 128 blocks per CU, 2000 iterations.

**What it measures.** Rather latency × parallelism of dependent LDS reads than bandwidth.
`d_gbyte_per_s` follows the rocprof-compute counting (4 B per read) so the value can be compared
with the LDS peak that rocprof-compute reports; `d_gbyte_per_s_actual` (1 B per read) and
`d_gread_per_s` (reads per second) are given as well.

**Changes for this suite.** Compiled ahead of time instead of hiprtc; time loop over
`--duration` with ROI time stamps and JSON output; argument check; no verification
(`verify_ok = null`). License: MIT (notice in the source file).

**Out-of-range reads and `--variant`.** `shmem` has 64 entries, but with 256 threads per block the
start index `index = tid` reaches 255: threads 64–255 read outside the array (undefined behaviour in
C++). On the hardware they read whatever lies beyond the array in the block's LDS allocation (or 0),
so their chain is undefined and may cause bank conflicts that depend on leftover LDS contents.
`--variant fixed` (own change, version 0.2.0) runs the same kernel with `index = tid % 64` as the only
difference; the run `ref_shared_read_rocprof_fixed` compares both. Same result: the bug exists but does
not change the number. Different result: the AMD reference value is distorted by the bug. The byte
counting (4 B counted per 1-byte read) is a separate, questionable but explainable choice (one 4-byte
bank slot per access, as in the rocprof-compute counter formula), not a bug.

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
