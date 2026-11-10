status: draft
version: 0.1.0

# Counter selection for gfx942 (MI300A)

Draft of the hardware counters collected in `validation` mode (contract section 13). It is the input
for the pass planner (`tools/plan_counter_passes.sh`, not implemented yet) and the starting point for
the literature-based justification of the selection.

## Method

- **Source of the selection:** the metric definitions of rocprof-compute (ROCm Compute Profiler) for
  gfx942, i.e. the panel configurations in
  `projects/rocprofiler-compute/src/rocprof_compute_soc/analysis_configs/gfx942/` of
  github.com/ROCm/rocm-systems at tag `rocm-7.2.4` (commit `97f5574fe2`, the ROCm version on the
  target system). rocprof-compute itself is not used; only the choice of counters is taken over.
- **Check:** every counter was matched against `rocprofv3 --list-avail` of the target system
  (479 entries per GPU: 265 hardware counters, 155 derived counters, 59 constants).
  `MemUnitBusy` is not available on gfx942 and was dropped.
- **Grouping:** by the components of the structure plan (compute, VALU, matrix cores, LDS, L1, L2,
  LLC/HBM), plus a time base that the ratios need.
- **Kinds:**
  - *hardware*: one hardware counter.
  - *hardware, sum over instances*: names ending in `_sum` are listed by rocprofv3 as derived, but
    are only the sum of one hardware counter over all instances (e.g. all L2 channels,
    `reduce(TCC_HIT,sum)`). They occupy hardware counter registers like a hardware counter.
  - *formula*: evaluated by rocprofv3 from other counters.
- **Aggregation over kernel starts:** counts are summed (`sum`); the formulas are ratios or
  latencies and are weighted by kernel duration (`wmean`), see the wish list.

Selected: 72 counters (48 hardware, 22 sums over instances, 2 formulas). Six further formulas
were considered and left out (see below).

## Expected signatures of the current benchmarks

What the validation mode should show if a benchmark does what it claims (to be checked on the GPU):

| Benchmark | Expected signature |
|---|---|
| `fma` (fp64) | `SQ_INSTS_VALU_FMA_F64` dominates `SQ_INSTS_VALU`; high `SQ_ACTIVE_INST_VALU`; almost no `TCP_*`/`TCC_*` traffic |
| `copy` (formerly `stream`) | few VALU instructions; `SQ_INSTS_VMEM_RD` ≈ `SQ_INSTS_VMEM_WR`; high `TCC_EA0_RDREQ_DRAM_sum` and `TCC_EA0_WRREQ_DRAM_sum` (working set >> LLC) |
| `chase` | one wavefront; very low `SQ_ACTIVE_INST_*`; L2 hit rate depends on the working set (high for `chase_l1`/`chase_l2`, low for `chase_hbm`); `TCP_UTCL1_TRANSLATION_MISS_sum` grows with the working set |
| `gemm` (blocked) | `SQ_INSTS_LDS` and `SQ_LDS_IDX_ACTIVE` > 0, FMA/MUL+ADD of the input precision; L2/fabric traffic falls with growing K (arithmetic intensity) |
| `gemm` (naive) | no LDS; much more `TCP_*`/`TCC_*` traffic than blocked at the same size |

## Time base and activity

Reference quantities for all others: cycles of the GPU (`GRBM_*`), busy CU cycles, number of wavefronts and their cycles, instructions and wait cycles. Ratios such as IPC, occupancy or utilization need these as denominator.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `GRBM_COUNT` | hardware | Tie High - Count Number of Clocks | - |
| `GRBM_GUI_ACTIVE` | hardware | The GUI is Active | - |
| `SQ_BUSY_CU_CYCLES` | hardware | Number of quad-cycles each CU is busy. Can be used to calculate the percentage of time each CU is busy. This value is returned on a per-S... | System SoL, SPI, Compute Pipeline |
| `SQ_WAVES` | hardware | Count number of waves sent to distributed sequencers (SQs). This value represents the number of waves that are sent to each SQ. This only... | Memory Chart, Wavefront |
| `SQ_WAVE_CYCLES` | hardware | The cycles spent executing waves in the CUs. This value is reported per-SE (aggregates of SIMD values) and is nondeterministic. Units are... | Memory Chart, Wavefront |
| `SQ_INSTS` | hardware | Total number of instructions issued. When used in combination with SQ_ACTIVE_INST_ANY (cycle count for executing instructions) the averag... | System SoL, Wavefront, Compute Pipeline |
| `SQ_ACTIVE_INST_ANY` | hardware | Number of cycles each wave spends working on any type of instruction. Useful in determining percentage of time spend executing wave workl... | Wavefront, Compute Pipeline |
| `SQ_WAIT_ANY` | hardware | Number of wave-cycles spent waiting for anything (per-simd, nondeterministic). Units in quad-cycles(4 cycles) | Wavefront |
| `SQ_WAIT_INST_ANY` | hardware | Number of wave-cycles spent waiting for any instruction issue. Units in quad-cycles(4 cycles). | Wavefront |

## Instruction mix

Which pipelines a kernel uses at all (vector, scalar, memory, LDS, branches, matrix). The first and coarsest signature of a benchmark.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `SQ_INSTS_VALU` | hardware | The number of VALU (Vector ALU) instructions issued. The value is returned per-SE (aggregate of values in SIMDs in the SE). See AMD ISAs ... | Memory Chart, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_MFMA` | hardware | Total number of MFMA (Matrix-Fused-Multiply-Add) instructions issued. This value is returned per-SE (aggregate of values in SIMDs in the ... | Memory Chart, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_SALU` | hardware | Total Number of SALU (Scalar ALU) instructions issued. This value is returned per-SE (aggregate of values in SIMDs in the SE). See AMD IS... | Memory Chart, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_SMEM` | hardware | Total number of SMEM (Scalar Memory Read) instructions issued. This value is returned per-SE (aggregate of values in SIMDs in the SE). Se... | Memory Chart, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VMEM_RD` | hardware | The number of VMEM (GPU Memory) read instructions issued (including FLAT/scratch memory). The value is returned per-SE (aggregate of valu... | - |
| `SQ_INSTS_VMEM_WR` | hardware | The number of VMEM (GPU Memory) write instructions issued (including FLAT/scratch memory). The value is returned per-SE (aggregate of val... | - |
| `SQ_INSTS_LDS` | hardware | Total number of LDS instructions issued (including FLAT). This value is returned per-SE (aggregate of values in SIMDs in the SE). See AMD... | Memory Chart, Instr. Mix, Compute Pipeline, LDS |
| `SQ_INSTS_BRANCH` | hardware | Total number of BRANCH instructions issued. This value is returned per-SE (aggregate of values in SIMDs in the SE). This value SHOULD NOT... | Memory Chart, Instr. Mix, Compute Pipeline |

## VALU arithmetic

Floating point and integer operations per type and precision. Shows e.g. whether `fma` really executes FMAs of the requested precision (`SQ_INSTS_VALU_FMA_F64`) and how many conversions (`CVT`) a kernel needs.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `SQ_INSTS_VALU_ADD_F16` | hardware | The number of VALU (Vector ALU) ADD/SUB instructions on float16. For maximum performance lower precision floating point ops are preferred... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_MUL_F16` | hardware | The number of VALU MUL instructions on float16 data. For maximum performance lower precision floating point ops are preferred to higher p... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_FMA_F16` | hardware | The number of VALU (Vector ALU) FMA (Fused-Multiply-Add)/MAD(Multiply-Add) instructions on float16. For maximum performance lower precisi... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_TRANS_F16` | hardware | The number of VALU transcendental instructions on float16 data. Transcendental instructions include sin, cos, exp, log, etc. For maximum ... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_ADD_F32` | hardware | The number of VALU (Vector ALU) ADD/SUB instructions on float32. For maximum performance lower precision floating point ops are preferred... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_MUL_F32` | hardware | The number of VALU MUL instructions on float32 data. For maximum performance lower precision floating point ops are preferred to higher p... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_FMA_F32` | hardware | The number of VALU (Vector ALU) FMA (Fused-Multiply-Add)/MAD(Multiply-Add) instructions on float32. For maximum performance lower precisi... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_TRANS_F32` | hardware | The number of VALU transcendental instructions on float32 data. Transcendental instructions include sin, cos, exp, log, etc. For maximum ... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_ADD_F64` | hardware | The number of VALU ADD/SUB instructions on float64. For maximum performance lower precision floating point ops are preferred to higher pr... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_MUL_F64` | hardware | The number of VALU MUL instructions on float64 data. For maximum performance lower precision floating point ops are preferred to higher p... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_FMA_F64` | hardware | The number of VALU (Vector ALU) FMA (Fused-Multiply-Add)/MAD(Multiply-Add) instructions on float64. For maximum performance lower precisi... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_TRANS_F64` | hardware | The number of VALU transcendental instructions on float64 data. Transcendental instructions include sin, cos, exp, log, etc. For maximum ... | System SoL, Roofline, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_INT32` | hardware | The number of VALU (Vector ALU) 32-bit integer (signed or unsigned) instructions. The value is returned per-SE (aggregate of values in SI... | System SoL, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_INT64` | hardware | The number of VALU (Vector ALU) 64-bit integer (signed or unsigned) instructions. The value is returned per-SE (aggregate of values in SI... | System SoL, Instr. Mix, Compute Pipeline |
| `SQ_INSTS_VALU_CVT` | hardware | The number of VALU (Vector ALU) data conversion instructions (ex. float -> int). The value is returned per-SE (aggregate of values in SIM... | Instr. Mix |

## Pipeline utilization

How many cycles the pipelines were busy (`SQ_ACTIVE_INST_*`) and how many lanes of a wavefront were active (`SQ_THREAD_CYCLES_VALU`). Distinguishes 'many instructions' from 'pipeline saturated'.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `SQ_ACTIVE_INST_VALU` | hardware | Number of cycles each wave spends working on a VALU instructions. This value represents the number of cycles each wave spends executing v... | System SoL, Compute Pipeline |
| `SQ_THREAD_CYCLES_VALU` | hardware | Number of thread-cycles used to execute VALU operations (similar to INST_CYCLES_VALU but multiplied by # of active threads). (per-simd) | System SoL, Compute Pipeline |
| `SQ_ACTIVE_INST_SCA` | hardware | Number of cycles each wave spends working on a SALU or SMEM instructions. This value represents the number of cycles each wave spends exe... | System SoL, Compute Pipeline |
| `SQ_ACTIVE_INST_VMEM` | hardware | Number of cycles each wave spends working on a VMEM instructions. This value represents the number of cycles each wave spends executing v... | System SoL, Compute Pipeline |
| `SQ_ACTIVE_INST_FLAT` | hardware | Number of cycles each wave spends working on FLAT instructions. This value represents the number of cycles each wave spends executing ins... | System SoL, Compute Pipeline |
| `SQ_ACTIVE_INST_LDS` | hardware | Number of cycles each wave spends working on LDS instructions. This value represents the number of cycles each wave spends executing inst... | LDS |
| `SQ_ACTIVE_INST_MISC` | hardware | Number of cycles each wave spends working on a BRANCH or SENDMSG instructions. This value represents the number of cycles each wave spend... | System SoL, Compute Pipeline |
| `SQ_VALU_MFMA_BUSY_CYCLES` | hardware | Number of cycles the MFMA (Matrixed-Fused-Multiply-Add) ALU is busy. This value is returned on a per-SIMD basis. | System SoL, Compute Pipeline |

## Matrix cores (for later)

Operations of the matrix cores (MFMA) per type. Zero for all current benchmarks; needed as soon as there are matrix core benchmarks.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | hardware | The number of math operation instructions on the VALU (Vector ALU) using MFMA (Matrix-Fused-Multiply-Add) and operating on F16 (float16) ... | System SoL, Roofline, Compute Pipeline |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | hardware | The number of math operation instructions on the VALU (Vector ALU) using MFMA (Matrix-Fused-Multiply-Add) and operating on BF16 (bfloat16... | System SoL, Roofline, Compute Pipeline |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` | hardware | The number of math operation instructions on the VALU (Vector ALU) using MFMA (Matrix-Fused-Multiply-Add) and operating on F32 (float32) ... | System SoL, Roofline, Compute Pipeline |
| `SQ_INSTS_VALU_MFMA_MOPS_F64` | hardware | The number of math operation instructions on the VALU (Vector ALU) using MFMA (Matrix-Fused-Multiply-Add) and operating on F64 (float64) ... | System SoL, Roofline, Compute Pipeline |
| `SQ_INSTS_VALU_MFMA_MOPS_I8` | hardware | The number of math operation instructions on the VALU (Vector ALU) using MFMA (Matrix-Fused-Multiply-Add) and operating on I8 (8 bit int)... | System SoL, Roofline, Compute Pipeline |
| `SQ_INSTS_VALU_MFMA_MOPS_F8` | hardware | The number of math operation on F8 datatype. Captures add or mul ops performed divided by 512. The value is returned per-SE (aggregate of... | System SoL, Roofline, Compute Pipeline |

## LDS

Activity of the local data share and bank conflicts. Relevant for the blocked `gemm` kernel, which works on tiles in LDS.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `SQ_LDS_IDX_ACTIVE` | hardware | Number of cycles LDS (local data store) is used for indexed (non-direct,non-interpolation) operations. This value is returned on a per-SE... | System SoL, Memory Chart, Roofline, LDS |
| `SQ_LDS_BANK_CONFLICT` | hardware | The number of cycles LDS (local data store) is stalled by bank conflicts. This value is returned on a per-SE (aggregate of values in SIMD... | System SoL, Roofline, LDS |

## Vector L1 (vL1D) and address translation

Accesses to the vector L1 cache and requests it passes on to L2 (misses), coalescing (`TA_TOTAL_WAVEFRONTS_sum` vs. accesses), and the address translation cache UTCL1. UTCL1 misses explain part of the latency of `chase` with large working sets.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `TCP_TOTAL_CACHE_ACCESSES_sum` | hardware, sum over instances | Count of total cache line (tag) accesses (includes hits and misses). Sum over TCP instances. | System SoL, Memory Chart, Roofline, vL1D |
| `TCP_TOTAL_ACCESSES_sum` | hardware, sum over instances | Total number of pixels/buffers from TA. Equals TCP_PERF_SEL_TOTAL_READ+TCP_PERF_SEL_TOTAL_NONREAD. Sum over TCP instances. | Memory Chart, vL1D |
| `TCP_TCC_READ_REQ_sum` | hardware, sum over instances | Total read requests from TCP to all TCCs Sum over TCP instances. | System SoL, Memory Chart, Roofline, vL1D |
| `TCP_TCC_WRITE_REQ_sum` | hardware, sum over instances | Total write requests from TCP to all TCCs Sum over TCP instances. | System SoL, Memory Chart, Roofline, vL1D |
| `TA_TOTAL_WAVEFRONTS_sum` | hardware, sum over instances | Total number of wavefronts processed by TA. Sum over TA instances. | Memory Chart, TA/TD, vL1D |
| `TCP_UTCL1_REQUEST_sum` | hardware, sum over instances | Total CLIENT_UTCL1 NORMAL requests Sum over TCP instances. | vL1D |
| `TCP_UTCL1_TRANSLATION_MISS_sum` | hardware, sum over instances | Total utcl1 translation misses Sum over TCP instances. | vL1D |

## L2 cache

Requests, hits and misses of the L2 cache and writebacks. Hit rate = HIT / (HIT + MISS).

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `TCC_REQ_sum` | hardware, sum over instances | Number of requests of all types. This is measured at the tag block. This may be more than the number of requests arriving at the TCC, but... | System SoL, L2 |
| `TCC_READ_sum` | hardware, sum over instances | Number of read requests. Compressed reads are included in this, but metadata reads are not included. Sum over TCC instances. | Memory Chart, L2 |
| `TCC_WRITE_sum` | hardware, sum over instances | Number of write requests. Sum over TCC instances. | Memory Chart, L2 |
| `TCC_HIT_sum` | hardware, sum over instances | Number of cache hits. Sum over TCC instances. | System SoL, Memory Chart, L2 |
| `TCC_MISS_sum` | hardware, sum over instances | Number of cache misses. UC reads count as misses. Sum over TCC instances. | System SoL, Memory Chart, L2 |
| `TCC_WRITEBACK_sum` | hardware, sum over instances | Number of lines written back to main memory. This includes writebacks of dirty lines and uncached write/atomic requests. Sum over TCC ins... | L2 |

## L2 to fabric (Infinity Cache / HBM)

Traffic leaving the L2 towards the Infinity Fabric: read and write requests by size (32/64/128 B, `TCC_BUBBLE_sum` = 128 B reads), the part that goes to DRAM, and the request levels from which rocprof-compute derives the mean fabric latency. The Infinity Cache itself has no counters on gfx942; these counters are the closest view of LLC and HBM traffic.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `TCC_EA0_RDREQ_sum` | hardware, sum over instances | Number of TCC/EA read requests (either 32-byte or 64-byte) Sum over TCC instances. | System SoL, Memory Chart, Roofline, L2 |
| `TCC_EA0_RDREQ_32B_sum` | hardware, sum over instances | Number of 32-byte TCC/EA read requests Sum over TCC instances. | System SoL, Roofline, L2 |
| `TCC_BUBBLE_sum` | hardware, sum over instances | Number of 128-byte read requests sent to EA. Sum over all TCC instances. | System SoL, Roofline, L2 |
| `TCC_EA0_WRREQ_sum` | hardware, sum over instances | Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Atomics may travel over the same interface and a... | System SoL, Memory Chart, Roofline, L2 |
| `TCC_EA0_WRREQ_64B_sum` | hardware, sum over instances | Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface. Sum over TCC instances. | System SoL, Roofline, L2 |
| `TCC_EA0_RDREQ_DRAM_sum` | hardware, sum over instances | Number of TCC/EA read requests (either 32-byte or 64-byte) destined for DRAM (MC). Sum over TCC instances. | Memory Chart, L2 |
| `TCC_EA0_WRREQ_DRAM_sum` | hardware, sum over instances | Number of TCC/EA write requests (either 32-byte of 64-byte) destined for DRAM (MC). Sum over TCC instances. | Memory Chart, L2 |
| `TCC_EA0_RDREQ_LEVEL_sum` | hardware, sum over instances | The sum of the number of TCC/EA read requests in flight. This is primarily meant for measure average EA read latency. Average read latenc... | System SoL, Memory Chart, L2 |
| `TCC_EA0_WRREQ_LEVEL_sum` | hardware, sum over instances | The sum of the number of EA write requests in flight. This is primarily meant for measure average EA write latency. Average write latency... | System SoL, Memory Chart, L2 |

## Derived quantities (rocprofv3)

Formulas evaluated by rocprofv3. Only `MeanOccupancyPerCU` and `VmemLatency` are collected: they are based on `accumulate(...)` counters and can only be obtained this way. The other six (marked *not collected*) are plain formulas of counters that are collected anyway; requesting them would occupy counter registers (more passes) for values the analysis computes itself. They are listed as a reference for the analysis.

| Counter | Kind | Description (rocprofv3) | rocprof-compute panels |
|---|---|---|---|
| `VALUBusy` | formula, *not collected* | The percentage of GPUTime vector ALU instructions are processed. Value range: 0% (bad) to 100% (optimal). | - |
| `VALUUtilization` | formula, *not collected* | The percentage of active vector ALU threads in a wave. A lower number can mean either more thread divergence in a wave or that the work-g... | - |
| `FETCH_SIZE` | formula, *not collected* | The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into acc... | - |
| `WRITE_SIZE` | formula, *not collected* | The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account. | - |
| `MeanOccupancyPerCU` | formula | Mean occupancy per compute unit. | - |
| `VmemLatency` | formula | Unit: cycles | - |
| `OccupancyPercent` | formula, *not collected* | GPU Occupancy as % of maximum. | - |
| `MfmaUtil` | formula, *not collected* | reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100 (percent) | - |

## Open points

- **Number of passes:** how many counters fit into one pass depends on the counter registers per
  hardware block (SQ, TCP, TCC, ...). The pass planner determines this on the system by trying
  combinations; the number of passes, and with it the duration of `validation` mode, is not known yet.
- **Literature:** the selection follows rocprof-compute. For the thesis it should be compared with
  the counters used in power and performance models (e.g. Guerreiro et al., AccelWattch) and reduced
  or extended accordingly; the PCA on validation data can show which counters are redundant.
- **Infinity Cache:** no counter block on gfx942 (contract section 13, limitations).
- **Formulas in the analysis:** the six formulas that are not collected must be reproduced in the
  analysis with the definitions of `rocprofv3 --list-avail` (e.g. `MfmaUtil` above); constants such
  as `SIMD_NUM` or `CU_NUM` are listed there as well.
