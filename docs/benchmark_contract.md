status: draft
version: 0.6.0

# Benchmark Contract and Wrapper Interfaces

This document specifies the interface between the wrapper and the benchmarks, the input files the wrapper reads and the output it produces. Every benchmark must comply with it, so the wrapper can stay generic and new benchmarks can be added without changing it (Q5).

Guiding constraint: **no external dependencies** (R1). The wrapper is written in Bash and uses only standard tools (`awk`, `sed`, `grep`). Therefore:

- JSON is used **only as machine output** (benchmark result line, `results.jsonl`). It is written with `printf` and read with `awk`, which is only possible because it is kept flat.
- All **input files** (metadata, component config, run config) use line-based formats that Bash reads natively.
- Human-readable output is **TSV**.

---

## 1. Directory layout

```
suite/
    run_suite.sh               the wrapper (Sections 9-14)
    build.sh                   builds all benchmarks (Section 2)
    lib/                       helpers of the wrapper (Bash library and awk scripts)
    docs/                      all documentation, including per-benchmark docs
    src/
        <name>.cpp             one file per benchmark: metadata block (Section 3) + source
    build/                     built binaries (generated, not versioned)
    tools/
        sampler.c              telemetry sampler (Section 13)                    [not implemented yet]
        plan_counter_passes.sh generates the counter config (Section 13)       [not implemented yet]
    config/
        component.cfg          component config (Section 8)
        counters_<arch>.cfg    counter config (Section 13)                       [not implemented yet]
        runs.cfg               run config (Section 9)
    results/                   result directories (generated, not versioned)
    tests/                     tests of wrapper and build script with mock benchmarks (no GPU needed)
```

- A benchmark is exactly one file. `<name>` is the benchmark name used in the run config and the name of the built binary.
- Documentation is kept out of `src/` so that only what is needed to build and run is shipped with the benchmarks.
- The wrapper reads the metadata from the source at runtime, so `src/` must be present on the system (it is anyway, since benchmarks are built there).

---

## 2. Build

- Build and run are separate steps. The wrapper never compiles; it only checks that the binary exists.
- All benchmarks are built by one build script with a common, explicit flag set:

  ```
  hipcc -O3 -std=c++17 -ffp-contract=fast --offload-arch=<arch> <extra_build_flags> src/<name>.cpp -o build/<name>
  ```

- Usage: `build.sh [--component <file>] [<name> ...]` builds all benchmarks or the given ones. The compiler can be replaced with the environment variable `HIPCC`. The metadata of each benchmark is validated before it is built; benchmarks with invalid metadata are not built.

- `-std=c++17` and `-ffp-contract=fast` are kept explicit on purpose, even if current compilers default to them, so the build does not depend on compiler defaults (Q2).
- `<arch>` comes from the component config (Section 8), `<extra_build_flags>` from the benchmark metadata (normally empty).
- The same build script builds the telemetry sampler from `tools/sampler.c` against the amdsmi library shipped with ROCm (`-lamd_smi`). This is not an external dependency, since ROCm is required anyway.

---

## 3. Metadata (block in `<name>.cpp`)

The metadata is a comment block at the very beginning of the source file:

```cpp
// ---- benchmark metadata ----
// benchmark=fma
// level=1
// component=GPU/Compute/VALU
// description=Four independent multiply-add chains per thread; compute pole of the design space
// tags=compute-bound;valu
// version=0.3.0
// precisions=fp64,fp32,fp16,int64,int32,int16,int8
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.ops_per_element=1000000|>= 1|Multiply-adds per element; controls arithmetic intensity
// param.precision=fp64|see precisions|Data type of the computation
//
// # Expectations (checked in validation mode)
// expect_isa.fp64=v_fma_f64
// expect_isa.fp32=v_fma_f32
// expect_counters=
// ---- end metadata ----

#include <hip/hip_runtime.h>
...
```

Block rules:

- The block is the first thing in the file (no lines before the start marker).
- Start and end markers are exactly `// ---- benchmark metadata ----` and `// ---- end metadata ----`.
- Every line inside starts with `//`. After removing the leading `// ` (or a bare `//`), the content is line-based `key=value`, with `#` for comment lines and empty lines ignored, the same syntax as the component config.
- The wrapper extracts the block with `sed` and parses it line by line; it is **not** executed as shell code.
- Build and usage instructions no longer belong in the file header; they follow from this contract and from `--help <benchmark>`.

| Key | Required | Meaning |
|---|---|---|
| `benchmark` | yes | Name; must equal the file name without `.cpp` |
| `level` | yes | 1, 2 or 3 (benchmark tier) |
| `component` | yes | Targeted component, as path in the structure of the design document |
| `description` | yes | One line, shown by `--list` and `--help <name>` |
| `tags` | yes | `;`-separated list |
| `version` | yes | Semantic version of the benchmark |
| `precisions` | if the benchmark has `--precision` | Precisions the benchmark implements |
| `extra_build_flags` | no | Additional compiler flags (escape hatch, normally empty) |
| `param.<name>` | one per parameter | `<default>\|<valid values>\|<effect>`; `required` as default for mandatory parameters |
| `expect_isa.<precision>` | no | `;`-separated mnemonics that must appear in the disassembly; without precision suffix (`expect_isa=`) for benchmarks without `--precision` |
| `expect_counters` | no | **Placeholder.** Format is defined together with the counter selection |

Rules:

- Flat values only, one key per line, values contain no line breaks and no `|` (except as separator in `param.*`).
- `param.*` entries are **descriptive** (help output, Q4). Argument validation lives in the benchmark code (`--check`, Section 4). To keep both consistent, `check` mode verifies that every documented parameter with its default is accepted by `--check`.
- Runs whose `--precision` is not in the intersection of the benchmark's `precisions` and the component config's `precisions` are skipped with a notice.

---

## 4. Benchmark invocation (CLI)

- Every parameter is passed as two separate arguments: `--duration 30`. Not allowed: `--duration=30`, positional arguments (`./bench fp64 1024`), short options.
- `--duration <seconds>` is **mandatory**. All other parameters have defaults.
- A parameter that selects the precision is always called `--precision`.
- Unknown arguments, missing values or invalid values: error message on stderr, exit code 1.
- `--check`: validate all arguments and exit with 0 (valid) or 1 (invalid). No GPU access, no stdout output. Must work on a login node.
- **Device selection:** the wrapper sets `HIP_VISIBLE_DEVICES=<n>` for each benchmark process. The benchmark only sees this one GPU and always uses device 0. Benchmarks contain no device selection code. This also enables parallel execution on several GPUs later (one process per GPU).
- The benchmark does not know the wrapper mode. Modes only change what the wrapper passes and does around the run.

---

## 5. Benchmark output

The benchmark's stdout is the **machine channel to the wrapper**. Users do not see it directly; the wrapper captures it.

- **stdout:** exactly one line, a flat JSON object. Values are numbers, strings, `true`, `false` or `null`. No nested objects or arrays.
- **stderr:** free text for humans. The wrapper stores it but does not parse it.

### Field prefixes

| Prefix | Meaning | Written by |
|---|---|---|
| `a_` | analytic, computed from formulas | benchmark |
| `d_` | derived rates (`a_` / runtime) | benchmark |
| `e_` | effective parameters: the values actually used, including those that differ from the request or depend on the hardware (e.g. truncated working set, number of blocks) | benchmark |
| `m_` | measured (counters, power, frequency) | wrapper |

### Required fields

| Field | Type | Meaning |
|---|---|---|
| `runtime_s` | number | Duration of the timed loop (steady clock) |
| `iterations` | number | Kernel launches in the timed loop |
| `roi_start_unix` | number | Start of the timed loop, Unix time in seconds (system clock, sub-second precision) |
| `roi_end_unix` | number | End of the timed loop, same clock |
| `roi_start_mono_ns` | number | Start of the timed loop, monotonic clock in nanoseconds (`steady_clock`, i.e. `CLOCK_MONOTONIC` on Linux) |
| `roi_end_mono_ns` | number | End of the timed loop, same clock |
| `e_pci_bus_id` | string | PCI address (BDF, e.g. `0000:42:00.0`) of the device used, from `hipDeviceProp_t`; used to map the device to the sampler data |
| `a_gflop` | number | Total GFlop/GIop in the timed loop |
| `a_gbyte` | number | Total GB moved (10^9 bytes) |
| `a_intensity` | number | `a_gflop / a_gbyte` |
| `d_gflop_per_s` | number | |
| `d_gbyte_per_s` | number | |
| `verify_ok` | bool/null | `null` if the benchmark has no verification |

Additional fields are allowed (e.g. `precision`, `op_type`, `e_blocks`, `e_working_set_bytes`). If an effective value differs from the requested one, it must be reported as an `e_` field. Field names match `[A-Za-z_][A-Za-z0-9_]*`; the wrapper field names (Section 10) and the prefix `m_` are reserved. Fields with reserved or invalid names make the run `error` and are left out of the result record.

### Contract check by the wrapper

The wrapper checks the output of **every** run, in all modes. A violation makes the run `error` with the reason `contract: ...`:

- exactly one non-empty line on stdout, valid flat JSON (no `nan`/`inf`, no nested values)
- all required fields present with the right type; `e_pci_bus_id` has the form `0000:42:00.0`
- `iterations` is an integer >= 1; `runtime_s` >= requested `duration`
- ROI end > ROI start in both clocks; ROI length matches `runtime_s` (monotonic: 1 % or 1 ms, Unix: 1 % or 10 ms)
- `d_gflop_per_s = a_gflop / runtime_s`, `d_gbyte_per_s = a_gbyte / runtime_s`, `a_intensity = a_gflop / a_gbyte` (1 % tolerance)
- exit code 0 with `verify_ok=false` makes the run `invalid`

```json
{"runtime_s": 30.012, "iterations": 412, "roi_start_unix": 1793280121.4031, "roi_end_unix": 1793280151.4152, "roi_start_mono_ns": 3209756798359705, "roi_end_mono_ns": 3209786810359705, "e_pci_bus_id": "0000:01:00.0", "a_gflop": 1234567.0, "a_gbyte": 1.35, "a_intensity": 914494.1, "d_gflop_per_s": 41135.9, "d_gbyte_per_s": 0.045, "verify_ok": null, "precision": "fp64", "op_type": "flop", "e_blocks": 1824, "e_threads": 256}
```

---

## 6. Timing behaviour

- Warmup is not part of the timed loop.
- The timed loop runs at least `duration` seconds and contains **at least one** kernel launch, so a very short `duration` (e.g. `0.001`, used by `check` mode) must work.
- A single launch should be much shorter than `duration` to keep the overshoot small.
- The ROI (region of interest) is exactly the timed loop. It is marked in two clocks: `roi_*_mono_ns` (monotonic) is authoritative and is used by the wrapper to cut all monitoring data to this window (Section 13); `roi_*_unix` is for humans and for correlation with external logs.
- Verification runs after the timed loop, outside the ROI.
- Reproducibility: random values only with a fixed seed. Identical parameters must produce identical `a_` values.

---

## 7. Exit codes

| Code | Meaning | Wrapper reaction |
|---|---|---|
| 0 | success | continue |
| 1 | error (arguments, HIP, memory) | log error, continue with next run |
| 2 | variant not supported on this hardware | skip, print notice (Q4) |
| 3 | verification failed | keep result, mark as invalid |

`--check` only returns 0 or 1; code 2 can only be detected by a real run.

---

## 8. Component config

Line-based `key=value`, `#` for comment lines. Parsed line by line, not executed.

```
# MI300A (gfx942)
arch=gfx942
gpus=4
precisions=fp64,fp32,fp16,int64,int32,int16,int8

# Memory hierarchy, in bytes
lds_bytes_per_cu=65536
l1_bytes_per_cu=32768
l2_bytes_per_xcd=4194304
llc_bytes=268435456
hbm_bytes=137438953472
```

| Key | Meaning |
|---|---|
| `arch` | Target architecture for the build |
| `gpus` | Number of GPUs available on the node |
| `precisions` | Precisions supported by the hardware |
| `lds_bytes_per_cu` | LDS size per compute unit |
| `l1_bytes_per_cu` | L1 (vector) cache size per compute unit |
| `l2_bytes_per_xcd` | L2 cache size per XCD |
| `llc_bytes` | Last level cache (Infinity Cache) size per GPU |
| `hbm_bytes` | Main memory (HBM) size per GPU |

---

## 9. Run config

One line per run:

```
# Format:  <label> <benchmark> <arguments ...>
fma_fp64        fma     --duration {DURATION} --ops_per_element 1000000 --precision fp64
fma_fp32        fma     --duration {DURATION} --ops_per_element 1000000 --precision fp32
#fma_int32      fma     --duration {DURATION} --ops_per_element 1000000 --precision int32
```

Rules:

- Fields are separated by whitespace. Values contain no whitespace and no quotes.
- Comments only as full lines starting with `#`. Empty lines are ignored.
- `label`: unique within the file, characters `A-Z a-z 0-9 _ -` only (used in file and directory names). Used in logs and for `--only`.
- `benchmark`: name of a benchmark in `src/` (Section 1).
- `{DURATION}` is replaced by the duration of the current mode (Section 11). A fixed value instead of the placeholder takes precedence, except in `check` mode, where every `--duration` is replaced by the check duration. A line without `--duration` is an error, as is `{DURATION}` as value of another option.
- Arguments must be pairs `--key value`; each option at most once; `--check` is not allowed.
- Options not documented as `param.*` in the metadata produce a warning.
- No sweeps: one line is exactly one run. Parameter studies are written as several lines with descriptive labels.
- No device in the line. Device assignment is done by the wrapper.

---

## 10. Wrapper output

### Terminal (wrapper stdout)

A TSV overview, one row per run (label × repetition), printed as runs finish. With `--repeat > 1`, an aggregated table (one row per label) follows at the end. Notices and errors of the wrapper go to its stderr.

```
label	rep	benchmark	device	status	runtime_s	d_gflop_per_s	d_gbyte_per_s	m_avg_power_w
fma_fp64	1	fma	0	ok	30.01	41135.9	0.045	512.3
```

On the terminal the columns are aligned; `summary.tsv` contains the same rows tab-separated. For every run that is not `ok`, the reason is printed on stderr below the row.

`status` is one of:

| Status | Source |
|---|---|
| `ok` | exit code 0 |
| `error` | exit code 1, or output violates the contract |
| `skipped` | exit code 2, or filtered out by the precision filter (precision not implemented by the benchmark or not supported by the component config) |
| `invalid` | exit code 3 |
| `timeout` | killed by the wrapper (Section 12) |

### Result directory

Every wrapper invocation creates one result directory:

```
results/<YYYYmmdd-HHMMSS>_<mode>/
    results.jsonl          internal working file: one line per run, appended after each run
    results.csv            for analysis: generated from results.jsonl
    aggregates.csv         for analysis: statistics per label (only with --repeat > 1)
    summary.tsv            human overview, one row per run (as on the terminal)
    aggregates.tsv         human overview, one row per label (only with --repeat > 1)
    wrapper.log            wrapper log: start/end, plan, notices, errors
    config/                copies of run config, component config, counter config and the source of every benchmark used
    telemetry.csv          raw sampler output for all GPUs over the whole invocation (Section 13)
    idle/                  idle baseline measurement (Section 12)
    runs/<label>/rep<i>/pass<k>/
        command.txt        exact invocation incl. environment (e.g. HIP_VISIBLE_DEVICES=0 ./fma --duration 30 ...)
        stdout.json        raw benchmark output
        stderr.log         benchmark stderr
        counters.csv       raw rocprofv3 output, per kernel start (passes k >= 1, validation mode only)
```

Run directories are created only for executed runs (not for runs filtered by the precision filter). In `check` and `production` mode there is only `pass0`. In `validation` mode, `pass0` is the run without profiler and `pass1` … `pass<k>` are the counter passes (Section 13).

### `results.jsonl`

One flat JSON line per run: the benchmark's JSON line extended by wrapper fields. It can be appended line by line without knowing all fields in advance, so it is safe against an abort (e.g. Slurm time limit).

| Field | Meaning |
|---|---|
| `label` | Label from the run config |
| `rep` | Repetition (round) number, starting at 1 |
| `benchmark` | Benchmark name |
| `benchmark_version` | `version` from the metadata |
| `mode` | Wrapper mode |
| `device` | Physical device index (value of `HIP_VISIBLE_DEVICES`) |
| `cmd` | Exact invocation (as in `command.txt`) |
| `exit_code` | Exit code of the benchmark (`null` for `timeout` and precision-filtered runs) |
| `status` | see table above |
| `reason` | why the run is not `ok` (empty if ok) |
| `m_*` | Measured values, cut to the ROI window |

For skipped or failed runs, benchmark fields are missing; wrapper fields are always present.

### `results.csv`

- Generated from `results.jsonl` and **rewritten after every run**, so it is current even after an abort.
- Columns: union of all fields of all lines, wrapper fields first. Missing values are empty cells (read as `NaN` by pandas).
- One row per run; repetitions are distinguished by the `rep` column, not by separate files.
- Quoting follows RFC 4180: string values are always in double quotes, numbers and booleans are not, `null` is an empty cell. This is read correctly by pandas (`pd.read_csv`).
- Analyses should filter on `status == "ok"`: rows of other runs contain partial or no benchmark fields.

### Aggregation (`aggregates.csv`)

Computed at the end, when all repetitions of a label are available. Only repetitions with status `ok` are included.

| Fields | Treatment |
|---|---|
| `d_*`, `m_*`, `runtime_s` | statistics: `<field>_mean`, `<field>_std`, `<field>_min`, `<field>_max`, `<field>_cv` |
| `e_*`, `a_intensity` | must be identical in all repetitions; copied once, warning if they differ |
| other `a_*`, `iterations` | not aggregated (depend on the number of launches, which varies with the duration-based loop) |

- `std` is the sample standard deviation (n−1). `cv = std / mean`.
- `n_ok` is the number of included repetitions; with `n_ok < 2`, `std` and `cv` are empty.
- No threshold on `cv` for now; it is reported so a realistic threshold can be derived from validation runs later.

---

## 11. Wrapper modes

Selected with `--mode check|validation|production`.

| | `check` | `validation` | `production` |
|---|---|---|---|
| Purpose | does every run work and comply with this contract? | does every benchmark stress what it claims to? | training runs for Powersched |
| `{DURATION}` (default) | 0.001 s (all runs) | 10 s | 120 s |
| Pause between runs (default) | 0 s | 30 s | 30 s |
| Monitoring | none | telemetry (pass 0) + hardware counters (passes 1…k) | telemetry only |
| Checks | exit code, JSON readable, required fields, ROI plausible, documented parameter defaults accepted | `expect_isa`, peak plausibility, counter signature | exit code, `verify_ok` |

- Mode durations and pauses have defaults and can be overridden with the wrapper's `--duration` and `--pause`.
- Before every mode, the wrapper calls each run once with `--check` and aborts before any GPU work if an argument is invalid.
- In `validation` mode every run is executed `1 + k` times: pass 0 without profiler, then one pass per counter set (Section 13).
- Typical workflow: `check` after every change, `validation` once per new benchmark or system, `production` for training.

---

## 12. Wrapper flow

### Options

| Option | Meaning |
|---|---|
| `--mode check\|validation\|production` | Mode (Section 11), required for a run |
| `--config <file>` | Run config |
| `--component <file>` | Component config |
| `--duration <s>` | Overrides the mode's default for `{DURATION}` |
| `--pause <s>` | Overrides the mode's default pause between runs |
| `--repeat <n>` | Number of repetitions (rounds), default 1 |
| `--shuffle [<seed>]` | Shuffle the order within each round, fixed seed |
| `--only <label>[,<label>...]` | Run only these labels |
| `--device <n>` | GPU index, set as `HIP_VISIBLE_DEVICES` (default 0; must be smaller than `gpus`) |
| `--timeout <s>` | Per-run timeout (default `duration × 2 + 60 s`) |
| `--results <dir>` | Where result directories are created (default `results/`) |
| `--src-dir <dir>`, `--bin-dir <dir>` | Benchmark sources and binaries (default `src/`, `build/`); used by the tests |
| `--dry-run` | Stop after the plan (phase 4) |
| `--help [<benchmark>]`, `--list` | Help (Section 14) |

### Phases

1. **Arguments:** parse and check the wrapper options.
2. **Load:** component config, run config, metadata of every referenced benchmark.
3. **Validate (no GPU):** component config, run config syntax, unique labels, metadata, benchmark source and binary exist, `--duration` present, precision filter (filtered runs become `skipped`, not an error), `--check` for every run. In `check` mode additionally: every documented parameter default (`param.*`) is accepted by `--check`. **All** errors are collected and reported together; if there is any, the wrapper exits without starting a run.
4. **Plan:** substitute `{DURATION}`, assign devices, form the run list (rounds × runs). Print the plan and the estimated total duration (runs + pauses + idle baseline), to help choosing the Slurm time limit.
5. **Prepare:** create the result directory, copy configs and sources, start the sampler (not in `check` mode), verify the device mapping (Section 13).
6. **Idle baseline:** short monitoring without load (not in `check` mode), stored in `idle/`, as reference (e.g. idle power per GPU).
7. **Execute:** for each run and each pass: start benchmark (in passes k >= 1 under rocprofv3) → evaluate (exit code, JSON, required fields) → cut telemetry and counters to the ROI → after the last pass: merge passes, append to `results.jsonl`, rewrite `results.csv`, print TSV row → pause.
8. **Finish:** stop the sampler, aggregation (with `--repeat > 1`), final tables, wrapper exit code.

### Repetitions and order

- With `--repeat <n>`, runs are executed in **rounds**: all runs once, then all runs again, and so on. Repetitions of one label are therefore spread over time and include slow effects (heating, frequency changes, other jobs).
- Default order within a round is the order of the run config. `--shuffle` shuffles each round with a fixed seed (logged, reproducible). The seed of round `r` is `seed + r`, so the rounds differ. The wrapper uses its own random generator (Park-Miller), because `srand`/`rand` are not reproducible in every awk implementation.

### Error handling

- **Timeout:** a run gets SIGTERM after `duration × 2 + 60 s` (or `--timeout`) and SIGKILL 10 s later; status `timeout`, continue with the next run.
- **Failed run:** logged with its status, continue with the next run.
- **Incremental results:** every run is written immediately, nothing is held back until the end.
- **Abort (`SIGTERM`/`SIGINT`, e.g. Slurm time limit):** the wrapper stops the running benchmark and monitor processes, notes the abort in `wrapper.log` and exits. All completed runs are preserved.

### Wrapper exit code

| Code | Meaning |
|---|---|
| 0 | all runs `ok` or `skipped` |
| 1 | invalid options or validation failed; nothing was run |
| 4 | at least one run `error`, `invalid` or `timeout` |
| 5 | aborted by signal |

---

## 13. Monitoring

### Levels

| Level | Tool | Modes | What |
|---|---|---|---|
| Telemetry | own sampler (`tools/sampler.c`, amdsmi C library) | `validation` (pass 0), `production` | power, energy, clocks, temperatures, utilization, VRAM, throttling, per GPU over time |
| Hardware counters | `rocprofv3`, per kernel start | `validation` (passes 1…k) | instruction mix, pipeline utilization, cache and memory traffic per kernel start |

The amd-smi command-line tool is not used: its timestamps are Unix time with one-second resolution, which is too coarse to cut data to the ROI.

### Sampler

- **One process for all GPUs** per wrapper invocation, started in phase 5 and stopped in phase 8. One common timeline, low overhead, ready for parallel runs later.
- **Interval** configurable (default e.g. 100 ms, `--sample-interval`).
- **Quantities per GPU and sample:** current power, accumulated energy counter, GFX clock, memory clock, temperatures, GFX and memory utilization, VRAM usage, throttling (power/thermal violation) status.
- **Two timestamps per sample:**
  - `mono_ns` (`CLOCK_MONOTONIC`): same clock as the benchmark's `steady_clock`; used for cutting to the ROI.
  - `unix_ns` (`CLOCK_REALTIME`): for humans and correlation with external logs.
- **Output:** one CSV line per GPU and sample in `telemetry.csv`, with a `bdf` column (PCI address) identifying the GPU.
- **Energy over the ROI:** the hardware energy counter accumulates continuously. The wrapper interpolates the counter linearly to `roi_start_mono_ns` and `roi_end_mono_ns` and takes the difference: `m_energy_j = E(end) − E(start)`. The only error is the interpolation at the two boundaries, at most a fraction of one sample interval. Average power over the ROI is `m_energy_j / runtime_s`.
- **Device mapping:** HIP and amdsmi number devices independently, so device indices may differ. The common key is the PCI address: the benchmark reports `e_pci_bus_id`, the sampler reports `bdf`. The wrapper uses only telemetry rows whose `bdf` matches; a missing match is an error. In phase 5 the wrapper additionally checks that every device it intends to use is visible to the sampler.

### Hardware counters

- Collected with `rocprofv3` in dispatch mode (per kernel start). This requires the profiler inside the benchmark process; reimplementing this is out of scope.
- **Passes in `validation` mode:**
  - **Pass 0:** benchmark without profiler, telemetry only. Telemetry in the results comes from this pass only, because the profiler serializes kernel starts and inserts gaps, which changes power and clocks.
  - **Passes 1…k:** benchmark under `rocprofv3`, one counter set per pass. The sampler keeps running, but its data from these passes is not used for results.
  - Merging relies on deterministic benchmarks (Section 6).
- **Counter types:**
  - *Hardware counters* (`Block:` in `rocprofv3 --list-avail`): occupy hardware counter registers; their combination determines the number of passes.
  - *Derived counters* (`Expression:`): computed by rocprofv3 from hardware counters (e.g. `VALUBusy`, `MfmaUtil`, `OccupancyPercent`, `FETCH_SIZE`, `VmemLatency`). Included where they help end users; counters based on `accumulate(...)` (latencies, mean occupancy) can only be obtained this way.
  - The counter selection is oriented on the metrics of rocprof-compute (its gfx942 panel configurations), which is not used itself.

### Counter config (`config/counters_<arch>.cfg`)

Line-based `key=value`, generated by `tools/plan_counter_passes.sh`, specific to an architecture.

```
# generated by plan_counter_passes.sh for gfx942
pass.1=SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR
pass.2=TCC_HIT_sum,TCC_MISS_sum,TCC_EA0_RDREQ_sum,TCC_EA0_WRREQ_sum
pass.3=VALUBusy,OccupancyPercent
...
agg.VALUBusy=wmean
agg.OccupancyPercent=wmean
```

- `pass.<k>`: comma-separated counters collected together in pass `k`.
- `agg.<counter>`: aggregation over kernel starts. `sum` (default) for quantities (counts, kilobytes); `wmean` (mean weighted by kernel duration) for ratios and rates (percentages, latencies, occupancy).

### Pass planner (`tools/plan_counter_passes.sh`)

- Input: a wish list of counters with their aggregation type.
- Packs counters greedily into passes and tests each candidate pass with `rocprofv3` on a small test kernel; if rocprofv3 rejects the combination, a new pass is started.
- Writes `counters_<arch>.cfg`. On a new system the file is regenerated with one call (Q1, Q2).

### Cutting to the ROI

- All sources are cut using the **monotonic clock** and `roi_*_mono_ns`: telemetry samples by `mono_ns`, kernel starts from rocprofv3 by their start/end timestamps.
- Telemetry: samples inside the ROI are used for averages; energy is interpolated at the boundaries (see above).
- Counters: kernel starts that lie completely inside the ROI are used.
- **Consistency check:** the number of kernel starts inside the ROI must equal `iterations` from the benchmark JSON. A mismatch (e.g. because rocprofv3 uses a different clock than assumed) produces a warning and the run is marked in `wrapper.log`.

### Measured fields (`m_*`)

| Field | Source | Meaning |
|---|---|---|
| `m_energy_j` | sampler | energy over the ROI (interpolated energy counter) |
| `m_avg_power_w` | sampler | `m_energy_j / runtime_s` |
| `m_avg_gfx_clk_mhz`, `m_avg_mem_clk_mhz` | sampler | mean clocks over the ROI |
| `m_max_temp_c` | sampler | maximum hotspot temperature over the ROI |
| `m_avg_gfx_util_pct`, `m_avg_mem_util_pct` | sampler | mean utilization over the ROI |
| `m_throttle_pct` | sampler | share of samples with power or thermal throttling |
| `m_<COUNTER>` | rocprofv3 | aggregated counter value over the ROI (`sum` or `wmean`) |
| `m_dispatches` | rocprofv3 | number of kernel starts inside the ROI |
| `m_kernel_time_s` | rocprofv3 | summed kernel duration inside the ROI |
| `m_prof_slowdown_p<k>` | wrapper | time per kernel start in pass `k` divided by that of pass 0 (`runtime_s / iterations`); 1.0 means no profiler overhead |

Raw data (`telemetry.csv`, `counters.csv`) is always kept; ratios beyond the derived counters are computed in the analysis, not in the wrapper.

### Limitations

- **Infinity Cache:** `rocprofv3` on gfx942 lists no counter block for the Infinity Cache. Hits and misses there cannot be measured directly; only traffic leaving the L2 (`TCC_EA0_*`) is visible. The Infinity Cache level appears only indirectly, e.g. in the latency steps of `chase`.
- **Sensor update rate:** the power sensor updates and averages internally at its own rate; a shorter sample interval adds no information. The rate is determined once by sampling at 1 ms and counting value changes, and the default interval is chosen accordingly.

---

## 14. Help

Help texts are generated from the metadata, so there is one source and benchmarks need no own help code.

| Call | Output |
|---|---|
| `--help` | Usage of the wrapper: modes, options, file locations |
| `--list` | One line per benchmark: name, level, component, description |
| `--help <benchmark>` | Description, tags, supported precisions and all parameters with default, valid values and effect |

---

## 15. Planned extensions (not implemented, considered in the design)

- **Portability check:** the wrapper currently checks only bash >= 4.3, the presence of `awk`, `sed`, `sort`, `timeout` and `date +%N`. A fuller preflight check (e.g. `run_suite.sh --doctor`: ROCm, `hipcc`, `amdsmi`, `rocprofv3`, versions) is planned. The awk scripts are POSIX awk and are tested with mawk; the target system uses GNU awk.

- **Counters as time series (stage 2):** collect hardware counters device-wide in the sampler (agent profiling of rocprofiler-sdk), on the same timeline as telemetry and possibly with less overhead than dispatch mode. Feasibility from a separate process and required permissions are to be tested first. The rest of the wrapper is unaffected, since all sources are cut by the monotonic clock.

- **Multiple GPUs:** the wrapper works with independent run units (benchmark, arguments, device, mode), each with its own output directory and per-device monitoring. Later, each run line is executed on all GPUs simultaneously with a synchronized start; `runs/<label>/` then gets one subdirectory per device.
- **Auto-generated component config:** read `arch`, `gpus`, precisions and cache sizes from the system (e.g. `rocminfo`, `amd-smi`).
- **Derived parameters:** working set sizes for memory benchmarks derived from the cache sizes in the component config.
- **Counter expectations:** format of `expect_counters`, once the counter selection is defined.
- **CPU benchmarks** (R7): same contract, device selection via thread count and pinning instead of `HIP_VISIBLE_DEVICES`.
