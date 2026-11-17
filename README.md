# Microbenchmark Suite

Benchmarks for AMD GPUs (MI300A, gfx942) and a wrapper that runs them, checks
their output and collects results, telemetry and hardware counters. No external
dependencies: HIP/C++ for the benchmarks, Bash and awk for everything else.

- Benchmarks (what they do, sources, implementation notes): [docs/benchmarks.md](docs/benchmarks.md)
- Specification of wrapper, benchmark interface and output files: [docs/benchmark_contract.md](docs/benchmark_contract.md)
- Selection of the hardware counters: [docs/counter_selection.md](docs/counter_selection.md)

## Requirements

- ROCm with `hipcc` (benchmarks) and the amdsmi library (telemetry sampler); tested with ROCm 7.2.4
- `rocprofv3` (only for `--mode profile`)
- Bash 4.3+, awk (mawk or gawk), coreutils

## Quick start

```bash
./build.sh                                   # build all benchmarks in src/ and the sampler into build/
./run_suite.sh --list                        # available benchmarks
./run_suite.sh --help fma                    # parameters of one benchmark
./run_suite.sh --mode check                  # do all runs work and follow the contract?
./run_suite.sh --mode production --dry-run   # show the plan and the estimated time
./run_suite.sh --mode production --repeat 3  # measure
./run_suite.sh --mode profile --only fma_fp64,copy_hbm   # only these runs (labels from config/runs.cfg)
```

For hardware counters, once per system on a GPU node, then profile mode:

```bash
tools/plan_counter_passes.sh                 # distribute the wished counters over profiling passes
./run_suite.sh --mode profile                # telemetry + hardware counters
```

If scripts fail with "Permission denied" after a checkout: `chmod +x run_suite.sh build.sh tools/*.sh tests/run_tests.sh tests/mock/bin/*`.

## Modes

| Mode | Duration per run | Collects | Use |
|---|---|---|---|
| `check` | minimal | contract check only | after every change |
| `profile` | medium (10 s) | telemetry + hardware counters (one extra rocprofv3 run per counter pass) | once per new benchmark or system |
| `production` | long (120 s) | telemetry | measurements, training data |

The durations are defaults and can be changed per call:

- `--duration <s>`: duration of every run whose line in `config/runs.cfg` uses `{DURATION}`
  (all runs by default), e.g. `./run_suite.sh --mode production --duration 300`.
  A run can also get a fixed duration in its line (`--duration 60` instead of `{DURATION}`);
  only check mode overrides it.
- `--pause <s>`: pause between runs (default 30 s in profile and production mode).
- `--profile-duration <s>`: duration of each counter pass in profile mode (default 2 s).
  In profile mode every run is executed once without profiler (`--duration`, with telemetry)
  and then once per counter pass under rocprofv3 (`--profile-duration`). The passes are kept
  short because there are several of them and the profiler records every kernel launch.

## Configuration

| File | Content |
|---|---|
| `config/runs.cfg` | the runs: one line per run (label, benchmark, arguments) |
| `config/system.cfg` | the hardware: architecture, number of GPUs, supported precisions, cache sizes |
| `config/counters_wishlist_gfx942.cfg` | counters to collect, in groups; input of `tools/plan_counter_passes.sh` |
| `config/counters_gfx942.cfg` | generated: counters per profiling pass |

## Results

Each call writes to `results/<date>-<time>_<mode>/`:

| File | Content |
|---|---|
| `results.csv` | one row per run: benchmark fields (`a_*` amounts, `d_*` rates, `e_*` parameters) and wrapper fields (`m_*`: energy, power, clocks, throttling) |
| `aggregates.csv` | statistics per label (with `--repeat > 1`) |
| `counters.csv` | profile mode: one row per run, pass and counter; join with `results.csv` by `label` and `rep` |
| `telemetry.csv` | raw telemetry samples of the whole call |
| `config/` | copies of all configs and benchmark sources used |

Field definitions: contract, Sections 5, 10 and 13.

## Known limitations

- Working-set sizes in `config/runs.cfg` are tuned for the MI300A (L1 32 KiB/CU,
  L2 4 MiB/XCD, Infinity Cache 256 MB). On other GPUs they have to be adapted by hand.
- The counter wish list exists only for gfx942.
- `counters.csv` covers only kernel launches that lie completely inside the timed region
  (ROI) of a run, not warm-up, initialization or the gaps between runs.
- Telemetry (`power_w`) is the socket power of the APU, including the CPU part.
- Throttling (`m_ppt_viol_pct`, `m_thm_viol_pct`) comes from firmware residency counters.
  On some nodes or firmware states the firmware does not report them (`amd-smi metric --violation`
  shows N/A); the columns are then empty (`-`). Throttling is still visible in
  `m_avg_gfx_clk_mhz` (clock below the maximum of ~2100 MHz) together with `m_avg_power_w`
  near the power limit (760 W on the MI300A).
- No Slurm integration yet: run on an allocated GPU node.

## Repository layout

```
src/          benchmarks (one .cpp per benchmark, metadata in the header)
config/       run, system and counter configs
lib/          wrapper helpers (Bash, awk)
tools/        sampler.c (telemetry), plan_counter_passes.sh
docs/         contract, benchmark notes, counter selection
tests/        wrapper tests with mock benchmarks (no GPU needed)
```

## Tests

```bash
tests/run_tests.sh        # tests of wrapper and build script with mock benchmarks, no GPU needed
tests/run_tests.sh -v     # also show the wrapper output of failed tests
```
