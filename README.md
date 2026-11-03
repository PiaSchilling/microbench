# Microbenchmark Suite

Benchmarks for AMD GPUs (MI300A, gfx942) and a wrapper that runs them, checks
their output and collects the results. No external dependencies: HIP/C++ for the
benchmarks, Bash and awk for everything else.

Specification: [docs/benchmark_contract.md](docs/benchmark_contract.md)

## Quick start

```bash
./build.sh                                   # build all benchmarks in src/ and the sampler into build/
./build/sampler --probe 5                    # how often does the GPU refresh its telemetry?
./run_suite.sh --list                        # available benchmarks
./run_suite.sh --help fma                    # parameters of one benchmark
./run_suite.sh --mode check                  # do all runs work and follow the contract?
./run_suite.sh --mode production --dry-run   # show the plan and the estimated time
./run_suite.sh --mode production --repeat 3  # measure
```

Runs are defined in `config/runs.cfg`, the hardware in `config/component.cfg`.
Results go to `results/<date>-<time>_<mode>/`; for analysis use `results.csv`
(one row per run) and `aggregates.csv` (statistics per label, with `--repeat > 1`).

## Tests

```bash
tests/run_tests.sh        # tests of wrapper and build script with mock benchmarks, no GPU needed
tests/run_tests.sh -v     # also show the wrapper output of failed tests
```

## Status

| Part | State |
|---|---|
| Wrapper: configs, metadata, validation, plan, execution, contract check, results, repetitions, aggregation, timeout, signals, help | implemented |
| Build script | implemented |
| Telemetry: sampler (`tools/sampler.c`) and its use in the wrapper (ROI cut, energy, `m_*` fields, idle baseline) | implemented, tested with a mock sampler; not yet tested on the GPU |
| Hardware counters (rocprofv3 passes, pass planner) | open |
| Slurm integration | open |
