# Microbenchmark-Suite (provisorischer Stand)

Benchmarks für die MI300A (gfx942) als Trainingsbasis für Powersched.

## Inhalt

| Label-Gruppe | Benchmark | Belastet |
|---|---|---|
| `fma_*`   | VALU Multiply-Add (fp64/fp32/fp16) | Compute, compute-bound |
| `stream_*` | Copy-Stream | Bandbreite L1 / L2 / Infinity Cache / HBM (nur HBM macht wirklich sinn, die anderen sind Overhead dominiert) |
| `chase_*` | Pointer Chasing, ein Thread | Speicherlatenz L1 / L2 / Infinity Cache / HBM |
| `gemm_*`       | geblockte Matrixmultiplikation (LDS-Kacheln)  | Compute und Speicher, AI über K steuerbar                    |
| `gemm_naive_*` | naive Matrixmultiplikation (`--kernel naive`) | viel Cache-Verkehr, kaum Rechenauslastung (vorläufig) |



## Verzeichnisstruktur

```
suite/
  run_suite.sh     Wrapper: baut und startet die Benchmarks
  runs.conf        welche Benchmarks mit welchen Parametern laufen
  src/             fma.cpp, stream.cpp, chase.cpp, matmul.cpp
  build/           wird automatisch angelegt
  results/         wird automatisch angelegt
```

## Schnellstart

```bash
./run_suite.sh                       # alle Einträge, je 30 s, 10 s Pause dazwischen
./run_suite.sh --duration 600        # je 10 min
./run_suite.sh --only fma_fp64,stream_hbm
./run_suite.sh --list                # verfügbare Labels
./run_suite.sh --help
```

## Ausgabe

Pro Lauf entsteht `results/<datum>_<host>/` mit:

- `runs.jsonl`: eine JSON-Zeile pro Benchmark, mit `label`, `args`, `start_unix`, `end_unix`, `exit_code` und dem Benchmark-Ergebnis unter `result`. 
- `summary.tsv`: kurze Übersicht (GFlop/s, GB/s).
- `logs/<label>.out|.err`: rohe Ausgabe der Benchmarks.
- `system_info.txt`: Host, ROCm-Version, GPU-Infos, Kopie der verwendeten `runs.conf`.

## Bekannte Einschränkungen

- Keine Hardware-Counter. Geplant ist rocprofv3.
- Working-Set-Größen in `runs.conf` sind fest auf MI300A eingetragen, noch nicht aus einer System-Config berechnet.
- GEMM gibt im JSON `traffic_model` aus: `kernel_level` (geblockt) oder `minimal` (naiv).
- GEMM: `--m` und `--n` müssen Vielfache von 128 sein.
- Integer-Varianten von `fma` sind implementiert, aber noch nicht validiert (in `runs.conf` auskommentiert).
- Nur eine GPU pro Lauf (`--gpu` bzw. `HIP_VISIBLE_DEVICES`).
