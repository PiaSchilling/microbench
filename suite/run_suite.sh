#!/usr/bin/env bash
# run_suite.sh -- provisorischer Wrapper fuer die Microbenchmark-Suite
#
# Baut die Benchmarks mit hipcc, startet die Eintraege aus runs.conf nacheinander
# und schreibt pro Lauf eine JSON-Zeile mit Start-/Endzeit (Unix-Zeit) und Ergebnis.
# Die Zeitstempel erlauben es, die Laeufe spaeter den Powersched-/Telemetrie-Logs zuzuordnen.
#
# Aufruf:  ./run_suite.sh [Optionen]
#   --duration S     Laufzeit pro Benchmark in Sekunden, ersetzt {DURATION} (Standard 30)
#   --gap S          Pause zwischen zwei Laeufen in Sekunden (Standard 10)
#   --only A,B       nur diese Labels aus der Konfiguration ausfuehren
#   --config DATEI   Run-Konfiguration (Standard: runs.conf neben dem Skript)
#   --out DIR        Basisverzeichnis fuer Ergebnisse (Standard: results/ neben dem Skript)
#   --arch GFX       Zielarchitektur (Standard: per rocminfo erkannt, sonst gfx942)
#   --gpu ID         GPU ueber HIP_VISIBLE_DEVICES waehlen
#   --list           Labels der (gefilterten) Konfiguration ausgeben und beenden
#   --build-only     nur bauen, nichts ausfuehren
#   --no-build       nicht bauen, vorhandene Binaries aus build/ verwenden
#   -h, --help       diese Hilfe

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
CONFIG="$SCRIPT_DIR/runs.conf"
OUT_BASE="$SCRIPT_DIR/results"
DURATION=30
GAP=10
ONLY=""
ARCH=""
GPU=""
MODE=run        # run | list | build
NO_BUILD=0

# Benchmark -> Quelldatei und zusaetzliche Compiler-Flags
declare -A SRC=(   [fma]=fma.cpp  [stream]=stream.cpp  [chase]=chase.cpp  [gemm]=gemm.cpp )
declare -A EXTRA=( [fma]="-ffp-contract=fast" )
COMMON_FLAGS="-O3 -std=c++17"

log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()   { log "FEHLER: $*"; exit 2; }
usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; }
need()  { [[ $# -ge 2 && -n $2 ]] || die "Option $1 braucht einen Wert"; }
is_num(){ [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]; }
now()   { date +%s.%N; }

# ------------------------------------------------------------------ Argumente
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)   need "$@"; DURATION=$2; shift 2 ;;
        --gap)        need "$@"; GAP=$2;      shift 2 ;;
        --only)       need "$@"; ONLY=$2;     shift 2 ;;
        --config)     need "$@"; CONFIG=$2;   shift 2 ;;
        --out)        need "$@"; OUT_BASE=$2; shift 2 ;;
        --arch)       need "$@"; ARCH=$2;     shift 2 ;;
        --gpu)        need "$@"; GPU=$2;      shift 2 ;;
        --list)       MODE=list;  shift ;;
        --build-only) MODE=build; shift ;;
        --no-build)   NO_BUILD=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "Unbekanntes Argument: $1 (siehe --help)" ;;
    esac
done
is_num "$DURATION" && awk "BEGIN{exit !($DURATION > 0)}" || die "--duration muss eine Zahl > 0 sein"
is_num "$GAP" || die "--gap muss eine Zahl >= 0 sein"

# ------------------------------------------------------------------ Konfiguration lesen
# Format pro Zeile: <label> <benchmark> <argumente ...>   (# leitet Kommentar ein)
[[ -f $CONFIG ]] || die "Konfiguration nicht gefunden: $CONFIG"
LABELS=(); BENCH=(); ARGS=()
declare -A SEEN=()
lineno=0
while IFS= read -r line || [[ -n $line ]]; do
    lineno=$((lineno + 1))
    line="${line%%#*}"
    label=""; bench=""; rest=""
    read -r label bench rest <<< "$line"
    [[ -z $label ]] && continue
    [[ -n $bench ]]               || die "$CONFIG:$lineno: Benchmark fehlt fuer Label '$label'"
    [[ -n ${SRC[$bench]+x} ]]     || die "$CONFIG:$lineno: unbekannter Benchmark '$bench' (bekannt: ${!SRC[*]})"
    [[ -z ${SEEN[$label]+x} ]]    || die "$CONFIG:$lineno: Label '$label' doppelt vergeben"
    SEEN[$label]=1
    LABELS+=("$label"); BENCH+=("$bench"); ARGS+=("$rest")
done < "$CONFIG"
[[ ${#LABELS[@]} -gt 0 ]] || die "Keine Eintraege in $CONFIG"

# Auswahl (--only)
SEL=()
if [[ -n $ONLY ]]; then
    declare -A WANT=()
    IFS=',' read -ra wanted <<< "$ONLY"
    for w in "${wanted[@]}"; do [[ -n $w ]] && WANT[$w]=1; done
    for i in "${!LABELS[@]}"; do
        if [[ -n ${WANT[${LABELS[$i]}]+x} ]]; then SEL+=("$i"); unset "WANT[${LABELS[$i]}]"; fi
    done
    [[ ${#WANT[@]} -eq 0 ]] || die "Unbekannte Labels in --only: ${!WANT[*]}"
else
    SEL=("${!LABELS[@]}")
fi

if [[ $MODE == list ]]; then
    for i in "${SEL[@]}"; do echo "${LABELS[$i]}"; done
    exit 0
fi

# ------------------------------------------------------------------ Bauen
build() {
    command -v hipcc >/dev/null || die "hipcc nicht gefunden (ROCm-Umgebung geladen?)"
    if [[ -z $ARCH ]]; then
        ARCH="$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]\+')"
        if [[ -z $ARCH ]]; then ARCH=gfx942; log "Architektur nicht erkannt, verwende $ARCH (mit --arch aenderbar)"; fi
    fi
    mkdir -p "$BUILD_DIR"
    local -A needed=()
    local i b src bin
    for i in "${SEL[@]}"; do needed[${BENCH[$i]}]=1; done
    for b in "${!needed[@]}"; do
        src="$SRC_DIR/${SRC[$b]}"; bin="$BUILD_DIR/$b"
        if [[ ! -f $src ]]; then
            log "WARNUNG: $src fehlt -- Laeufe mit '$b' werden uebersprungen"
            continue
        fi
        if [[ -x $bin && $bin -nt $src && -f $bin.arch && "$(<"$bin.arch")" == "$ARCH" ]]; then
            continue   # aktuell
        fi
        log "Baue $b fuer $ARCH"
        # shellcheck disable=SC2086
        hipcc $COMMON_FLAGS ${EXTRA[$b]-} --offload-arch="$ARCH" "$src" -o "$bin" \
            || die "Build von $b fehlgeschlagen"
        echo "$ARCH" > "$bin.arch"
    done
}

[[ $NO_BUILD -eq 1 ]] || build
[[ $MODE == build ]] && { log "Build abgeschlossen"; exit 0; }

# ------------------------------------------------------------------ Ausfuehren
[[ -n $GPU ]] && export HIP_VISIBLE_DEVICES="$GPU"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$(hostname -s)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}"
RUN_DIR="$OUT_BASE/$RUN_ID"
mkdir -p "$RUN_DIR/logs" || die "Kann $RUN_DIR nicht anlegen"
JSONL="$RUN_DIR/runs.jsonl"
SUMMARY="$RUN_DIR/summary.tsv"

json_str() { local s=$1; s=${s//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; }
json_get() { grep -o "\"$2\": *[^,}]*" <<< "$1" | head -n1 | sed 's/^[^:]*: *//; s/"//g'; }

{
    echo "run_id:              $RUN_ID"
    echo "date:                $(date -Iseconds)"
    echo "host:                $(hostname)"
    echo "arch:                ${ARCH:-<nicht gebaut in diesem Lauf>}"
    echo "duration_s:          $DURATION"
    echo "gap_s:               $GAP"
    echo "config:              $CONFIG"
    echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES-<unset>}"
    echo "slurm_job_id:        ${SLURM_JOB_ID-<none>}"
    echo; echo "== hipcc --version =="; hipcc --version 2>&1 || true
    if command -v amd-smi >/dev/null; then echo; echo "== amd-smi static =="; amd-smi static 2>&1 || true
    elif command -v rocm-smi >/dev/null; then echo; echo "== rocm-smi =="; rocm-smi --showproductname --showclocks 2>&1 || true
    fi
} > "$RUN_DIR/system_info.txt"
cp "$CONFIG" "$RUN_DIR/runs.conf"

printf 'label\tbenchmark\texit_code\tstart_unix\tend_unix\twall_s\tgflop_per_s\tgbyte_per_s\tns_per_access\n' > "$SUMMARY"
trap 'log "Abgebrochen -- Teilergebnisse in $RUN_DIR"; exit 130' INT TERM

host_s="$(hostname -s)"
n=${#SEL[@]}; k=0; failed=0
log "Starte $n Laeufe, je ${DURATION}s, Ergebnisse in $RUN_DIR"

for i in "${SEL[@]}"; do
    k=$((k + 1))
    label=${LABELS[$i]}; bench=${BENCH[$i]}
    args=${ARGS[$i]//'{DURATION}'/$DURATION}
    bin="$BUILD_DIR/$bench"
    out="$RUN_DIR/logs/$label.out"; err="$RUN_DIR/logs/$label.err"
    read -ra argv <<< "$args"

    if [[ ! -x $bin ]]; then
        log "[$k/$n] $label: Binary $bin fehlt -- uebersprungen"
        : > "$out"; echo "Binary fehlt: $bin" > "$err"
        rc=127; start=$(now); end=$start
    else
        log "[$k/$n] $label: $bench $args"
        start=$(now)
        "$bin" "${argv[@]}" > "$out" 2> "$err"
        rc=$?
        end=$(now)
    fi

    # Jeder Benchmark schreibt als letzte Zeile auf stdout ein JSON-Objekt
    result=$(grep '^{' "$out" | tail -n1)
    gf=$(json_get "$result" d_gflop_per_s); gb=$(json_get "$result" d_gbyte_per_s)
    ns=$(json_get "$result" d_ns_per_access)      # nur chase
    [[ -n $result ]] || result=null
    wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b - a}')

    printf '{"run_id": %s, "host": %s, "label": %s, "benchmark": %s, "args": %s, "hip_visible_devices": %s, "start_unix": %s, "end_unix": %s, "wall_s": %s, "exit_code": %d, "result": %s}\n' \
        "$(json_str "$RUN_ID")" "$(json_str "$host_s")" "$(json_str "$label")" "$(json_str "$bench")" \
        "$(json_str "$args")" "$(json_str "${HIP_VISIBLE_DEVICES-}")" \
        "$start" "$end" "$wall" "$rc" "$result" >> "$JSONL"
    printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$bench" "$rc" "$start" "$end" "$wall" "${gf:--}" "${gb:--}" "${ns:--}" >> "$SUMMARY"

    if [[ $rc -ne 0 ]]; then
        failed=$((failed + 1))
        log "  -> FEHLER (exit $rc), Details: $err"
        tail -n 3 "$err" | sed 's/^/     /' >&2
    elif [[ -n $ns ]]; then
        log "  -> ${wall}s  ${ns} ns/Zugriff"
    else
        log "  -> ${wall}s  ${gf:--} GFlop/s  ${gb:--} GB/s"
    fi

    if (( k < n )) && awk "BEGIN{exit !($GAP > 0)}"; then sleep "$GAP"; fi
done

echo >&2
if command -v column >/dev/null; then column -t -s $'\t' "$SUMMARY" >&2; else cat "$SUMMARY" >&2; fi
log "Fertig: $((n - failed))/$n erfolgreich. Ergebnisse: $RUN_DIR"
[[ $failed -eq 0 ]]
