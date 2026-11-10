#!/usr/bin/env bash
# run_suite.sh - wrapper of the benchmark suite
#
# Specification: docs/benchmark_contract.md (sections 9-14)
# Implemented:   modes, run/component config, metadata, validation, plan,
#                execution, contract check, results (JSONL/CSV/TSV), repetitions,
#                aggregation, timeout, signal handling, help, telemetry (sampler),
#                hardware counters (rocprofv3 passes, validation mode)

set -uo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LIB=$SUITE_DIR/lib
# shellcheck source=lib/common.sh
source "$LIB/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

declare -A MODE_DURATION=([check]=0.001 [validation]=10 [production]=120)
declare -A MODE_PAUSE=([check]=0 [validation]=30 [production]=30)
TIMEOUT_FACTOR=2        # per-run timeout = duration * factor + offset
TIMEOUT_OFFSET=60
KILL_AFTER=10           # SIGKILL this many seconds after SIGTERM on timeout
DEFAULT_SEED=1
SAMPLER_START_TIMEOUT=5 # seconds to wait for the first telemetry sample
WRAPPER_FIELDS="label rep benchmark benchmark_version mode device status reason exit_code cmd"

MODE=""
CONFIG=$SUITE_DIR/config/runs.cfg
COMPONENT=$SUITE_DIR/config/component.cfg
SRC_DIR=$SUITE_DIR/src
BIN_DIR=$SUITE_DIR/build
RESULTS_ROOT=$SUITE_DIR/results
OPT_DURATION=""
OPT_PAUSE=""
OPT_TIMEOUT=""
REPEAT=1
SHUFFLE=0
SEED=$DEFAULT_SEED  # Seed for shuffle 
ONLY=""
DEVICE=0
DRY_RUN=0
DO_HELP=0
HELP_BENCH=""
DO_LIST=0
SAMPLE_INTERVAL=100     # ms between two telemetry samples (--sample-interval)
IDLE_SECONDS=5          # idle baseline before the first run (--idle)
COUNTERS_FILE=""        # counter passes (--counters; default config/counters_<arch>.cfg)
ROCPROF=rocprofv3       # profiler command (--rocprof)
PROFILE_DURATION=2      # --duration of the benchmark in counter passes (--profile-duration)
PASS_SETS=()            # PASS_SETS[k-1] = counters of pass k, comma separated
WMEAN=""                # counters aggregated as duration-weighted mean, comma separated
NCOUNTERS=0

# run table (index = position in the run config)
RUN_LABEL=() RUN_BENCH=() RUN_ARGS=() RUN_LINE=()
RUN_EFF=()                # effective arguments after {DURATION} substitution
RUN_DUR=()                # effective duration
RUN_SKIP_PREC=()          # reason text if the precision filter skips the run ("" otherwise)
RUN_FAILED_VALIDATION=()  # 1 if the run failed validation in phase 3 (then the wrapper aborts)
SEL=()                    # selected run indices (after --only)
declare -A LABEL_IDX=()
PLAN=()         # "<round>:<index>"

RESDIR=""
CHILD_PID=""
SAMPLER_PID=""          # telemetry sampler running in the background
SAMPLER_DIED=0          # 1 if the sampler stopped during the runs
TELEMETRY=""            # telemetry.csv of the current invocation
TEL_HEADER=""           # its header line
TEL_GPUS=""             # PCI addresses the sampler sees
TEL_LINES=0             # lines in telemetry.csv when the current run started
declare -A COUNT=([ok]=0 [skipped]=0 [error]=0 [invalid]=0 [timeout]=0)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: run_suite.sh --mode check|validation|production [options]
       run_suite.sh --list
       run_suite.sh --help [<benchmark>]

Modes:
  check        every run with a minimal duration: do all benchmarks work and
               comply with the contract? (no monitoring)
  validation   medium duration, telemetry + hardware counters
  production   long duration, telemetry only
  In validation mode every run is repeated once per counter pass (pass 1..K)
  under rocprofv3; the counters of the ROI are written to counters.csv.

Options:
  --config <file>       run config                (default: config/runs.cfg)
  --component <file>    component config          (default: config/component.cfg)
  --duration <s>        value for {DURATION}      (default: check ${MODE_DURATION[check]}, validation ${MODE_DURATION[validation]}, production ${MODE_DURATION[production]})
                        in check mode it applies to all runs, also fixed values
  --pause <s>           pause between runs        (default: check ${MODE_PAUSE[check]}, validation ${MODE_PAUSE[validation]}, production ${MODE_PAUSE[production]})
  --repeat <n>          repetitions, run in rounds (default: 1)
  --shuffle [<seed>]    shuffle the order within each round (default seed: $DEFAULT_SEED)
  --only <l1,l2,...>    run only these labels
  --device <n>          GPU index, set as HIP_VISIBLE_DEVICES (default: 0)
  --timeout <s>         per-run timeout           (default: duration x $TIMEOUT_FACTOR + $TIMEOUT_OFFSET)
  --sample-interval <ms> telemetry interval       (default: $SAMPLE_INTERVAL; validation/production)
  --idle <s>            idle baseline before the first run (default: $IDLE_SECONDS; 0: none)
  --counters <file>     counter passes            (default: config/counters_<arch>.cfg; validation)
                        create it with tools/plan_counter_passes.sh
  --rocprof <cmd>       profiler                  (default: $ROCPROF)
  --profile-duration <s> --duration of the benchmark in each counter pass (default: $PROFILE_DURATION)
  --dry-run             validate and print the plan, run nothing
  --results <dir>       where result directories are created (default: results/)
  --src-dir <dir>       benchmark sources         (default: src/)
  --bin-dir <dir>       benchmark binaries        (default: build/)
  --list                list all benchmarks
  --help [<benchmark>]  this help, or details of one benchmark

Output: result table on stdout, messages on stderr, everything else in
        results/<date>-<time>_<mode>/ (see docs/benchmark_contract.md, section 10)

Exit codes: 0 all runs ok or skipped    1 invalid options or validation failed
            4 a run error/invalid/timeout  5 aborted by signal
EOF
}

# prints a list of all available benchmarks found in the src directory
list_benchmarks() {
    local f name files=()
    shopt -s nullglob
    files=("$SRC_DIR"/*.cpp)
    shopt -u nullglob
    if ((${#files[@]} == 0)); then
        info "no benchmarks found in $SRC_DIR"
        return 0
    fi
    printf '%-16s %-5s %-24s %s\n' benchmark level component description
    for f in "${files[@]}"; do
        name=$(basename -- "$f" .cpp)
        if meta_load "$name" "$SRC_DIR"; then
            printf '%-16s %-5s %-24s %s\n' "$name" "$(meta_get "$name" level)" \
                "$(meta_get "$name" component)" "$(meta_get "$name" description)"
        else
            printf '%-16s %s\n' "$name" "(invalid metadata, see: run_suite.sh --help $name)"
        fi
    done
}

# prints help output for th specified benchmark by parsing the benchmark metadata from the src/*.cpp header
help_benchmark() {
    local name=$1 k def valid effect bin
    if ! meta_load "$name" "$SRC_DIR"; then
        report_errors "metadata of '$name' is invalid"
        exit 1
    fi
    local -n m="META_$name"
    local -n ord="METAORD_$name"
    bin="$BIN_DIR/$name"
    printf '%s %s - %s\n\n' "$name" "${m[version]}" "${m[description]}"
    printf '  level       %s\n' "${m[level]}"
    printf '  component   %s\n' "${m[component]}"
    printf '  tags        %s\n' "${m[tags]//;/, }"
    [[ -n ${m[precisions]:-} ]] && printf '  precisions  %s\n' "${m[precisions]//,/, }"
    printf '  source      %s\n' "$SRC_DIR/$name.cpp"
    printf '  binary      %s%s\n' "$bin" "$([[ -x $bin ]] || echo '  (not built)')"
    printf '\nParameters:\n'
    for k in "${ord[@]}"; do
        [[ $k == param.* ]] || continue
        IFS='|' read -r def valid effect <<<"${m[$k]}"
        printf '  --%-20s default: %-10s valid: %-18s %s\n' "${k#param.}" "$def" "$valid" "$effect"
    done
    local first=1
    for k in "${ord[@]}"; do
        [[ $k == expect_* && -n ${m[$k]} ]] || continue
        ((first)) && printf '\nExpectations (validation mode):\n' && first=0
        printf '  %-20s %s\n' "$k" "${m[$k]}"
    done
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

abspath() {
    local p=$1
    while [[ $p == ./* ]]; do p=${p#./}; done
    if [[ $p == /* ]]; then printf '%s' "$p"; else printf '%s/%s' "$PWD" "$p"; fi
}

# optval <option> <next arg...>: make sure the option has a value
optval() { [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "option $1 requires a value (see --help)"; }

positive() { awk -v x="$1" 'BEGIN { exit !(x + 0 > 0) }'; }

json_str() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//[[:cntrl:]]/ }
    printf '"%s"' "$s"
}

csv_str() { local s=${1//\"/\"\"}; printf '"%s"' "$s"; }

fmt_secs() {
    awk -v s="$1" 'BEGIN {
        s = int(s + 0.5); h = int(s / 3600); m = int((s % 3600) / 60); r = s % 60
        if (h) printf "%dh %02dm %02ds", h, m, r; else if (m) printf "%dm %02ds", m, r; else printf "%ds", r }'
}

# last non-empty line of a file, shortened
last_line() { awk 'NF { l = $0 } END { print substr(l, 1, 200) }' "$1" 2>/dev/null; }

# ---------------------------------------------------------------------------
# Phase 1: arguments
# ---------------------------------------------------------------------------

# parses command line arguments into the corresponding variables. 
parse_args() {
    while (($#)); do
        case $1 in
            --mode)      optval "$@"; MODE=$2; shift 2 ;;
            --config)    optval "$@"; CONFIG=$2; shift 2 ;;
            --component) optval "$@"; COMPONENT=$2; shift 2 ;;
            --duration)  optval "$@"; OPT_DURATION=$2; shift 2 ;;
            --pause)     optval "$@"; OPT_PAUSE=$2; shift 2 ;;
            --repeat)    optval "$@"; REPEAT=$2; shift 2 ;;
            --only)      optval "$@"; ONLY=$2; shift 2 ;;
            --device)    optval "$@"; DEVICE=$2; shift 2 ;;
            --timeout)   optval "$@"; OPT_TIMEOUT=$2; shift 2 ;;
            --sample-interval) optval "$@"; SAMPLE_INTERVAL=$2; shift 2 ;;
            --idle)      optval "$@"; IDLE_SECONDS=$2; shift 2 ;;
            --counters)  optval "$@"; COUNTERS_FILE=$2; shift 2 ;;
            --rocprof)   optval "$@"; ROCPROF=$2; shift 2 ;;
            --profile-duration) optval "$@"; PROFILE_DURATION=$2; shift 2 ;;
            --results)   optval "$@"; RESULTS_ROOT=$2; shift 2 ;;
            --src-dir)   optval "$@"; SRC_DIR=$2; shift 2 ;;
            --bin-dir)   optval "$@"; BIN_DIR=$2; shift 2 ;;
            --shuffle)
                SHUFFLE=1
                if [[ ${2:-} =~ ^[0-9]+$ ]]; then SEED=$2; shift; fi
                shift ;;
            --dry-run)   DRY_RUN=1; shift ;;
            --list)      DO_LIST=1; shift ;;
            --help | -h)
                DO_HELP=1
                if [[ -n ${2:-} && ${2:-} != -* ]]; then HELP_BENCH=$2; shift; fi
                shift ;;
            *) die "unknown option '$1' (see --help)" ;;
        esac
    done

    SRC_DIR=$(abspath "$SRC_DIR")
    BIN_DIR=$(abspath "$BIN_DIR")
    RESULTS_ROOT=$(abspath "$RESULTS_ROOT")

    if ((DO_HELP)); then
        if [[ -n $HELP_BENCH ]]; then help_benchmark "$HELP_BENCH"; else usage; fi
        exit 0
    fi
    if ((DO_LIST)); then list_benchmarks; exit 0; fi

    [[ -n $MODE ]] || die "--mode is required (see --help)"
    [[ -v MODE_DURATION[$MODE] ]] || die "unknown mode '$MODE' (check, validation, production)"
    [[ -z $OPT_DURATION ]] || { is_number "$OPT_DURATION" && positive "$OPT_DURATION"; } ||
        die "--duration must be a positive number"
    [[ -z $OPT_PAUSE ]] || is_number "$OPT_PAUSE" || die "--pause must be a number >= 0"
    [[ -z $OPT_TIMEOUT ]] || { is_number "$OPT_TIMEOUT" && positive "$OPT_TIMEOUT"; } ||
        die "--timeout must be a positive number"
    [[ $REPEAT =~ ^[1-9][0-9]*$ ]] || die "--repeat must be an integer >= 1"
    is_uint "$DEVICE" || die "--device must be an integer >= 0"
    [[ $SAMPLE_INTERVAL =~ ^[1-9][0-9]*$ ]] && ((SAMPLE_INTERVAL <= 60000)) ||
        die "--sample-interval must be an integer between 1 and 60000 (ms)"
    is_number "$IDLE_SECONDS" || die "--idle must be a number >= 0"
    { is_number "$PROFILE_DURATION" && positive "$PROFILE_DURATION"; } ||
        die "--profile-duration must be a positive number"

    DURATION=${OPT_DURATION:-${MODE_DURATION[$MODE]}}
    PAUSE=${OPT_PAUSE:-${MODE_PAUSE[$MODE]}}
}

# ---------------------------------------------------------------------------
# Phase 2: load
# ---------------------------------------------------------------------------

# parses the run configuration file (and checks for incorrectly formatted run configurations)
parse_runs() {
    local file=$1 line n=0 label bench rest
    if [[ ! -r $file ]]; then
        add_error "run config: cannot read '$file'"
        return
    fi
    while IFS= read -r line || [[ -n $line ]]; do
        n=$((n + 1))
        line=${line%$'\r'}
        line=${line%%#*}             # comment: whole line or rest of the line after '#'
        trim line
        [[ -z $line ]] && continue
        label="" bench="" rest=""
        read -r label bench rest <<<"$line"
        if [[ -z $bench ]]; then
            add_error "$file:$n: expected '<label> <benchmark> <arguments ...>'"
            continue
        fi
        if [[ ! $label =~ ^[A-Za-z0-9_-]+$ ]]; then
            add_error "$file:$n: label '$label' has invalid characters (allowed: A-Z a-z 0-9 _ -)"
            continue
        fi
        if [[ -v LABEL_IDX[$label] ]]; then
            add_error "$file:$n: label '$label' is already used in line ${RUN_LINE[${LABEL_IDX[$label]}]}"
            continue
        fi
        LABEL_IDX[$label]=${#RUN_LABEL[@]}
        RUN_LABEL+=("$label")
        RUN_BENCH+=("$bench")
        RUN_ARGS+=("$rest")
        RUN_LINE+=("$n")
    done <"$file"
    ((${#RUN_LABEL[@]})) || add_error "$file: no runs defined"
}

# selects the benchmarks specified via the --only parameter 
select_runs() {
    local l i
    local -a want
    if [[ -z $ONLY ]]; then
        for i in "${!RUN_LABEL[@]}"; do SEL+=("$i"); done
        return
    fi
    IFS=, read -r -a want <<<"$ONLY"
    for l in "${want[@]}"; do
        if [[ -v LABEL_IDX[$l] ]]; then SEL+=("${LABEL_IDX[$l]}")
        else add_error "--only: label '$l' not found in $CONFIG"; fi
    done
}

# ---------------------------------------------------------------------------
# Phase 3: validate (no GPU work)
# ---------------------------------------------------------------------------

validate_runs() {
    local i j k v label bench ctx prec bprec bin
    local -a toks eff
    local -A seen
    for i in "${SEL[@]}"; do
        label=${RUN_LABEL[i]}
        bench=${RUN_BENCH[i]}
        ctx="$CONFIG:${RUN_LINE[i]} ($label)"
        RUN_FAILED_VALIDATION[i]=0
        RUN_DUR[i]=""
        RUN_SKIP_PREC[i]=""

        if ! meta_load "$bench" "$SRC_DIR"; then RUN_FAILED_VALIDATION[i]=1; continue; fi
        bin="$BIN_DIR/$bench"
        if [[ ! -x $bin ]]; then
            add_error "$ctx: binary '$bin' not found or not executable (run build.sh)"
            RUN_FAILED_VALIDATION[i]=1
        fi

        if [[ ${RUN_ARGS[i]} == *[\"\']* ]]; then
            add_error "$ctx: quotes are not allowed in arguments"
            RUN_FAILED_VALIDATION[i]=1; continue
        fi
        read -r -a toks <<<"${RUN_ARGS[i]}"
        if ((${#toks[@]} % 2)); then
            add_error "$ctx: arguments must be pairs '--key value'"
            RUN_FAILED_VALIDATION[i]=1; continue
        fi

        seen=() eff=() prec=""
        for ((j = 0; j < ${#toks[@]}; j += 2)); do
            k=${toks[j]}
            v=${toks[j + 1]}
            if [[ ! $k =~ ^--[a-z_][a-z0-9_]*$ ]]; then
                add_error "$ctx: '$k' is not an option of the form --key"
                RUN_FAILED_VALIDATION[i]=1; continue
            fi
            if [[ $v == --* ]]; then
                add_error "$ctx: option $k has no value"
                RUN_FAILED_VALIDATION[i]=1; continue
            fi
            if [[ -v seen[$k] ]]; then
                add_error "$ctx: option $k is given twice"
                RUN_FAILED_VALIDATION[i]=1; continue
            fi
            seen[$k]=1
            if [[ $k == --check ]]; then
                add_error "$ctx: --check must not be used in the run config"
                RUN_FAILED_VALIDATION[i]=1; continue
            fi
            if [[ $k == --duration ]]; then
                [[ $MODE == check || $v == "{DURATION}" ]] && v=$DURATION
                if ! { is_number "$v" && positive "$v"; }; then
                    add_error "$ctx: --duration must be a positive number or {DURATION} (got '$v')"
                    RUN_FAILED_VALIDATION[i]=1
                fi
                RUN_DUR[i]=$v
            elif [[ $v == *"{DURATION}"* ]]; then
                add_error "$ctx: {DURATION} is only allowed as value of --duration"
                RUN_FAILED_VALIDATION[i]=1
            fi
            [[ $k == --precision ]] && prec=$v
            if [[ ! -v "META_${bench}[param.${k#--}]" ]]; then
                warn "$ctx: parameter $k is not documented in the metadata of $bench"
            fi
            eff+=("$k" "$v")
        done
        if [[ ! -v seen[--duration] ]]; then
            add_error "$ctx: --duration is missing (mandatory)"
            RUN_FAILED_VALIDATION[i]=1
        fi

        # precision filter (contract section 3)
        if [[ -n $prec ]]; then
            bprec=$(meta_get "$bench" precisions)
            if [[ -z $bprec ]]; then
                add_error "$ctx: --precision is given, but $bench declares no 'precisions' in its metadata"
                RUN_FAILED_VALIDATION[i]=1
            elif ! in_list "$prec" "$bprec"; then
                RUN_SKIP_PREC[i]="precision $prec is not implemented by $bench (precisions=$bprec)"
            elif ! in_list "$prec" "${COMP[precisions]:-}"; then
                RUN_SKIP_PREC[i]="precision $prec is not supported by the hardware (component config)"
            fi
        fi
        RUN_EFF[i]="${eff[*]}"
    done
}

# run_check <bin> <args...>: call the benchmark with --check; sets CHECK_RC and CHECK_MSG
run_check() {
    local bin=$1 tmp out
    shift
    tmp=$(mktemp)
    out=$(HIP_VISIBLE_DEVICES=$DEVICE timeout 30 "$bin" --check "$@" 2>"$tmp")
    CHECK_RC=$?
    CHECK_MSG=$(last_line "$tmp")
    rm -f "$tmp"
    if ((CHECK_RC == 0)) && [[ -n $out ]]; then
        CHECK_RC=99
        CHECK_MSG="--check must not write to stdout"
    fi
}

check_args() {
    local i bench bin
    local -a args
    for i in "${SEL[@]}"; do
        ((RUN_FAILED_VALIDATION[i])) || [[ -n ${RUN_SKIP_PREC[i]} ]] && continue
        bench=${RUN_BENCH[i]}
        read -r -a args <<<"${RUN_EFF[i]}"
        run_check "$BIN_DIR/$bench" "${args[@]}"
        if ((CHECK_RC != 0)); then
            add_error "$CONFIG:${RUN_LINE[i]} (${RUN_LABEL[i]}): '$bench --check' rejected the arguments (exit $CHECK_RC): ${CHECK_MSG:-no message}"
            RUN_FAILED_VALIDATION[i]=1
        fi
    done
}

# check mode: every documented parameter default must be accepted by --check
check_param_defaults() {
    local bench k name def valid effect
    local -A done_bench=()
    local i
    for i in "${SEL[@]}"; do
        bench=${RUN_BENCH[i]}
        [[ -v done_bench[$bench] ]] && continue
        done_bench[$bench]=1
        [[ ${META_LOADED[$bench]:-0} == 1 && -x $BIN_DIR/$bench ]] || continue
        local -n ord="METAORD_$bench"
        local -n m="META_$bench"
        for k in "${ord[@]}"; do
            [[ $k == param.* ]] || continue
            name=${k#param.}
            IFS='|' read -r def valid effect <<<"${m[$k]}"
            trim def
            [[ $name == duration || $def == required ]] && continue
            run_check "$BIN_DIR/$bench" --duration 1 "--$name" "$def"
            ((CHECK_RC == 0)) ||
                add_error "$bench: documented default of --$name ($def) is rejected by --check (exit $CHECK_RC): ${CHECK_MSG:-no message}"
        done
        unset -n ord m
    done
}

# counter passes (validation mode, contract section 13)
profiling() { [[ $MODE == validation ]]; }

# load_counters: read the counter config into PASS_SETS and WMEAN
load_counters() {
    local f=${COUNTERS_FILE:-$SUITE_DIR/config/counters_${COMP[arch]:-unknown}.cfg} k c n
    local hint="create it on a GPU node with tools/plan_counter_passes.sh, or pass --counters"
    local -A cfg=() seen=()
    local -a ord=() list
    COUNTERS_FILE=$(abspath "$f")
    if [[ ! -r $COUNTERS_FILE ]]; then
        add_error "counters: cannot read '$COUNTERS_FILE' ($hint)"
        return
    fi
    kv_parse cfg ord "$COUNTERS_FILE" <"$COUNTERS_FILE"
    for k in "${ord[@]}"; do
        case $k in
            pass.*)
                n=${k#pass.}
                if [[ ! $n =~ ^[1-9][0-9]*$ ]]; then
                    add_error "counters: $COUNTERS_FILE: invalid key '$k' (pass.<1..K>)"; continue
                fi
                IFS=, read -r -a list <<<"${cfg[$k]}"
                PASS_SETS[n - 1]=""
                for c in "${list[@]}"; do
                    trim c
                    [[ -z $c ]] && continue
                    if [[ ! $c =~ ^[A-Za-z0-9_]+$ ]]; then
                        add_error "counters: $COUNTERS_FILE: invalid counter name '$c' in $k"; continue
                    fi
                    if [[ -v seen[$c] ]]; then
                        add_error "counters: $COUNTERS_FILE: counter $c appears in two passes"; continue
                    fi
                    seen[$c]=1
                    NCOUNTERS=$((NCOUNTERS + 1))
                    PASS_SETS[n - 1]+="${PASS_SETS[n - 1]:+,}$c"
                done
                [[ -n ${PASS_SETS[n - 1]} ]] || add_error "counters: $COUNTERS_FILE: $k has no counters" ;;
            agg.*)
                case ${cfg[$k]} in
                    sum) ;;
                    wmean) WMEAN+="${WMEAN:+,}${k#agg.}" ;;
                    *) add_error "counters: $COUNTERS_FILE: $k must be sum or wmean" ;;
                esac ;;
            *) add_error "counters: $COUNTERS_FILE: unknown key '$k' (pass.<k> or agg.<counter>)" ;;
        esac
    done
    if ((${#PASS_SETS[@]} == 0)); then
        add_error "counters: $COUNTERS_FILE defines no passes ($hint)"
        return
    fi
    for ((n = 1; n <= ${#PASS_SETS[@]}; n++)); do
        [[ -v "PASS_SETS[$((n - 1))]" ]] || add_error "counters: $COUNTERS_FILE: pass.$n is missing (passes must be numbered 1..K)"
    done
    command -v "$ROCPROF" >/dev/null 2>&1 ||
        add_error "counters: profiler '$ROCPROF' not found (load the ROCm module or pass --rocprof)"
}

# ---------------------------------------------------------------------------
# Phase 4: plan
# ---------------------------------------------------------------------------

build_plan() {
    local r i
    local -a order
    for ((r = 1; r <= REPEAT; r++)); do
        order=("${SEL[@]}")
        if ((SHUFFLE)); then
            mapfile -t order < <(printf '%s\n' "${SEL[@]}" |
                awk -v s=$((SEED + r)) '
                    # own generator (Park-Miller): srand/rand differ between awk
                    # implementations and are not reproducible in all of them
                    BEGIN { x = s % 2147483646 + 1 }
                    { x = (16807 * x) % 2147483647; printf "%010d\t%s\n", x, $0 }' |
                sort -k1,1 | cut -f2)
        fi
        for i in "${order[@]}"; do PLAN+=("$r:$i"); done
    done
}

# pinfo: like info, but also kept for wrapper.log (the log does not exist yet)
PLAN_TEXT=()
pinfo() { info "$*"; PLAN_TEXT+=("$*"); }

print_plan() {
    local i execs=0 total p
    local mode_note=""
    [[ $MODE == check ]] && mode_note=" (applied to all runs)"
    pinfo "Plan: mode $MODE, ${#SEL[@]} run(s) x $REPEAT round(s), device $DEVICE, duration $DURATION s$mode_note, pause $PAUSE s"
    ((SHUFFLE)) && pinfo "      order shuffled within each round, seed $SEED"
    for i in "${SEL[@]}"; do
        if [[ -n ${RUN_SKIP_PREC[i]} ]]; then
            pinfo "  $(printf "%-${LW}s  %-${BW}s  skipped: %s" "${RUN_LABEL[i]}" "${RUN_BENCH[i]}" "${RUN_SKIP_PREC[i]}")"
        else
            pinfo "  $(printf "%-${LW}s  %-${BW}s  %s" "${RUN_LABEL[i]}" "${RUN_BENCH[i]}" "${RUN_EFF[i]}")"
        fi
    done
    total=0
    for p in "${PLAN[@]}"; do
        i=${p#*:}
        [[ -n ${RUN_SKIP_PREC[i]} ]] && continue
        execs=$((execs + 1))
        total=$(awk -v t="$total" -v d="${RUN_DUR[i]}" 'BEGIN { print t + d }')
    done
    ((execs > 1)) && total=$(awk -v t="$total" -v p="$PAUSE" -v n="$execs" 'BEGIN { print t + p * (n - 1) }')
    if monitoring; then
        pinfo "Telemetry: sampler every $SAMPLE_INTERVAL ms, idle baseline $IDLE_SECONDS s"
        total=$(awk -v t="$total" -v i="$IDLE_SECONDS" 'BEGIN { print t + i }')
    fi
    if profiling; then
        pinfo "Counters: ${#PASS_SETS[@]} pass(es) per run with $NCOUNTERS counters, $PROFILE_DURATION s each ($ROCPROF, $COUNTERS_FILE)"
        total=$(awk -v t="$total" -v n="$execs" -v k="${#PASS_SETS[@]}" -v d="$PROFILE_DURATION" 'BEGIN { print t + n * k * d }')
    fi
    pinfo "Estimated time: at least $(fmt_secs "$total") for $execs execution(s) (runs + pauses, without start-up and verification)"
}

# ---------------------------------------------------------------------------
# Phase 5: prepare
# ---------------------------------------------------------------------------

prepare() {
    local stamp n=1 b i
    local -A copied=()
    stamp=$(date +%Y%m%d-%H%M%S)
    RESDIR="$RESULTS_ROOT/${stamp}_$MODE"
    while [[ -e $RESDIR ]]; do n=$((n + 1)); RESDIR="$RESULTS_ROOT/${stamp}_${MODE}_$n"; done
    mkdir -p "$RESDIR/config/src" "$RESDIR/runs" || die "cannot create result directory '$RESDIR'"
    LOG_FILE=$RESDIR/wrapper.log

    log "START  run_suite.sh $ORIG_ARGS"
    log "suite $SUITE_DIR, mode $MODE, duration $DURATION, pause $PAUSE, repeat $REPEAT, shuffle $SHUFFLE (seed $SEED), device $DEVICE"
    local l
    for l in "${PLAN_TEXT[@]}"; do log "PLAN   $l"; done
    cp -- "$CONFIG" "$RESDIR/config/runs.cfg"
    cp -- "$COMPONENT" "$RESDIR/config/component.cfg"
    if profiling; then
        cp -- "$COUNTERS_FILE" "$RESDIR/config/counters.cfg"
        echo "label,rep,benchmark,benchmark_version,device,pass,counter,value,agg,dispatches,iterations,kernel_time_s,roi_s" \
            >"$RESDIR/counters.csv"
    fi
    for i in "${SEL[@]}"; do
        b=${RUN_BENCH[i]}
        [[ -v copied[$b] ]] && continue
        copied[$b]=1
        cp -- "$SRC_DIR/$b.cpp" "$RESDIR/config/src/"
    done
    : >"$RESDIR/results.jsonl"
}

# ---------------------------------------------------------------------------
# Telemetry (sampler, contract section 13)
# ---------------------------------------------------------------------------
# The sampler writes telemetry.csv with one line per GPU and sample. Its first
# three columns are mono_ns, unix_ns and bdf; the waits below rely on that.

monitoring() { [[ $MODE != check ]]; }

sampler_alive() { [[ -n $SAMPLER_PID ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; }

# start the sampler and wait for its first sample (phase 5)
start_sampler() {
    local w tries=$((SAMPLER_START_TIMEOUT * 20)) rc
    TELEMETRY=$RESDIR/telemetry.csv
    "$BIN_DIR/sampler" --interval-ms "$SAMPLE_INTERVAL" --out "$TELEMETRY" 2>"$RESDIR/sampler.log" &
    SAMPLER_PID=$!
    log "SAMPLER started: pid $SAMPLER_PID, interval $SAMPLE_INTERVAL ms"
    for ((w = 0; w < tries; w++)); do
        if ! sampler_alive; then
            wait "$SAMPLER_PID"
            rc=$?
            SAMPLER_PID=""
            die "sampler exited at start (exit $rc): $(last_line "$RESDIR/sampler.log")"
        fi
        [[ -s $TELEMETRY ]] && (($(wc -l <"$TELEMETRY") >= 2)) && break
        sleep 0.05
    done
    if ((w == tries)); then
        stop_sampler
        die "no telemetry from the sampler after $SAMPLER_START_TIMEOUT s (see $RESDIR/sampler.log)"
    fi
    TEL_HEADER=$(head -1 "$TELEMETRY")
    if [[ $TEL_HEADER != mono_ns,unix_ns,bdf,* ]]; then
        stop_sampler
        die "unexpected header in $TELEMETRY: ${TEL_HEADER:0:80}"
    fi
    TEL_GPUS=$(awk -F, 'NR > 1 { if ($3 in s) exit; s[$3] = 1; printf "%s%s", (n++ ? " " : ""), $3 }' "$TELEMETRY")
    info "Telemetry: sampler running every $SAMPLE_INTERVAL ms, GPUs: $TEL_GPUS"
    log "SAMPLER GPUs: $TEL_GPUS"
}

stop_sampler() {
    [[ -n $SAMPLER_PID ]] || return 0
    kill -TERM "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
    SAMPLER_PID=""
    log "SAMPLER stopped"
}

# wait_for_sample_after <column 1|2> <t_ns> [<bdf>]: wait until telemetry.csv has a
# sample later than t (column 1: mono_ns, 2: unix_ns), needed to interpolate at t.
# Waits at most 3 intervals + 2 s; returns 1 if there is none.
wait_for_sample_after() {
    local c=$1 t=$2 b=${3:-} w tries=$(((3 * SAMPLE_INTERVAL + 2000) / 50))
    for ((w = 0; w < tries; w++)); do
        tail -n 300 "$TELEMETRY" | awk -F, -v c="$c" -v t="$t" -v b="$b" \
            '(b == "" || $3 == b) && $c + 0 > t + 0 { f = 1 } END { exit !f }' && return 0
        sampler_alive || return 1
        sleep 0.05
    done
    return 1
}

# idle baseline (phase 6): telemetry without load, per GPU in idle/idle.csv.
# The wrapper has no monotonic clock, so this window is cut by unix_ns.
idle_baseline() {
    local u0 u1 line
    awk -v s="$IDLE_SECONDS" 'BEGIN { exit !(s > 0) }' || return 0
    info "Idle baseline: $IDLE_SECONDS s without load"
    mkdir -p "$RESDIR/idle"
    u0=$(date +%s%N)
    sleep "$IDLE_SECONDS" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    CHILD_PID=""
    u1=$(date +%s%N)
    wait_for_sample_after 2 "$u1" || warn "idle baseline: no telemetry sample after the idle window"
    awk -F, -v timecol=unix_ns -v t0="$u0" -v t1="$u1" -f "$LIB/json.awk" -f "$LIB/telemetry.awk" \
        "$TELEMETRY" >"$RESDIR/idle/idle.jsonl"
    awk -f "$LIB/json.awk" -f "$LIB/jsonl2csv.awk" "$RESDIR/idle/idle.jsonl" >"$RESDIR/idle/idle.csv"
    while IFS= read -r line; do
        info "  $line"
        log "IDLE   $line"
    done < <(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
        { gsub(/"/, ""); printf "%s  power %.1f W  gfx clock %.0f MHz  temperature %s C\n",
          $c["bdf"], $c["m_avg_power_w"], $c["m_avg_gfx_clk_mhz"], $c["m_max_temp_c"] }' "$RESDIR/idle/idle.csv")
    rm -f "$RESDIR/idle/idle.jsonl"
}

# telemetry_for_run <dir> <roi "start end" in mono ns> <bdf>: cut telemetry to the ROI
# of a run. Sets MFRAG (m_* JSON members) or TEL_REASON (why there is no telemetry).
telemetry_for_run() {
    local dir=$1 roi=$2 bdf=$3 t0 t1 from line
    MFRAG="" TEL_REASON=""
    read -r t0 t1 <<<"$roi"
    if ! wait_for_sample_after 1 "$t1" "$bdf"; then
        if ! tail -n 300 "$TELEMETRY" | grep -qF ",$bdf,"; then
            TEL_REASON="no samples for device $bdf (sampler sees: $TEL_GPUS)"
        else
            TEL_REASON="no sample after the end of the ROI for device $bdf"
        fi
        return
    fi
    # only the part of the file written since shortly before the run
    from=$((TEL_LINES > 300 ? TEL_LINES - 300 : 1))
    line=$(tail -n +"$from" "$TELEMETRY" | awk -F, -v header="$TEL_HEADER" -v t0="$t0" -v t1="$t1" \
        -v bdf="$bdf" -v cut="$dir/telemetry_roi.csv" -f "$LIB/json.awk" -f "$LIB/telemetry.awk")
    if [[ -z $line ]]; then
        TEL_REASON="no samples for device $bdf around the ROI"
        return
    fi
    line=${line#\{}
    line=${line%\}}
    MFRAG=${line#*\"bdf\":\"*\",}     # drop the bdf member, keep the m_* fields
}

# ---------------------------------------------------------------------------
# Phase 7: execute
# ---------------------------------------------------------------------------

on_signal() {
    trap '' INT TERM
    warn "received SIG$1, aborting"
    if [[ -n $CHILD_PID ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null
        wait "$CHILD_PID" 2>/dev/null
    fi
    stop_sampler
    if [[ -n $RESDIR ]]; then
        log "ABORT  by SIG$1; completed runs are preserved"
        info "Results (incomplete): $RESDIR"
    fi
    exit 5
}

# timed_out <t0_ns> <t1_ns> <timeout_s>: did the run last (almost) as long as the timeout?
timed_out() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN { exit !((b - a) / 1e9 >= t * 0.99) }'; }

# run_passes <round> <index> <pass0 "runtime iterations">: counter passes 1..K of a run
# under rocprofv3 (contract section 13). Appends to counters.csv and sets PFRAG
# (m_prof_slowdown_p<k> members); on failure PASS_REASON is set and nothing is appended.
run_passes() {
    local r=$1 i=$2 perf0=$3
    local label=${RUN_LABEL[i]} bench=${RUN_BENCH[i]} bin="$BIN_DIR/$bench"
    local k n dir code tmo t0 t1 tag rest jstatus jreason roi perf cmd csv meta prefix sd
    local rows="" nd it
    local -a args pmc
    PFRAG="" PASS_REASON=""
    read -r -a args <<<"${RUN_EFF[i]}"
    for ((n = 0; n < ${#args[@]}; n += 2)); do
        [[ ${args[n]} == --duration ]] && args[n + 1]=$PROFILE_DURATION
    done
    tmo=${OPT_TIMEOUT:-$(awk -v d="$PROFILE_DURATION" -v f=$TIMEOUT_FACTOR -v o=$TIMEOUT_OFFSET \
        'BEGIN { printf "%.3f", d * f + o }')}
    prefix="$(csv_str "$label"),$r,$(csv_str "$bench"),$(csv_str "$(meta_get "$bench" version)"),$DEVICE"

    for ((k = 1; k <= ${#PASS_SETS[@]}; k++)); do
        dir="$RESDIR/runs/$label/rep$r/pass$k"
        mkdir -p "$dir"
        IFS=, read -r -a pmc <<<"${PASS_SETS[k - 1]}"
        printf -v cmd '%q ' "HIP_VISIBLE_DEVICES=$DEVICE" timeout "$tmo" "$ROCPROF" --pmc "${pmc[@]}" \
            --output-format csv -d "$dir/rocprof" -- "$bin" "${args[@]}"
        printf '%s\n' "${cmd% }" >"$dir/command.txt"
        t0=$(date +%s%N)
        HIP_VISIBLE_DEVICES=$DEVICE timeout --kill-after=$KILL_AFTER "$tmo" \
            "$ROCPROF" --pmc "${pmc[@]}" --output-format csv -d "$dir/rocprof" -- "$bin" "${args[@]}" \
            >"$dir/stdout_raw.log" 2>"$dir/stderr.log" &
        CHILD_PID=$!
        wait "$CHILD_PID"
        code=$?
        CHILD_PID=""
        t1=$(date +%s%N)
        if { ((code == 124)) || ((code == 137)); } && timed_out "$t0" "$t1" "$tmo"; then
            PASS_REASON="pass $k: killed after $tmo s"; return
        fi
        # the profiler may print log lines on stdout: the JSON line is the last line starting with {
        grep '^{' "$dir/stdout_raw.log" | tail -n 1 >"$dir/stdout.json"
        if ((code != 0 && code != 3)); then
            PASS_REASON="pass $k: exit code $code: $(last_line "$dir/stderr.log")"; return
        fi
        jstatus="" jreason="" roi="" perf=""
        while read -r tag rest; do
            case $tag in
                STATUS) jstatus=$rest ;;
                REASON) jreason=$rest ;;
                ROI)    roi=$rest ;;
                PERF)   perf=$rest ;;
            esac
        done < <(awk -v duration="$PROFILE_DURATION" -f "$LIB/json.awk" -f "$LIB/bench_json.awk" "$dir/stdout.json")
        if [[ $jstatus != ok || -z $roi ]]; then
            PASS_REASON="pass $k: contract: ${jreason:-no ROI}"; return
        fi
        csv=$(find "$dir/rocprof" -name '*counter_collection.csv' 2>/dev/null | head -n 1)
        if [[ -z $csv ]]; then
            PASS_REASON="pass $k: no counter_collection.csv from $ROCPROF"; return
        fi
        read -r t0 t1 <<<"$roi"
        read -r _ it <<<"$perf"
        meta=$(awk -v t0="$t0" -v t1="$t1" -v wmean="$WMEAN" -v prefix="$prefix,$k" -v iterations="$it" \
            -f "$LIB/counters.awk" "$csv")
        if grep -q '^ERR ' <<<"$meta"; then
            PASS_REASON="pass $k: $(grep -m1 '^ERR ' <<<"$meta" | cut -c5-)"; return
        fi
        read -r _ nd _ <<<"$(grep -m1 '^META ' <<<"$meta")"
        if [[ -z $nd ]] || ((nd == 0)); then
            PASS_REASON="pass $k: no kernel dispatch inside the ROI (profiler timestamps not on the monotonic clock?)"; return
        fi
        ((nd == it)) || log "       pass $k: $nd dispatches in the ROI, $it iterations"
        for n in "${pmc[@]}"; do
            grep -q "^ROW .*,\"$n\"," <<<"$meta" ||
                { PASS_REASON="pass $k: counter $n missing in the output of $ROCPROF"; return; }
        done
        rows+=$(grep '^ROW ' <<<"$meta" | cut -c5-)$'\n'
        sd=$(awk -v p="$perf" -v q="$perf0" 'BEGIN { split(p, a, " "); split(q, b, " ")
            if (a[2] > 0 && b[1] > 0 && b[2] > 0) printf "%.4f", (a[1] / a[2]) / (b[1] / b[2]) }')
        [[ -n $sd ]] && PFRAG+="${PFRAG:+,}\"m_prof_slowdown_p$k\":$sd"
    done
    printf '%s' "$rows" >>"$RESDIR/counters.csv"
}

# throttle_note: after a run, point out power throttling (from the telemetry of pass 0)
# when the GPU was at the power limit for more than THROTTLE_WARN_PCT % of the ROI;
# then the hardware limited the run, not the kernel (clock below max_sclk).
THROTTLE_WARN_PCT=5
throttle_note() {
    local ppt clk re='"m_ppt_viol_pct":([0-9.eE+-]+)' re2='"m_avg_gfx_clk_mhz":([0-9.eE+-]+)'
    [[ $MFRAG =~ $re ]] || return 0
    ppt=${BASH_REMATCH[1]}
    clk="?"
    [[ $MFRAG =~ $re2 ]] && clk=$(awk -v v="${BASH_REMATCH[1]}" 'BEGIN { printf "%.0f", v }')
    awk -v v="$ppt" -v t="$THROTTLE_WARN_PCT" 'BEGIN { exit !(v > t) }' || return 0
    ppt=$(awk -v v="$ppt" 'BEGIN { printf "%.0f", v }')
    printf '%*s-> throttled at the power limit in %s %% of the ROI, avg clock %s MHz\n' \
        $((LW + 2)) '' "$ppt" "$clk" >&2
    log "       throttled at the power limit in $ppt % of the ROI, avg clock $clk MHz"
}

# run_one <round> <index>: execute one run, write its record; sets RAN=1 if executed
run_one() {
    local r=$1 i=$2
    local label=${RUN_LABEL[i]} bench=${RUN_BENCH[i]}
    local dir="$RESDIR/runs/$label/rep$r/pass0"
    local bin="$BIN_DIR/$bench"
    local status="" reason="" rc="null" fragment="" cmd tmo t0 t1 code
    local jstatus="" jreason="" jverify="null" tag rest roi="" bdf="" perf=""
    local -a args
    read -r -a args <<<"${RUN_EFF[i]}"
    printf -v cmd '%q ' "HIP_VISIBLE_DEVICES=$DEVICE" "$bin" "${args[@]}"
    cmd=${cmd% }
    RAN=0
    MFRAG="" PFRAG=""

    if [[ -n ${RUN_SKIP_PREC[i]} ]]; then
        status=skipped
        reason=${RUN_SKIP_PREC[i]}
    else
        RAN=1
        mkdir -p "$dir"
        printf '%s\n' "$cmd" >"$dir/command.txt"
        tmo=${OPT_TIMEOUT:-$(awk -v d="${RUN_DUR[i]}" -v f=$TIMEOUT_FACTOR -v o=$TIMEOUT_OFFSET \
            'BEGIN { printf "%.3f", d * f + o }')}
        monitoring && TEL_LINES=$(wc -l <"$TELEMETRY")
        t0=$(date +%s%N)
        HIP_VISIBLE_DEVICES=$DEVICE timeout --kill-after=$KILL_AFTER "$tmo" "$bin" "${args[@]}" \
            >"$dir/stdout.json" 2>"$dir/stderr.log" &
        CHILD_PID=$!
        wait "$CHILD_PID"
        code=$?
        CHILD_PID=""
        t1=$(date +%s%N)

        if { ((code == 124)) || ((code == 137)); } && timed_out "$t0" "$t1" "$tmo"; then
            status=timeout
            reason="killed after ${tmo} s"
        else
            rc=$code
            case $code in
                0 | 3)
                    while read -r tag rest; do
                        case $tag in
                            STATUS)   jstatus=$rest ;;
                            REASON)   jreason=$rest ;;
                            VERIFY)   jverify=$rest ;;
                            ROI)      roi=$rest ;;
                            BDF)      bdf=$rest ;;
                            PERF)     perf=$rest ;;
                            FRAGMENT) fragment=$rest ;;
                        esac
                    done < <(awk -v duration="${RUN_DUR[i]}" -f "$LIB/json.awk" -f "$LIB/bench_json.awk" \
                        "$dir/stdout.json")
                    if [[ $jstatus != ok ]]; then
                        status=error
                        reason="contract: ${jreason:-could not check output}"
                    elif ((code == 3)); then
                        status=invalid
                        reason="verification failed"
                        [[ $jverify == false ]] || reason+=" (but verify_ok=$jverify)"
                    elif [[ $jverify == false ]]; then
                        status=invalid
                        reason="exit code 0 but verify_ok=false"
                    else
                        status=ok
                    fi
                    # telemetry of the ROI; a run without telemetry is an error (section 13)
                    if monitoring && [[ $status != error && -n $roi && -n $bdf ]]; then
                        telemetry_for_run "$dir" "$roi" "$bdf"
                        if [[ -n $TEL_REASON ]]; then
                            status=error
                            reason="telemetry: $TEL_REASON"
                        fi
                    fi
                    # hardware counters: passes 1..K (only after a usable pass 0)
                    if profiling && [[ $status == ok || $status == invalid ]]; then
                        run_passes "$r" "$i" "$perf"
                        if [[ -n $PASS_REASON ]]; then
                            status=error
                            reason="counters: $PASS_REASON"
                            PFRAG=""
                        fi
                    fi
                    ;;
                1) status=error;   reason="exit code 1: $(last_line "$dir/stderr.log")" ;;
                2) status=skipped; reason="not supported on this hardware: $(last_line "$dir/stderr.log")" ;;
                *)
                    status=error
                    if ((code > 128)); then reason="terminated by signal $((code - 128))"
                    else reason="unexpected exit code $code: $(last_line "$dir/stderr.log")"; fi
                    ;;
            esac
        fi
    fi

    local rec
    rec="{\"label\":$(json_str "$label"),\"rep\":$r,\"benchmark\":$(json_str "$bench")"
    rec+=",\"benchmark_version\":$(json_str "$(meta_get "$bench" version)"),\"mode\":$(json_str "$MODE")"
    rec+=",\"device\":$DEVICE,\"status\":$(json_str "$status"),\"reason\":$(json_str "$reason")"
    rec+=",\"exit_code\":$rc,\"cmd\":$(json_str "$cmd")"
    [[ -n $fragment ]] && rec+=",$fragment"
    [[ -n $MFRAG ]] && rec+=",$MFRAG"
    [[ -n $PFRAG ]] && rec+=",$PFRAG"
    rec+="}"

    printf '%s\n' "$rec" >>"$RESDIR/results.jsonl"
    awk -f "$LIB/json.awk" -f "$LIB/jsonl2csv.awk" "$RESDIR/results.jsonl" >"$RESDIR/results.csv.tmp" &&
        mv -f "$RESDIR/results.csv.tmp" "$RESDIR/results.csv"
    printf '%s\n' "$rec" | awk -v kind=run -v tsv="$RESDIR/summary.tsv" -v lw="$LW" -v bw="$BW" \
        -f "$LIB/json.awk" -f "$LIB/row.awk"

    log "RUN    $label rep $r: $status${reason:+ ($reason)}"
    throttle_note
    if [[ $status != ok ]]; then
        log "       $cmd"
        printf '%*s-> %s\n' $((LW + 2)) '' "$reason" >&2
    fi
    COUNT[$status]=$((COUNT[$status] + 1))
}

execute() {
    local p r i n=0 last_exec=0 k
    # index of the last plan entry that is actually executed (no pause after it)
    for k in "${!PLAN[@]}"; do
        i=${PLAN[k]#*:}
        [[ -z ${RUN_SKIP_PREC[i]} ]] && last_exec=$k
    done

    awk -v kind=run -v header=1 -v tsv="$RESDIR/summary.tsv" -v lw="$LW" -v bw="$BW" \
        -f "$LIB/json.awk" -f "$LIB/row.awk"
    for k in "${!PLAN[@]}"; do
        p=${PLAN[k]}
        r=${p%%:*}
        i=${p#*:}
        if monitoring && ! sampler_alive; then
            SAMPLER_DIED=1
            break
        fi
        run_one "$r" "$i"
        if ((RAN)) && ((k < last_exec)) && awk -v p="$PAUSE" 'BEGIN { exit !(p > 0) }'; then
            sleep "$PAUSE" &
            CHILD_PID=$!
            wait "$CHILD_PID"
            CHILD_PID=""
        fi
    done
}

# ---------------------------------------------------------------------------
# Phase 8: finish
# ---------------------------------------------------------------------------

finish() {
    local line agg="$RESDIR/aggregates.jsonl.tmp" warnf="$RESDIR/aggregate.warn.tmp"
    if ((SAMPLER_DIED)) || { [[ -n $SAMPLER_PID ]] && ! sampler_alive; }; then
        SAMPLER_DIED=1
        err "the sampler stopped unexpectedly; remaining runs were not executed (see $RESDIR/sampler.log)"
    fi
    stop_sampler
    if ((REPEAT > 1)); then
        awk -f "$LIB/json.awk" -f "$LIB/aggregate.awk" "$RESDIR/results.jsonl" >"$agg" 2>"$warnf"
        while IFS= read -r line; do warn "$line"; done <"$warnf"
        awk -f "$LIB/json.awk" -f "$LIB/jsonl2csv.awk" "$agg" >"$RESDIR/aggregates.csv"
        printf '\n'
        awk -v kind=agg -v header=1 -v tsv="$RESDIR/aggregates.tsv" -v lw="$LW" -v bw="$BW" \
            -f "$LIB/json.awk" -f "$LIB/row.awk"
        awk -v kind=agg -v tsv="$RESDIR/aggregates.tsv" -v lw="$LW" -v bw="$BW" \
            -f "$LIB/json.awk" -f "$LIB/row.awk" "$agg"
        rm -f "$agg" "$warnf"
    fi

    local summary="ok ${COUNT[ok]}, skipped ${COUNT[skipped]}, error ${COUNT[error]}, invalid ${COUNT[invalid]}, timeout ${COUNT[timeout]}"
    info ""
    info "Done: $summary"
    info "Results: $RESDIR"
    log "END    $summary"
    if ((SAMPLER_DIED || COUNT[error] + COUNT[invalid] + COUNT[timeout] > 0)); then
        info "Details of failed runs: $RESDIR/wrapper.log and runs/<label>/rep<i>/pass<k>/stderr.log"
        exit 4
    fi
    exit 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ORIG_ARGS="$*"
require_env
parse_args "$@"

# phase 2: load
comp_load "$COMPONENT"
parse_runs "$CONFIG"
select_runs
((${#SEL[@]})) || { report_errors "nothing was run"; exit 1; }
if [[ -n ${COMP[gpus]:-} && ${COMP[gpus]} =~ ^[0-9]+$ ]] && ((DEVICE >= COMP[gpus])); then
    add_error "--device $DEVICE: the component config defines only ${COMP[gpus]} GPU(s)"
fi

# phase 3: validate
validate_runs
check_args
[[ $MODE == check ]] && check_param_defaults
profiling && load_counters
if monitoring && [[ ! -x $BIN_DIR/sampler ]]; then
    add_error "telemetry: sampler '$BIN_DIR/sampler' not found or not executable (build it with: build.sh sampler)"
fi
report_errors "nothing was run" || exit 1

# phase 4: plan
build_plan
LW=5 BW=9
for i in "${SEL[@]}"; do
    ((${#RUN_LABEL[i]} > LW)) && LW=${#RUN_LABEL[i]}
    ((${#RUN_BENCH[i]} > BW)) && BW=${#RUN_BENCH[i]}
done
print_plan
if ((DRY_RUN)); then info "Dry run: nothing was executed."; exit 0; fi

# phase 5-8
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
prepare
if monitoring; then
    start_sampler
    idle_baseline
fi
info ""
execute
finish
