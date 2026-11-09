#!/usr/bin/env bash
# plan_counter_passes.sh - distribute the wished counters over profiling passes (contract section 13)
#
# Usage: tools/plan_counter_passes.sh [--component <file>] [--wishlist <file>] [--out <file>]
#                                     [--rocprof <cmd>] [--device <n>] [--test-command "<cmd>"]
#                                     [--test-timeout <s>]
#
# Reads the wish list (config/counters_wishlist_<arch>.cfg), packs the counters greedily
# into passes and tests every candidate pass with rocprofv3 on a short test run: a
# counter is added to the current pass if rocprofv3 accepts the combination and
# reports every counter; otherwise a new pass is started. A counter that is
# rejected even alone is skipped with a warning.
# Writes config/counters_<arch>.cfg (pass.<k>=... and the agg.* lines of the wish list).
# Run once per system, on a GPU node. Takes about one test run per counter; every test
# run is printed (on stderr) with its result and duration.

set -uo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/common.sh
source "$SUITE_DIR/lib/common.sh"

COMPONENT=$SUITE_DIR/config/component.cfg
WISHLIST=""
OUT=""
ROCPROF=${ROCPROF:-rocprofv3}
DEVICE=0
TEST_CMD=""
TEST_TIMEOUT=60

while (($#)); do
    case $1 in
        --component)    COMPONENT=${2:?--component requires a value}; shift 2 ;;
        --wishlist)     WISHLIST=${2:?--wishlist requires a value}; shift 2 ;;
        --out)          OUT=${2:?--out requires a value}; shift 2 ;;
        --rocprof)      ROCPROF=${2:?--rocprof requires a value}; shift 2 ;;
        --device)       DEVICE=${2:?--device requires a value}; shift 2 ;;
        --test-command) TEST_CMD=${2:?--test-command requires a value}; shift 2 ;;
        --test-timeout) TEST_TIMEOUT=${2:?--test-timeout requires a value}; shift 2 ;;
        --help | -h)    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

comp_load "$COMPONENT"
report_errors "aborted" || exit 1
ARCH=${COMP[arch]}
WISHLIST=${WISHLIST:-$SUITE_DIR/config/counters_wishlist_$ARCH.cfg}
OUT=${OUT:-$SUITE_DIR/config/counters_$ARCH.cfg}
TEST_CMD=${TEST_CMD:-$SUITE_DIR/build/fma --duration 0.001 --ops_per_element 1000 --precision fp64}

command -v "$ROCPROF" >/dev/null 2>&1 || die "rocprofv3 not found ('$ROCPROF'; set --rocprof or load the ROCm module)"
read -r -a TEST <<<"$TEST_CMD"
[[ $TEST_TIMEOUT =~ ^[1-9][0-9]*$ ]] || die "--test-timeout must be an integer >= 1 (s)"
[[ -x ${TEST[0]} ]] || die "test command '${TEST[0]}' not found or not executable (run build.sh first)"

# ---- wish list ----
declare -A WISH=()
WISH_ORDER=()
[[ -r $WISHLIST ]] || die "cannot read wish list '$WISHLIST'"
kv_parse WISH WISH_ORDER "$WISHLIST" <"$WISHLIST"
report_errors "aborted" || exit 1

COUNTERS=()
AGG_LINES=()
declare -A SEEN=()
for k in "${WISH_ORDER[@]}"; do
    case $k in
        want.*)
            IFS=, read -r -a list <<<"${WISH[$k]}"
            for c in "${list[@]}"; do
                trim c
                [[ -z $c ]] && continue
                [[ $c =~ ^[A-Za-z0-9_]+$ ]] || die "$WISHLIST: invalid counter name '$c' in $k"
                [[ -v SEEN[$c] ]] && { warn "counter $c listed twice, used once"; continue; }
                SEEN[$c]=1
                COUNTERS+=("$c")
            done ;;
        agg.*)
            [[ ${WISH[$k]} == sum || ${WISH[$k]} == wmean ]] || die "$WISHLIST: $k must be sum or wmean"
            AGG_LINES+=("$k=${WISH[$k]}") ;;
        *) die "$WISHLIST: unknown key '$k' (want.<group> or agg.<counter>)" ;;
    esac
done
((${#COUNTERS[@]})) || die "$WISHLIST: no counters"

# ---- test one combination ----
TMP=$(mktemp -d)
CHILD_PID=""
trap 'rm -rf "$TMP"' EXIT
# timeout puts the profiler into its own process group, so Ctrl-C from the terminal
# does not reach it. The test runs in the background; on INT/TERM it is stopped here.
on_signal() {
    trap '' INT TERM
    printf '\n' >&2
    [[ -n $CHILD_PID ]] && { kill -TERM "$CHILD_PID" 2>/dev/null; wait "$CHILD_PID" 2>/dev/null; }
    warn "aborted by SIG$1; nothing was written"
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
NTEST=0

# accepts <counter...>: 0 if rocprofv3 collects all of them in one pass.
# Prints one progress line per test run: number, pass size, new counter, result, duration.
accepts() {
    local d="$TMP/t$NTEST" csv c rc t0 t1 res=ok
    NTEST=$((NTEST + 1))
    printf '  test %3d: %2d counter(s), last %-34s ' "$NTEST" "$#" "${!#}" >&2
    t0=$(date +%s%N)
    HIP_VISIBLE_DEVICES=$DEVICE timeout --kill-after=10 "$TEST_TIMEOUT" "$ROCPROF" --pmc "$@" \
        --output-format csv -d "$d" -- "${TEST[@]}" >"$TMP/out" 2>"$TMP/err" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""
    t1=$(date +%s%N)
    if ((rc == 124 || rc == 137)); then res="timeout after $TEST_TIMEOUT s"
    elif ((rc != 0)); then res="rejected (exit $rc)"
    else
        csv=$(find "$d" -name '*counter_collection.csv' 2>/dev/null | head -1)
        if [[ -z $csv ]]; then res="rejected (no counter_collection.csv)"
        else
            for c in "$@"; do
                grep -qF "\"$c\"" "$csv" || { res="rejected ($c missing in output)"; break; }
            done
        fi
    fi
    awk -v r="$res" -v a="$t0" -v b="$t1" 'BEGIN { printf "%s  %.1f s\n", r, (b - a) / 1e9 }' >&2
    rm -rf "$d"
    [[ $res == ok ]]
}

# ---- greedy packing ----
PASSES=()
SKIPPED=()
current=()
info "planning ${#COUNTERS[@]} counters with $ROCPROF on device $DEVICE (test: $TEST_CMD)"
for c in "${COUNTERS[@]}"; do
    if ((${#current[@]})) && accepts "${current[@]}" "$c"; then
        current+=("$c")
        continue
    fi
    if accepts "$c"; then
        if ((${#current[@]})); then
            PASSES+=("$(IFS=,; echo "${current[*]}")")
            info "  pass ${#PASSES[@]}: ${#current[@]} counters"
        fi
        current=("$c")
    else
        warn "counter $c is rejected even alone: skipped ($(last=$(tail -1 "$TMP/err" 2>/dev/null); echo "${last:0:120}"))"
        SKIPPED+=("$c")
    fi
done
if ((${#current[@]})); then
    PASSES+=("$(IFS=,; echo "${current[*]}")")
    info "  pass ${#PASSES[@]}: ${#current[@]} counters"
fi
((${#PASSES[@]})) || die "no counter could be collected"

# ---- write the counter config ----
{
    echo "# Counter passes for $ARCH, generated by tools/plan_counter_passes.sh"
    echo "# $(date '+%Y-%m-%d %H:%M:%S') on $(hostname), $($ROCPROF --version 2>/dev/null | head -1)"
    echo "# from $WISHLIST; ${#COUNTERS[@]} wished, ${#SKIPPED[@]} skipped, ${#PASSES[@]} passes, $NTEST test runs"
    if ((${#SKIPPED[@]})); then echo "# skipped (rejected even alone): ${SKIPPED[*]}"; fi
    echo
    for i in "${!PASSES[@]}"; do echo "pass.$((i + 1))=${PASSES[i]}"; done
    if ((${#AGG_LINES[@]})); then echo; printf '%s\n' "${AGG_LINES[@]}"; fi
} >"$OUT.tmp" && mv -f "$OUT.tmp" "$OUT" || die "cannot write '$OUT'"

info "wrote $OUT: ${#PASSES[@]} passes, ${#SKIPPED[@]} counter(s) skipped"
