#!/usr/bin/env bash
# run_tests.sh - black-box tests of run_suite.sh and build.sh with mock benchmarks (no GPU)
#
# Usage: tests/run_tests.sh [-v] [<test> ...]
#   -v        show the wrapper output of failed tests
#   <test>    run only these tests (names as printed, e.g. timeout); default: all
#
# Principles
#   - one test = one behaviour. A test fails if one of its checks fails.
#   - every test is independent: own directory with own run config and results/
#   - a check fails if the thing it checks does not exist (never green by absence)
#   - new tests must fail before the code they test is written or fixed (red before green)
#
# The mock benchmarks in tests/mock/ imitate benchmarks that follow (or
# deliberately violate) the contract; see tests/mock/mockbench.sh.

set -uo pipefail

TESTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUITE=$(cd -- "$TESTS_DIR/.." && pwd)
MOCK=$TESTS_DIR/mock

VERBOSE=0
ONLY_TESTS=()
for a in "$@"; do
    if [[ $a == -v ]]; then VERBOSE=1; else ONLY_TESTS+=("$a"); fi
done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ===========================================================================
# Framework
# ===========================================================================

TESTS=()          # registered tests, in order
CDIR=""           # directory of the current test
FAILS=()          # failed checks of the current test
RC=""             # exit code of the last wrapper call
DIR=""            # result directory of the last wrapper call ("" if none)

register() { TESTS+=("$1"); }

fail_check() { FAILS+=("$*"); }

# cfg <text>: write the run config of the current test
cfg() { printf '%s\n' "$1" >"$CDIR/runs.cfg"; }

# wrapper command with the mocks. Later options override earlier ones,
# so a test can pass its own --config, --component etc.
wrap_cmd() {
    WRAP_CMD=("$SUITE/run_suite.sh" --src-dir "$MOCK/src" --bin-dir "$MOCK/bin"
              --component "$MOCK/component.cfg" --config "$CDIR/runs.cfg"
              --results "$CDIR/results" "$@")
}

# wrap <args...>: run the wrapper; sets RC and DIR
wrap() {
    wrap_cmd "$@"
    "${WRAP_CMD[@]}" >"$CDIR/out" 2>"$CDIR/err"
    RC=$?
    find_dir
}

# find_dir: set DIR to the result directory of the current test ("" if none)
find_dir() {
    local dirs=()
    shopt -s nullglob
    dirs=("$CDIR"/results/*/)
    shopt -u nullglob
    DIR=""
    if ((${#dirs[@]} == 1)); then DIR=${dirs[0]%/}
    elif ((${#dirs[@]} > 1)); then fail_check "more than one result directory"; fi
}

# need_dir: the wrapper must have created exactly one result directory
need_dir() {
    [[ -n $DIR ]] && return 0
    fail_check "no result directory created"
    return 1
}

# record <label> <rep>: the results.jsonl line of one run (empty if missing)
record() {
    awk -v l="\"label\":\"$1\"," -v r="\"rep\":$2," 'index($0, l) && index($0, r)' "$DIR/results.jsonl" 2>/dev/null
}

check_rc() { ((RC == $1)) || fail_check "exit code $RC, expected $1"; }

# check_status <label> <rep> <status>
check_status() {
    need_dir || return
    local rec got
    rec=$(record "$1" "$2")
    [[ -n $rec ]] || { fail_check "no record for $1 rep $2"; return; }
    got=$(sed -n 's/.*"status":"\([a-z]*\)".*/\1/p' <<<"$rec")
    [[ $got == "$3" ]] || fail_check "$1 rep $2: status '$got', expected '$3'"
}

# check_record <label> <rep> <text>: the run's record contains <text>
check_record() {
    need_dir || return
    local rec
    rec=$(record "$1" "$2")
    [[ -n $rec ]] || { fail_check "no record for $1 rep $2"; return; }
    [[ $rec == *"$3"* ]] || fail_check "record of $1 rep $2 does not contain '$3'"
}

# check_contains <file> <text>
check_contains() {
    [[ -f $1 ]] || { fail_check "file $1 missing"; return; }
    grep -qF -- "$2" "$1" || fail_check "'$2' not found in $(basename "$1")"
}

# check_not_contains <file> <text>
check_not_contains() {
    [[ -f $1 ]] || { fail_check "file $1 missing"; return; }
    ! grep -qF -- "$2" "$1" || fail_check "'$2' unexpectedly found in $(basename "$1")"
}

check_exists() { [[ -e $1 ]] || fail_check "$1 missing"; }

# check_absent <path>: path does not exist, but its parent does
check_absent() {
    [[ -d $(dirname "$1") ]] || { fail_check "parent of $1 missing"; return; }
    [[ ! -e $1 ]] || fail_check "$1 exists"
}

check_no_results() { [[ ! -e $CDIR/results ]] || fail_check "a result directory was created"; }

# check_rejected <message>: validation failed, message reported, nothing run
check_rejected() {
    check_rc 1
    check_contains "$CDIR/err" "$1"
    check_no_results
}

# check_csv <file> <rows>: same number of fields in every line and <rows> data rows
check_csv() {
    [[ -f $1 ]] || { fail_check "file $1 missing"; return; }
    local res
    res=$(awk -v want="$2" '
        { n = 0; q = 0
          for (i = 1; i <= length($0); i++) { c = substr($0, i, 1)
              if (c == "\"") q = !q; else if (c == "," && !q) n++ }
          if (NR == 1) h = n; else if (n != h) { print "line " NR ": " n + 1 " fields, header " h + 1; exit } }
        END { if (NR - 1 != want) print NR - 1 " data rows, expected " want }' "$1")
    [[ -z $res ]] || fail_check "$(basename "$1"): $res"
}

# csv_value <file> <first-column value incl. quotes> <column name>
csv_value() {
    awk -F, -v key="$2" -v col="$3" '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) c = i; next }
        $1 == key && c { print $c; exit }' "$1" 2>/dev/null
}

# order of labels in results.jsonl, one character per run (labels are single letters)
label_order() {
    awk '{ if (match($0, /"label":"[a-z]"/)) printf "%s", substr($0, RSTART + 9, 1) }' "$DIR/results.jsonl" 2>/dev/null
}

run_test() {
    local name=$1 f
    CDIR=$TMP/$name
    mkdir -p "$CDIR"
    : >"$CDIR/out"; : >"$CDIR/err"
    FAILS=() RC="" DIR=""
    "test_$name"
    if ((${#FAILS[@]} == 0)); then
        PASS=$((PASS + 1))
        printf '  ok    %s\n' "$name"
    else
        FAIL=$((FAIL + 1))
        FAILED+=("$name")
        printf '  FAIL  %s\n' "$name"
        for f in "${FAILS[@]}"; do printf '          - %s\n' "$f"; done
        if ((VERBOSE)); then
            sed 's/^/          | /' "$CDIR/out" "$CDIR/err"
        fi
    fi
}

# ===========================================================================
# Execution and status of a run
# ===========================================================================

register ok_run
test_ok_run() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    check_rc 0
    check_status a 1 ok
    need_dir || return
    check_exists "$DIR/runs/a/rep1/pass0/command.txt"
    check_exists "$DIR/runs/a/rep1/pass0/stdout.json"
    check_exists "$DIR/runs/a/rep1/pass0/stderr.log"
}

register exit1_is_error_with_stderr_message
test_exit1_is_error_with_stderr_message() {
    cfg "a  mock_error  --duration {DURATION}"
    wrap --mode check
    check_rc 4
    check_status a 1 error
    check_record a 1 "hipMalloc failed"
}

register exit2_is_skipped
test_exit2_is_skipped() {
    cfg "a  mock_ok  --duration {DURATION} --precision fp16"
    wrap --mode check
    check_rc 0
    check_status a 1 skipped
    check_record a 1 "not supported on this hardware"
}

register exit3_is_invalid
test_exit3_is_invalid() {
    cfg "a  mock_verifyfail  --duration {DURATION}"
    wrap --mode check
    check_rc 4
    check_status a 1 invalid
}

register timeout
test_timeout() {
    cfg "a  mock_hang  --duration {DURATION}"
    wrap --mode check --timeout 1
    check_rc 4
    check_status a 1 timeout
    check_record a 1 '"exit_code":null'
}

register sigterm_aborts_and_keeps_results
test_sigterm_aborts_and_keeps_results() {
    local pid
    cfg "a  mock_ok    --duration {DURATION}
h  mock_hang  --duration {DURATION}"
    # start the wrapper itself in the background (not the function wrap, which would
    # run in a subshell that receives the signal instead of the wrapper)
    wrap_cmd --mode check
    "${WRAP_CMD[@]}" >"$CDIR/out" 2>"$CDIR/err" &
    pid=$!
    sleep 2
    kill -TERM "$pid"
    wait "$pid"
    RC=$?
    find_dir
    check_rc 5
    check_status a 1 ok
    need_dir || return
    check_contains "$DIR/results.csv" '"a",1,'
    check_contains "$DIR/wrapper.log" "ABORT"
    sleep 0.5
    if pgrep -fx "sleep 100000" >/dev/null; then
        fail_check "benchmark process still running"
        pkill -fx "sleep 100000"
    fi
}

# ===========================================================================
# Duration
# ===========================================================================

register check_mode_overrides_fixed_duration
test_check_mode_overrides_fixed_duration() {
    cfg "a  mock_ok  --duration 600"
    wrap --mode check --timeout 5     # without the override the run would take 600 s
    check_rc 0
    need_dir || return
    check_contains "$DIR/runs/a/rep1/pass0/command.txt" "--duration 0.001"
}

register placeholder_replaced_by_mode_duration
test_placeholder_replaced_by_mode_duration() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode production --duration 0.02 --pause 0
    check_rc 0
    need_dir || return
    check_contains "$DIR/runs/a/rep1/pass0/command.txt" "--duration 0.02"
}

register fixed_duration_kept_outside_check_mode
test_fixed_duration_kept_outside_check_mode() {
    cfg "a  mock_ok  --duration 0.03"
    wrap --mode production --duration 0.02 --pause 0
    check_rc 0
    need_dir || return
    check_contains "$DIR/runs/a/rep1/pass0/command.txt" "--duration 0.03"
}

# ===========================================================================
# Precision filter
# ===========================================================================

register precision_not_implemented_is_skipped
test_precision_not_implemented_is_skipped() {
    cfg "a  mock_ok  --duration {DURATION} --precision int8"
    wrap --mode check
    check_rc 0
    check_status a 1 skipped
    check_record a 1 "precision int8 is not implemented by mock_ok"
    check_record a 1 '"exit_code":null'
    need_dir || return
    check_absent "$DIR/runs/a"
}

register precision_not_supported_by_hardware_is_skipped
test_precision_not_supported_by_hardware_is_skipped() {
    cfg "a  mock_ok  --duration {DURATION} --precision int32"
    wrap --mode check
    check_rc 0
    check_status a 1 skipped
    check_record a 1 "precision int32 is not supported by the hardware"
    check_record a 1 '"exit_code":null'
    need_dir || return
    check_absent "$DIR/runs/a"
}

# ===========================================================================
# Contract check of the benchmark output
# ===========================================================================

# contract_case <mock> <expected text in reason> [<status>]
contract_case() {
    cfg "a  $1  --duration {DURATION}"
    wrap --mode check
    check_rc 4
    check_status a 1 "${3:-error}"
    check_record a 1 "$2"
}

register contract_invalid_json
test_contract_invalid_json() { contract_case mock_badjson "contract: invalid JSON"; }

register contract_two_lines
test_contract_two_lines() { contract_case mock_twolines "expected exactly one line on stdout, got 2"; }

register contract_missing_required_field
test_contract_missing_required_field() { contract_case mock_contract "required field 'e_pci_bus_id' missing"; }

register contract_wrong_rate
test_contract_wrong_rate() { contract_case mock_contract "d_gflop_per_s does not equal a_gflop / runtime_s"; }

register contract_runtime_shorter_than_duration
test_contract_runtime_shorter_than_duration() { contract_case mock_shortrun "is shorter than the requested duration"; }

register contract_roi_does_not_match_runtime
test_contract_roi_does_not_match_runtime() { contract_case mock_roi "ROI length in monotonic clock does not match runtime_s"; }

register contract_reserved_field_names
test_contract_reserved_field_names() {
    contract_case mock_reserved "field 'status' is reserved for the wrapper"
    check_record a 1 "prefix m_ is reserved for the wrapper"
    need_dir || return
    check_csv "$DIR/results.csv" 1      # the record must stay valid JSON (no duplicate keys)
}

register exit0_with_failed_verification_is_invalid
test_exit0_with_failed_verification_is_invalid() {
    contract_case mock_verify0 "exit code 0 but verify_ok=false" invalid
}

# ===========================================================================
# Validation of the run config: rejected before anything runs
# ===========================================================================

register duplicate_label
test_duplicate_label() {
    cfg "a  mock_ok  --duration {DURATION}
a  mock_ok  --duration {DURATION}"
    wrap --mode check
    check_rejected "label 'a' is already used in line 1"
}

register invalid_label_characters
test_invalid_label_characters() {
    cfg "a.b  mock_ok  --duration {DURATION}"
    wrap --mode check
    check_rejected "label 'a.b' has invalid characters"
}

register line_without_benchmark
test_line_without_benchmark() {
    cfg "a"
    wrap --mode check
    check_rejected "expected '<label> <benchmark> <arguments ...>'"
}

register empty_run_config
test_empty_run_config() {
    cfg "# only a comment"
    wrap --mode check
    check_rejected "no runs defined"
}

register unknown_benchmark
test_unknown_benchmark() {
    cfg "a  mock_nothere  --duration {DURATION}"
    wrap --mode check
    check_rejected "source '$MOCK/src/mock_nothere.cpp' not found"
}

register binary_missing
test_binary_missing() {
    mkdir -p "$CDIR/emptybin"
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check --bin-dir "$CDIR/emptybin"
    check_rejected "binary '$CDIR/emptybin/mock_ok' not found or not executable"
}

register missing_duration
test_missing_duration() {
    cfg "a  mock_ok  --precision fp64"
    wrap --mode check
    check_rejected "(a): --duration is missing"
}

register nonpositive_duration
test_nonpositive_duration() {
    cfg "a  mock_ok  --duration 0"
    wrap --mode production
    check_rejected "(a): --duration must be a positive number"
}

register placeholder_outside_duration
test_placeholder_outside_duration() {
    cfg "a  mock_ok  --duration {DURATION} --ops_per_element {DURATION}"
    wrap --mode check
    check_rejected "{DURATION} is only allowed as value of --duration"
}

register option_given_twice
test_option_given_twice() {
    cfg "a  mock_ok  --duration {DURATION} --duration 2"
    wrap --mode check
    check_rejected "(a): option --duration is given twice"
}

register odd_number_of_arguments
test_odd_number_of_arguments() {
    cfg "a  mock_ok  --duration"
    wrap --mode check
    check_rejected "(a): arguments must be pairs"
}

register quotes_in_arguments
test_quotes_in_arguments() {
    cfg "a  mock_ok  --duration {DURATION} --precision \"fp64\""
    wrap --mode check
    check_rejected "(a): quotes are not allowed in arguments"
}

register check_option_in_run_config
test_check_option_in_run_config() {
    cfg "a  mock_ok  --duration {DURATION} --check 1"
    wrap --mode check
    check_rejected "(a): --check must not be used in the run config"
}

register benchmark_check_rejects_arguments
test_benchmark_check_rejects_arguments() {
    cfg "a  mock_ok  --duration {DURATION} --ops_per_element 0"
    wrap --mode check
    check_rejected "(a): 'mock_ok --check' rejected the arguments (exit 1)"
}

register only_with_unknown_label
test_only_with_unknown_label() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check --only a,nope
    check_rejected "--only: label 'nope' not found"
}

register all_errors_reported_together
test_all_errors_reported_together() {
    cfg "a  mock_ok  --precision fp64
b  mock_nothere  --duration {DURATION}"
    wrap --mode check
    check_rejected "(a): --duration is missing"
    check_contains "$CDIR/err" "mock_nothere.cpp' not found"
}

# ===========================================================================
# Metadata
# ===========================================================================

# metadata_case <expected message>: mock_badmeta contains several deliberate errors
metadata_case() {
    cfg "a  mock_badmeta  --duration {DURATION}"
    wrap --mode check
    check_rejected "$1"
}

register metadata_invalid_level
test_metadata_invalid_level() { metadata_case "level must be 1, 2 or 3"; }

register metadata_invalid_version
test_metadata_invalid_version() { metadata_case "version must be MAJOR.MINOR.PATCH"; }

register metadata_unknown_key
test_metadata_unknown_key() { metadata_case "unknown metadata key 'colour'"; }

register metadata_malformed_parameter
test_metadata_malformed_parameter() { metadata_case "param.duration must have the form <default>|<valid values>|<effect>"; }

register check_mode_rejects_invalid_documented_default
test_check_mode_rejects_invalid_documented_default() {
    cfg "a  mock_baddefault  --duration {DURATION}"
    wrap --mode check
    check_rejected "documented default of --ops_per_element (0) is rejected by --check"
}

register documented_defaults_checked_only_in_check_mode
test_documented_defaults_checked_only_in_check_mode() {
    cfg "a  mock_baddefault  --duration {DURATION}"
    wrap --mode production --duration 0.02 --pause 0
    check_rc 0
    check_status a 1 ok
}

# ===========================================================================
# Component config and options
# ===========================================================================

# component_case <component config text> <expected message>
component_case() {
    printf '%s\n' "$1" >"$CDIR/component.cfg"
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check --component "$CDIR/component.cfg"
    check_rejected "$2"
}

register component_invalid_arch
test_component_invalid_arch() {
    component_case $'arch=sm_90\ngpus=1\nprecisions=fp64' "arch 'sm_90' is not a gfx target"
}

register component_invalid_gpus
test_component_invalid_gpus() {
    component_case $'arch=gfx942\ngpus=0\nprecisions=fp64' "gpus must be a positive integer"
}

register component_missing_key
test_component_missing_key() {
    component_case $'arch=gfx942\ngpus=1' "required key 'precisions' missing or empty"
}

register component_unknown_key
test_component_unknown_key() {
    component_case $'arch=gfx942\ngpus=1\nprecisions=fp64\nfoo=1' "unknown key 'foo'"
}

register device_beyond_gpu_count
test_device_beyond_gpu_count() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check --device 2
    check_rejected "--device 2: the component config defines only 2 GPU(s)"
}

register unknown_mode
test_unknown_mode() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode fast
    check_rejected "unknown mode 'fast'"
}

register missing_mode
test_missing_mode() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap
    check_rejected "--mode is required"
}

register invalid_repeat
test_invalid_repeat() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check --repeat 0
    check_rejected "--repeat must be an integer >= 1"
}

# ===========================================================================
# Output files
# ===========================================================================

register stdout_contains_only_the_table
test_stdout_contains_only_the_table() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    check_rc 0
    check_contains "$CDIR/out" "status"
    check_not_contains "$CDIR/out" "Plan:"
    check_contains "$CDIR/err" "Plan:"
}

register csv_one_consistent_row_per_run
test_csv_one_consistent_row_per_run() {
    cfg "a  mock_ok     --duration {DURATION}
b  mock_error  --duration {DURATION}
c  mock_ok     --duration {DURATION} --precision int8"
    wrap --mode check
    need_dir || return
    check_csv "$DIR/results.csv" 3
}

register csv_wrapper_fields_first
test_csv_wrapper_fields_first() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    need_dir || return
    check_contains "$DIR/results.csv" "label,rep,benchmark,benchmark_version,mode,device,status,reason,exit_code,cmd,"
}

register summary_tsv_one_row_per_run
test_summary_tsv_one_row_per_run() {
    cfg "a  mock_ok  --duration {DURATION}
b  mock_ok  --duration {DURATION}"
    wrap --mode check
    need_dir || return
    check_exists "$DIR/summary.tsv"
    local rows
    rows=$(awk 'END { print NR - 1 }' "$DIR/summary.tsv" 2>/dev/null)
    [[ $rows == 2 ]] || fail_check "summary.tsv has $rows data rows, expected 2"
}

register wrapper_log_contains_plan
test_wrapper_log_contains_plan() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    need_dir || return
    check_contains "$DIR/wrapper.log" "PLAN     a"
}

register configs_and_sources_copied
test_configs_and_sources_copied() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    need_dir || return
    check_contains "$DIR/config/runs.cfg" "mock_ok"
    check_contains "$DIR/config/component.cfg" "arch=gfx942"
    check_contains "$DIR/config/src/mock_ok.cpp" "benchmark=mock_ok"
}

# ===========================================================================
# Repetitions, order, aggregation
# ===========================================================================

register repetitions_run_in_rounds
test_repetitions_run_in_rounds() {
    cfg "a  mock_ok  --duration {DURATION}
b  mock_ok  --duration {DURATION}"
    wrap --mode check --repeat 2
    check_rc 0
    need_dir || return
    local order
    order=$(label_order)
    [[ $order == abab ]] || fail_check "order '$order', expected 'abab'"
}

register shuffle_is_reproducible
test_shuffle_is_reproducible() {
    cfg "a  mock_ok  --duration {DURATION}
b  mock_ok  --duration {DURATION}
c  mock_ok  --duration {DURATION}
d  mock_ok  --duration {DURATION}"
    local first second
    wrap --mode check --repeat 3 --shuffle 7
    need_dir || return
    first=$(label_order)
    rm -rf "$CDIR/results"
    wrap --mode check --repeat 3 --shuffle 7
    need_dir || return
    second=$(label_order)
    [[ ${#first} == 12 ]] || fail_check "expected 12 runs, got '$first'"
    [[ $first == "$second" ]] || fail_check "orders differ: '$first' vs '$second'"
}

register shuffle_changes_order_between_rounds
test_shuffle_changes_order_between_rounds() {
    cfg "a  mock_ok  --duration {DURATION}
b  mock_ok  --duration {DURATION}
c  mock_ok  --duration {DURATION}
d  mock_ok  --duration {DURATION}"
    wrap --mode check --repeat 3 --shuffle 7
    need_dir || return
    local o r
    o=$(label_order)
    [[ ${#o} == 12 ]] || { fail_check "expected 12 runs, got '$o'"; return; }
    for r in 0 4 8; do
        [[ $(fold -w1 <<<"${o:r:4}" | sort | tr -d '\n') == abcd ]] ||
            fail_check "round $((r / 4 + 1)) ('${o:r:4}') does not contain every run once"
    done
    [[ ${o:0:4} != "${o:4:4}" || ${o:4:4} != "${o:8:4}" ]] || fail_check "same order in all rounds: '$o'"
}

register aggregation_statistics
test_aggregation_statistics() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode production --duration 0.02 --pause 0 --repeat 3
    check_rc 0
    need_dir || return
    check_csv "$DIR/aggregates.csv" 1
    local v
    v=$(csv_value "$DIR/aggregates.csv" '"a"' n_ok)
    [[ $v == 3 ]] || fail_check "n_ok is '$v', expected 3"
    for c in d_gflop_per_s_mean d_gflop_per_s_std d_gflop_per_s_min d_gflop_per_s_max d_gflop_per_s_cv; do
        v=$(csv_value "$DIR/aggregates.csv" '"a"' "$c")
        [[ -n $v ]] || fail_check "column $c missing or empty"
    done
}

register aggregation_counts_only_ok_runs
test_aggregation_counts_only_ok_runs() {
    cfg "a  mock_ok  --duration {DURATION} --precision fp16"
    wrap --mode check --repeat 2
    need_dir || return
    local v
    v=$(csv_value "$DIR/aggregates.csv" '"a"' n_ok)
    [[ $v == 0 ]] || fail_check "n_ok is '$v', expected 0"
    v=$(csv_value "$DIR/aggregates.csv" '"a"' n_total)
    [[ $v == 2 ]] || fail_check "n_total is '$v', expected 2"
}

register no_aggregation_without_repeat
test_no_aggregation_without_repeat() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode check
    need_dir || return
    check_absent "$DIR/aggregates.csv"
}

# ===========================================================================
# Dry run and help
# ===========================================================================

register dry_run_runs_nothing
test_dry_run_runs_nothing() {
    cfg "a  mock_ok  --duration {DURATION}"
    wrap --mode validation --dry-run
    check_rc 0
    check_contains "$CDIR/err" "Estimated time"
    check_no_results
}

register list_shows_benchmarks
test_list_shows_benchmarks() {
    wrap --list
    check_rc 0
    check_contains "$CDIR/out" "mock_ok"
}

register list_marks_invalid_metadata
test_list_marks_invalid_metadata() {
    wrap --list
    check_contains "$CDIR/out" "(invalid metadata"
}

register help_shows_benchmark_parameters
test_help_shows_benchmark_parameters() {
    wrap --help mock_ok
    check_rc 0
    check_contains "$CDIR/out" "--ops_per_element"
}

# ===========================================================================
# build.sh (with a fake compiler that records its arguments)
# ===========================================================================

# build <names...>: run build.sh with the fake compiler; sets RC
build() {
    mkdir -p "$CDIR/fakebin"
    cat >"$CDIR/fakebin/hipcc" <<'EOF'
#!/usr/bin/env bash
echo "$*" >>"${FAKE_LOG:?}"
while (($#)); do [[ $1 == -o ]] && { : >"$2"; chmod +x "$2"; }; shift; done
EOF
    chmod +x "$CDIR/fakebin/hipcc"
    FAKE_LOG=$CDIR/hipcc.log HIPCC=$CDIR/fakebin/hipcc "$SUITE/build.sh" --src-dir "$MOCK/src" \
        --bin-dir "$CDIR/build" --component "$MOCK/component.cfg" "$@" >"$CDIR/out" 2>"$CDIR/err"
    RC=$?
}

register build_uses_common_flags_and_arch
test_build_uses_common_flags_and_arch() {
    build mock_ok
    check_rc 0
    check_exists "$CDIR/build/mock_ok"
    check_contains "$CDIR/hipcc.log" "-O3 -std=c++17 -ffp-contract=fast --offload-arch=gfx942"
}

register build_skips_invalid_metadata
test_build_skips_invalid_metadata() {
    build mock_ok mock_badmeta
    check_rc 1
    check_exists "$CDIR/build/mock_ok"
    check_absent "$CDIR/build/mock_badmeta"
}

# ===========================================================================
# Main
# ===========================================================================

PASS=0 FAIL=0 FAILED=()
if ((${#ONLY_TESTS[@]})); then
    for t in "${ONLY_TESTS[@]}"; do
        if declare -F "test_$t" >/dev/null; then run_test "$t"
        else echo "unknown test: $t" >&2; FAIL=$((FAIL + 1)); fi
    done
else
    for t in "${TESTS[@]}"; do run_test "$t"; done
fi

echo
echo "passed: $PASS, failed: $FAIL"
((FAIL == 0)) || echo "failed: ${FAILED[*]}"
((FAIL == 0))
