#!/usr/bin/env bash
# run_tests.sh - tests of the wrapper and build script with mock benchmarks (no GPU needed)
#
# Usage: tests/run_tests.sh [-v]      (-v: show wrapper output of failed tests)
#
# The mock benchmarks in tests/mock/ imitate benchmarks that follow (or
# deliberately violate) the contract. Exit code 0 if all tests pass.

set -uo pipefail

TESTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUITE=$(cd -- "$TESTS_DIR/.." && pwd)
MOCK=$TESTS_DIR/mock
VERBOSE=0
[[ ${1:-} == -v ]] && VERBOSE=1

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PASS=0 FAIL=0 CASE=""

WRAP=("$SUITE/run_suite.sh" --src-dir "$MOCK/src" --bin-dir "$MOCK/bin"
      --component "$MOCK/component.cfg" --results "$TMP/results")

# ---------------------------------------------------------------------------

ok()   { PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"; }
fail() {
    FAIL=$((FAIL + 1)); printf '  FAIL  %s\n' "$1"
    if ((VERBOSE)); then
        sed 's/^/        | /' "$TMP/$CASE.out" "$TMP/$CASE.err" 2>/dev/null
    fi
}

# run_case <name> <run config text> <wrapper args...>: sets RC and DIR (newest result dir)
run_case() {
    CASE=$1
    printf '%s\n' "$2" >"$TMP/$CASE.cfg"
    shift 2
    "${WRAP[@]}" --config "$TMP/$CASE.cfg" "$@" >"$TMP/$CASE.out" 2>"$TMP/$CASE.err"
    RC=$?
    DIR=$(ls -td "$TMP"/results/*/ 2>/dev/null | head -1)
    DIR=${DIR%/}
}

expect_rc() { if ((RC == $1)); then ok "$CASE: exit code $1"; else fail "$CASE: exit code $RC, expected $1"; fi; }

# expect_in <file> <fixed string> [description]
expect_in() {
    if grep -qF -- "$2" "$1" 2>/dev/null; then ok "$CASE: ${3:-contains '$2'}"
    else fail "$CASE: ${3:-'$2' not found in $(basename "$1")}"; fi
}

# expect_status <label> <rep> <status>: check the status in results.jsonl
expect_status() {
    local got
    got=$(awk -v l="\"label\":\"$1\"" -v r="\"rep\":$2," 'index($0, l) && index($0, r)' "$DIR/results.jsonl" |
        sed -n 's/.*"status":"\([a-z]*\)".*/\1/p')
    if [[ $got == "$3" ]]; then ok "$CASE: $1 rep $2 is $3"
    else fail "$CASE: $1 rep $2 is '${got:-missing}', expected $3"; fi
}

# expect_csv <file> <rows>: same number of fields in every line, given number of data rows
expect_csv() {
    local res
    res=$(awk -v want="$2" '
        { n = 0; q = 0; line = $0
          for (i = 1; i <= length(line); i++) { c = substr(line, i, 1)
              if (c == "\"") q = !q; else if (c == "," && !q) n++ }
          if (NR == 1) h = n; else if (n != h) { print "line " NR ": " n + 1 " fields, header " h + 1; exit } }
        END { if (NR - 1 != want) print NR - 1 " data rows, expected " want }' "$1")
    if [[ -z $res ]]; then ok "$CASE: $(basename "$1") is a consistent CSV with $2 rows"
    else fail "$CASE: $(basename "$1"): $res"; fi
}

# ---------------------------------------------------------------------------

echo "== check mode: all statuses"
run_case statuses "
ok_fp64   mock_ok          --duration {DURATION} --precision fp64
ok_fp32   mock_ok          --duration 600        --precision fp32 --ops_per_element 50
ok_fp16   mock_ok          --duration {DURATION} --precision fp16
ok_int32  mock_ok          --duration {DURATION} --precision int32
err       mock_error       --duration {DURATION}
verify    mock_verifyfail  --duration {DURATION}
badjson   mock_badjson     --duration {DURATION}
contract  mock_contract    --duration {DURATION}" --mode check
expect_rc 4
expect_status ok_fp64 1 ok
expect_status ok_fp32 1 ok
expect_status ok_fp16 1 skipped
expect_status ok_int32 1 skipped
expect_status err 1 error
expect_status verify 1 invalid
expect_status badjson 1 error
expect_status contract 1 error
expect_in "$DIR/runs/ok_fp32/rep1/pass0/command.txt" "--duration 0.001" "fixed duration replaced in check mode"
expect_in "$DIR/results.jsonl" "required field 'e_pci_bus_id' missing" "contract violation reported"
expect_in "$DIR/results.jsonl" "d_gflop_per_s does not equal" "wrong rate detected"
expect_in "$DIR/results.jsonl" "hipMalloc failed" "stderr message of failed run in reason"
expect_in "$DIR/wrapper.log" "PLAN" "plan written to wrapper.log"
expect_in "$DIR/config/src/mock_ok.cpp" "benchmark=mock_ok" "benchmark source copied"
expect_csv "$DIR/results.csv" 8
expect_in "$DIR/results.csv" '"ok_fp64",1,"mock_ok"' "wrapper fields first in CSV"
[[ ! -d $DIR/runs/ok_int32 ]] && ok "$CASE: no run directory for filtered run" || fail "$CASE: run directory for filtered run"

echo "== check mode: success"
run_case allok "
a  mock_ok  --duration {DURATION}
b  mock_ok  --duration {DURATION} --precision fp32" --mode check
expect_rc 0
expect_in "$TMP/$CASE.out" "status" "table header on stdout"
if grep -q 'Plan:' "$TMP/$CASE.out"; then fail "$CASE: messages leaked to stdout"; else ok "$CASE: messages only on stderr"; fi

echo "== validation errors (all reported, nothing run)"
rm -rf "$TMP/results"
run_case invalid "
dup        mock_ok      --duration {DURATION}
dup        mock_ok      --duration {DURATION}
bad.label  mock_ok      --duration {DURATION}
nodur      mock_ok      --precision fp64
missing    mock_nothere --duration {DURATION}
rejected   mock_ok      --duration {DURATION} --ops_per_element 0
twice      mock_ok      --duration {DURATION} --duration 2
odd        mock_ok      --duration
place      mock_ok      --duration {DURATION} --ops_per_element {DURATION}
negdur     mock_ok      --duration -1
onlylabel" --mode production --only dup,nodur,missing,rejected,twice,odd,place,negdur,nope
expect_rc 1
expect_in "$TMP/$CASE.err" "label 'dup' is already used" "duplicate label"
expect_in "$TMP/$CASE.err" "label 'bad.label' has invalid characters" "invalid label"
expect_in "$TMP/$CASE.err" "(nodur): --duration is missing" "missing --duration"
expect_in "$TMP/$CASE.err" "source '$MOCK/src/mock_nothere.cpp' not found" "unknown benchmark"
expect_in "$TMP/$CASE.err" "(rejected): 'mock_ok --check' rejected the arguments" "--check rejects arguments"
expect_in "$TMP/$CASE.err" "(twice): option --duration is given twice" "option given twice"
expect_in "$TMP/$CASE.err" "(odd): arguments must be pairs" "odd number of tokens"
expect_in "$TMP/$CASE.err" "{DURATION} is only allowed as value of --duration" "misplaced placeholder"
expect_in "$TMP/$CASE.err" "(negdur): --duration must be a positive number" "negative duration"
expect_in "$TMP/$CASE.err" "expected '<label> <benchmark> <arguments ...>'" "line without benchmark"
expect_in "$TMP/$CASE.err" "--only: label 'nope' not found" "unknown --only label"
[[ ! -d $TMP/results ]] && ok "$CASE: no result directory created" || fail "$CASE: result directory created"

echo "== invalid metadata"
run_case badmeta "
x  mock_badmeta  --duration {DURATION}" --mode check
expect_rc 1
expect_in "$TMP/$CASE.err" "level must be 1, 2 or 3" "invalid level"
expect_in "$TMP/$CASE.err" "version must be MAJOR.MINOR.PATCH" "invalid version"
expect_in "$TMP/$CASE.err" "unknown metadata key 'colour'" "unknown key"
expect_in "$TMP/$CASE.err" "param.duration must have the form" "malformed param"

echo "== invalid component config"
printf 'arch=sm_90\ngpus=0\nprecisions=fp64\nfoo=1\n' >"$TMP/badcomp.cfg"
CASE=badcomp
"$SUITE/run_suite.sh" --src-dir "$MOCK/src" --bin-dir "$MOCK/bin" --component "$TMP/badcomp.cfg" \
    --config "$TMP/allok.cfg" --mode check >"$TMP/$CASE.out" 2>"$TMP/$CASE.err"
RC=$?
expect_rc 1
expect_in "$TMP/$CASE.err" "is not a gfx target" "invalid arch"
expect_in "$TMP/$CASE.err" "gpus must be a positive integer" "invalid gpus"
expect_in "$TMP/$CASE.err" "unknown key 'foo'" "unknown key"

echo "== repetitions, shuffle, aggregation"
rm -rf "$TMP/results"
run_case repeat "
a  mock_ok  --duration {DURATION} --precision fp64
b  mock_ok  --duration {DURATION} --precision fp32
s  mock_ok  --duration {DURATION} --precision fp16" --mode production --duration 0.05 --pause 0 --repeat 3 --shuffle 7
expect_rc 0
expect_csv "$DIR/results.csv" 9
expect_csv "$DIR/aggregates.csv" 3
expect_in "$DIR/aggregates.csv" "d_gflop_per_s_cv" "statistics columns"
ls "$DIR"/*.tmp >/dev/null 2>&1 && fail "$CASE: temporary files left" || ok "$CASE: no temporary files left"
n_ok=$(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "n_ok") c = i; next } $1 == "\"a\"" { print $c }' "$DIR/aggregates.csv")
[[ $n_ok == 3 ]] && ok "$CASE: n_ok = 3 for label a" || fail "$CASE: n_ok = '$n_ok' for label a, expected 3"
order=$(awk '{ match($0, /"label":"[a-z]"/); printf "%s", substr($0, RSTART + 9, 1) }' "$DIR/results.jsonl")
[[ ${order:0:3} != ${order:3:3} || ${order:3:3} != ${order:6:3} ]] && ok "$CASE: order differs between rounds ($order)" ||
    fail "$CASE: order identical in all rounds ($order)"
[[ $(printf '%s' "$order" | fold -w3 | sort | uniq -c | wc -l) -ge 1 && ${#order} == 9 ]] &&
    ok "$CASE: every run once per round" || fail "$CASE: rounds incomplete ($order)"
run_case repeat2 "$(cat "$TMP/repeat.cfg")" --mode production --duration 0.05 --pause 0 --repeat 3 --shuffle 7
order2=$(awk '{ match($0, /"label":"[a-z]"/); printf "%s", substr($0, RSTART + 9, 1) }' "$DIR/results.jsonl")
[[ $order == "$order2" ]] && ok "repeat: shuffle is reproducible with the same seed" ||
    fail "repeat: shuffle not reproducible ($order vs $order2)"

echo "== timeout"
run_case timeout "
h  mock_hang  --duration {DURATION}" --mode check --timeout 1
expect_rc 4
expect_status h 1 timeout

echo "== abort by signal"
rm -rf "$TMP/results"
CASE=signal
printf 'a mock_ok --duration {DURATION}\nh mock_hang --duration {DURATION}\n' >"$TMP/signal.cfg"
"${WRAP[@]}" --config "$TMP/signal.cfg" --mode check >"$TMP/$CASE.out" 2>"$TMP/$CASE.err" &
pid=$!
sleep 2
kill -TERM "$pid"
wait "$pid"
RC=$?
DIR=$(ls -td "$TMP"/results/*/ 2>/dev/null | head -1)
expect_rc 5
expect_in "${DIR%/}/results.jsonl" '"label":"a"' "completed run preserved"
expect_in "${DIR%/}/wrapper.log" "ABORT" "abort logged"
sleep 0.5
if pgrep -f "mockbench.sh hang" >/dev/null || pgrep -fx "sleep 100000" >/dev/null; then
    fail "$CASE: benchmark process still running"; pkill -fx "sleep 100000"
else ok "$CASE: benchmark process terminated"; fi

echo "== dry run, list, help"
rm -rf "$TMP/results"
run_case dryrun "$(cat "$TMP/allok.cfg")" --mode validation --dry-run
expect_rc 0
expect_in "$TMP/$CASE.err" "Estimated time" "plan printed"
[[ ! -d $TMP/results ]] && ok "$CASE: nothing executed" || fail "$CASE: result directory created"
CASE=list
"$SUITE/run_suite.sh" --src-dir "$MOCK/src" --list >"$TMP/$CASE.out" 2>&1; RC=$?
expect_rc 0
expect_in "$TMP/$CASE.out" "mock_ok" "benchmark listed"
expect_in "$TMP/$CASE.out" "mock_badmeta     (invalid metadata" "invalid metadata marked"
CASE=helpbench
"$SUITE/run_suite.sh" --src-dir "$MOCK/src" --bin-dir "$MOCK/bin" --help mock_ok >"$TMP/$CASE.out" 2>&1; RC=$?
expect_rc 0
expect_in "$TMP/$CASE.out" "--ops_per_element" "parameters shown"
CASE=badopt
"$SUITE/run_suite.sh" --mode fast >"$TMP/$CASE.out" 2>"$TMP/$CASE.err"; RC=$?
expect_rc 1
expect_in "$TMP/$CASE.err" "unknown mode 'fast'" "invalid mode rejected"

echo "== build.sh with a fake compiler"
CASE=build
mkdir -p "$TMP/fakebin"
cat >"$TMP/fakebin/hipcc" <<'EOF'
#!/usr/bin/env bash
echo "$*" >>"${FAKE_LOG:?}"
while (($#)); do [[ $1 == -o ]] && { : >"$2"; chmod +x "$2"; }; shift; done
EOF
chmod +x "$TMP/fakebin/hipcc"
FAKE_LOG=$TMP/hipcc.log HIPCC=$TMP/fakebin/hipcc "$SUITE/build.sh" --src-dir "$MOCK/src" \
    --bin-dir "$TMP/build" --component "$MOCK/component.cfg" mock_ok mock_badmeta >"$TMP/$CASE.out" 2>&1
RC=$?
expect_rc 1
[[ -x $TMP/build/mock_ok ]] && ok "$CASE: mock_ok built" || fail "$CASE: mock_ok not built"
[[ ! -e $TMP/build/mock_badmeta ]] && ok "$CASE: invalid metadata not built" || fail "$CASE: invalid metadata built"
expect_in "$TMP/hipcc.log" "-O3 -std=c++17 -ffp-contract=fast --offload-arch=gfx942" "common flags and arch"

echo
echo "passed: $PASS, failed: $FAIL"
((FAIL == 0))
