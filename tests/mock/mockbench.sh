#!/usr/bin/env bash
# mockbench.sh <behavior> <benchmark arguments...>
#
# Imitates a benchmark that follows the contract, without a GPU. Used by
# tests/run_tests.sh to test the wrapper. Behaviors:
#   ok          correct output (fp16 -> exit 2 "not supported")
#   error       exit 1 with a message on stderr
#   verifyfail  correct output with verify_ok=false, exit 3
#   badjson     broken JSON on stdout
#   contract    valid JSON that violates the contract (missing field, wrong rate)
#   hang        never finishes (for the timeout test)

behavior=$1
shift
duration="" precision=fp64 ops=1000 check=0

while (($#)); do
    case $1 in
        --check) check=1; shift ;;
        --duration | --precision | --ops_per_element)
            if (($# < 2)); then echo "$1 requires a value" >&2; exit 1; fi
            case $1 in
                --duration) duration=$2 ;;
                --precision) precision=$2 ;;
                --ops_per_element) ops=$2 ;;
            esac
            shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n $duration ]] || { echo "--duration is required" >&2; exit 1; }
awk -v d="$duration" 'BEGIN { exit !(d ~ /^[0-9]*\.?[0-9]+$/ && d + 0 > 0) }' ||
    { echo "--duration must be a number > 0" >&2; exit 1; }
case $precision in fp64 | fp32 | fp16 | int32) ;; *) echo "invalid --precision '$precision'" >&2; exit 1 ;; esac
[[ $ops =~ ^[1-9][0-9]*$ ]] || { echo "--ops_per_element must be an integer >= 1" >&2; exit 1; }
((check)) && exit 0

case $behavior in
    error) echo "hipMalloc failed: out of memory" >&2; exit 1 ;;
    hang) exec sleep 100000 ;;
esac
if [[ $precision == fp16 ]]; then
    echo "precision fp16 is not supported on this device" >&2
    exit 2
fi
if [[ $behavior == badjson ]]; then
    echo '{"runtime_s": 1.0, "iterations": '
    exit 0
fi

echo "mock benchmark ($behavior), precision $precision, duration $duration s" >&2
t0=$(date +%s%N)
sleep "$duration"
t1=$(date +%s%N)

verify=true rc=0
[[ $behavior == verifyfail ]] && verify=false rc=3

# monotonic-like timestamps (realtime modulo 1e15 ns, exact in bash integers)
m0=$((t0 % 1000000000000000)) m1=$((t1 % 1000000000000000))
awk -v t0="$t0" -v t1="$t1" -v m0="$m0" -v m1="$m1" -v ops="$ops" -v prec="$precision" -v verify="$verify" -v behavior="$behavior" 'BEGIN {
    rt = (t1 - t0) / 1e9
    it = int(rt / 0.0005); if (it < 1) it = 1
    gflop = it * ops * 1e-6; gbyte = it * 1e-3
    gflops = gflop / rt
    if (behavior == "contract") gflops *= 2          # wrong rate
    printf "{\"runtime_s\": %.9f, \"iterations\": %d, ", rt, it
    printf "\"roi_start_unix\": %.6f, \"roi_end_unix\": %.6f, ", t0 / 1e9, t1 / 1e9
    printf "\"roi_start_mono_ns\": %s, \"roi_end_mono_ns\": %s, ", m0, m1
    if (behavior != "contract") printf "\"e_pci_bus_id\": \"0000:01:00.0\", "
    printf "\"a_gflop\": %.9g, \"a_gbyte\": %.9g, \"a_intensity\": %.9g, ", gflop, gbyte, gflop / gbyte
    printf "\"d_gflop_per_s\": %.9g, \"d_gbyte_per_s\": %.9g, ", gflops, gbyte / rt
    printf "\"verify_ok\": %s, \"precision\": \"%s\", \"e_blocks\": 1824}\n", verify, prec
}'
exit $rc
