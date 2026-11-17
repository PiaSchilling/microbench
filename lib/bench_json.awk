# bench_json.awk - check a benchmark's stdout against the contract (Section 5/6)
#
# Usage:  awk -v duration=<s> -f json.awk -f bench_json.awk stdout.json
#
# Output (one item per line):
#   STATUS ok|error
#   REASON <text>                  (all contract violations, separated by "; ")
#   VERIFY true|false|null
#   ROI <start_mono_ns> <end_mono_ns>   (only if both are valid; raw text, no rounding)
#   BDF <pci address>                   (only if e_pci_bus_id is valid)
#   PERF <runtime_s> <iterations>       (only if both are valid; for m_prof_slowdown_p<k>)
#   FRAGMENT <json members>        benchmark fields as  "k":v,"k":v  (no braces);
#                                  fields with reserved or invalid names are left out

{
    sub(/\r$/, "")
    if ($0 ~ /^[ \t]*$/) next
    nl++
    line = $0
}

function violation(m) { reason = (reason == "" ? m : reason "; " m) }

function num(k) { return J_val[J_idx[k]] + 0 }

# relative difference check: |a-b| <= tol * max(|a|,|b|) (+ absolute floor)
function close_to(a, b, rtol, atol,    d, m) {
    d = a - b; if (d < 0) d = -d
    m = (a < 0 ? -a : a); if ((b < 0 ? -b : b) > m) m = (b < 0 ? -b : b)
    return d <= rtol * m + atol
}

END {
    verify = "null"; fragment = ""
    if (nl == 0) violation("no JSON line on stdout")
    else if (nl > 1) violation("expected exactly one line on stdout, got " nl)
    else if (!jparse(line)) violation("invalid JSON: " J_err)
    else check()

    print "STATUS " (reason == "" ? "ok" : "error")
    print "REASON " reason
    print "VERIFY " verify
    if (roi != "") print "ROI " roi
    if (bdf != "") print "BDF " bdf
    if (perf != "") print "PERF " perf
    print "FRAGMENT " fragment
}

function check(    i, k, n, req, rt, ws) {
    # field names
    n = split("label rep benchmark benchmark_version mode device status reason exit_code cmd", req, " ")
    for (i = 1; i <= n; i++)
        if (req[i] in J_idx) { violation("field '" req[i] "' is reserved for the wrapper"); drop[req[i]] = 1 }
    for (i = 1; i <= J_n; i++) {
        k = J_key[i]
        if (k !~ /^[A-Za-z_][A-Za-z0-9_]*$/) { violation("invalid field name '" k "'"); drop[k] = 1 }
        else if (k ~ /^m_/) { violation("prefix m_ is reserved for the wrapper ('" k "')"); drop[k] = 1 }
    }

    # required numeric fields
    n = split("runtime_s iterations roi_start_unix roi_end_unix roi_start_mono_ns roi_end_mono_ns " \
              "a_gflop a_gbyte a_intensity d_gflop_per_s d_gbyte_per_s", req, " ")
    for (i = 1; i <= n; i++) {
        if (!(req[i] in J_idx)) { violation("required field '" req[i] "' missing"); bad = 1 }
        else if (J_type[J_idx[req[i]]] != "n") { violation("field '" req[i] "' must be a number"); bad = 1 }
    }
    if (!("e_pci_bus_id" in J_idx)) violation("required field 'e_pci_bus_id' missing")
    else if (J_type[J_idx["e_pci_bus_id"]] != "s" ||
             jdecode(J_val[J_idx["e_pci_bus_id"]]) !~ /^[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]\.[0-7]$/)
        violation("e_pci_bus_id must be a PCI address like \"0000:42:00.0\"")
    else bdf = tolower(jdecode(J_val[J_idx["e_pci_bus_id"]]))
    if (!("verify_ok" in J_idx)) violation("required field 'verify_ok' missing")
    else if (J_type[J_idx["verify_ok"]] != "b" && J_type[J_idx["verify_ok"]] != "z")
        violation("verify_ok must be true, false or null")
    else verify = J_val[J_idx["verify_ok"]]

    # plausibility (only if all numeric fields are present)
    if (!bad) {
        rt = num("runtime_s")
        if (num("iterations") < 1 || num("iterations") != int(num("iterations")))
            violation("iterations must be an integer >= 1")
        else if (rt > 0) perf = J_val[J_idx["runtime_s"]] " " J_val[J_idx["iterations"]]
        if (rt <= 0) violation("runtime_s must be > 0")
        else if (rt < duration * 0.999)
            violation("runtime_s (" rt ") is shorter than the requested duration (" duration ")")
        if (num("roi_end_mono_ns") <= num("roi_start_mono_ns"))
            violation("roi_end_mono_ns must be greater than roi_start_mono_ns")
        else if (!close_to((num("roi_end_mono_ns") - num("roi_start_mono_ns")) / 1e9, rt, 0.01, 1e-3))
            violation("ROI length in monotonic clock does not match runtime_s")
        else roi = J_val[J_idx["roi_start_mono_ns"]] " " J_val[J_idx["roi_end_mono_ns"]]
        if (num("roi_end_unix") <= num("roi_start_unix"))
            violation("roi_end_unix must be greater than roi_start_unix")
        else if (!close_to(num("roi_end_unix") - num("roi_start_unix"), rt, 0.01, 1e-2))
            violation("ROI length in Unix time does not match runtime_s")
        if (num("a_gflop") < 0 || num("a_gbyte") < 0) violation("a_gflop and a_gbyte must be >= 0")
        if (rt > 0 && !close_to(num("d_gflop_per_s"), num("a_gflop") / rt, 0.01, 1e-9))
            violation("d_gflop_per_s does not equal a_gflop / runtime_s")
        if (rt > 0 && !close_to(num("d_gbyte_per_s"), num("a_gbyte") / rt, 0.01, 1e-9))
            violation("d_gbyte_per_s does not equal a_gbyte / runtime_s")
        if (num("a_gbyte") > 0 && !close_to(num("a_intensity"), num("a_gflop") / num("a_gbyte"), 0.01, 1e-9))
            violation("a_intensity does not equal a_gflop / a_gbyte")
    }

    # fields with reserved or invalid names are left out, so the record stays valid
    for (i = 1; i <= J_n; i++)
        if (!(J_key[i] in drop))
            fragment = fragment (fragment == "" ? "" : ",") jstr(J_key[i]) ":" J_val[i]
}
