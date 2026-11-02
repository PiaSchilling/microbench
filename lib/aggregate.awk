# aggregate.awk - statistics over repetitions (contract Section 10, "Aggregation")
#
# Usage:  awk -f json.awk -f aggregate.awk results.jsonl > aggregates.jsonl
#
# One output line per label (in order of first appearance). Only repetitions
# with status "ok" are included.
#   d_*, m_*, runtime_s       -> <f>_mean, <f>_std, <f>_min, <f>_max, <f>_cv
#   e_*, a_intensity, other   -> must be identical in all repetitions; copied once
#   benchmark fields
#   a_* (other), iterations,  -> not aggregated
#   roi_*, per-run wrapper fields
# Warnings go to stderr.

BEGIN {
    n = split("label benchmark benchmark_version mode device", tmp, " ")
    for (i = 1; i <= n; i++) { idfield[tmp[i]] = 1; idorder[i] = tmp[i] }
    nid = n
    n = split("rep status reason exit_code cmd iterations", tmp, " ")
    for (i = 1; i <= n; i++) drop[tmp[i]] = 1
}

{
    if ($0 ~ /^[ \t]*$/) next
    if (!jparse($0)) { print "aggregate: line " NR " skipped: " J_err > "/dev/stderr"; next }
    if (!("label" in J_idx)) next
    lab = jdecode(J_val[J_idx["label"]])
    if (!(lab in ntotal)) {
        nlab++; labels[nlab] = lab; ntotal[lab] = 0; nok[lab] = 0
        for (i = 1; i <= nid; i++)
            if (idorder[i] in J_idx) id[lab, idorder[i]] = J_val[J_idx[idorder[i]]]
    }
    ntotal[lab]++
    if (!("status" in J_idx) || jdecode(J_val[J_idx["status"]]) != "ok") next
    nok[lab]++

    for (i = 1; i <= J_n; i++) {
        k = J_key[i]; t = J_type[i]; v = J_val[i]
        if ((k in idfield) || (k in drop) || k ~ /^roi_/ || (k ~ /^a_/ && k != "a_intensity")) continue

        if ((k ~ /^(d_|m_)/ || k == "runtime_s") && t == "n") {
            if (!((lab, k) in cnt)) { nstat[lab]++; statkey[lab, nstat[lab]] = k; cnt[lab, k] = 0 }
            cnt[lab, k]++
            vals[lab, k, cnt[lab, k]] = v + 0
            continue
        }

        if (!((lab, k) in cval)) {
            nconst[lab]++; constkey[lab, nconst[lab]] = k; cval[lab, k] = v; ctype[lab, k] = t
        } else if (!((lab, k) in differs) && !same(t, v, ctype[lab, k], cval[lab, k])) {
            differs[lab, k] = 1
            print "label '" lab "': field '" k "' differs between repetitions (" cval[lab, k] " vs " v ")" > "/dev/stderr"
        }
    }
}

function same(t1, v1, t2, v2) {
    if (t1 != t2) return 0
    if (t1 == "n") return (v1 + 0) == (v2 + 0)
    return v1 == v2
}

function num(x) { return sprintf("%.10g", x) }

END {
    for (l = 1; l <= nlab; l++) {
        lab = labels[l]; out = ""
        for (i = 1; i <= nid; i++)
            if ((lab, idorder[i]) in id) out = out (out == "" ? "" : ",") jstr(idorder[i]) ":" id[lab, idorder[i]]
        out = out ",\"n_total\":" ntotal[lab] ",\"n_ok\":" nok[lab]

        for (j = 1; j <= nconst[lab]; j++) {
            k = constkey[lab, j]
            out = out "," jstr(k) ":" (((lab, k) in differs) ? "null" : cval[lab, k])
        }

        for (j = 1; j <= nstat[lab]; j++) {
            k = statkey[lab, j]; n = cnt[lab, k]
            s = 0; mn = vals[lab, k, 1]; mx = mn
            for (r = 1; r <= n; r++) {
                x = vals[lab, k, r]; s += x
                if (x < mn) mn = x
                if (x > mx) mx = x
            }
            mean = s / n
            if (n >= 2) {
                ss = 0
                for (r = 1; r <= n; r++) ss += (vals[lab, k, r] - mean) ^ 2
                std = sqrt(ss / (n - 1))
                sstd = num(std)
                scv = (mean != 0 ? num(std / mean) : "null")
            } else { sstd = "null"; scv = "null" }
            out = out "," jstr(k "_mean") ":" num(mean) "," jstr(k "_std") ":" sstd \
                      "," jstr(k "_min") ":" num(mn) "," jstr(k "_max") ":" num(mx) \
                      "," jstr(k "_cv") ":" scv
        }
        print "{" out "}"
    }
}
