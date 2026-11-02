# row.awk - overview rows for the terminal (aligned) and for the TSV files
#
# Usage:
#   awk -v kind=run|agg -v header=1 -v tsv=<file> -v lw=<label width> -v bw=<bench width> \
#       -f json.awk -f row.awk             (prints the header)
#   awk -v kind=run|agg -v tsv=<file> -v lw=.. -v bw=.. -f json.awk -f row.awk <jsonl>
#
# The terminal row goes to stdout, the TSV row is appended to the file <tsv>.

BEGIN {
    if (kind == "agg")
        ncols = split("label n_ok n_total runtime_s_mean d_gflop_per_s_mean d_gflop_per_s_cv " \
                      "d_gbyte_per_s_mean d_gbyte_per_s_cv m_avg_power_w_mean", cols, " ")
    else
        ncols = split("label rep benchmark device status runtime_s d_gflop_per_s d_gbyte_per_s " \
                      "m_avg_power_w", cols, " ")
    if (header) {
        line = ""
        for (c = 1; c <= ncols; c++) {
            line = line (c > 1 ? "\t" : "") cols[c]
            printf "%s", cell(cols[c], cols[c])
        }
        printf "\n"
        print line > tsv
        exit
    }
}

# terminal cell with fixed width
function cell(k, v) {
    if (k == "label")     return sprintf("%-" lw "s  ", v)
    if (k == "benchmark") return sprintf("%-" bw "s  ", v)
    if (k == "status")    return sprintf("%-8s  ", v)
    if (k == "rep" || k == "device" || k ~ /^n_/) return sprintf("%-" (length(k) > 3 ? length(k) : 3) "s  ", v)
    return sprintf("%14s  ", v)
}

# human formatting of a number
function fmt(k, v,    a) {
    if (k ~ /_cv$/) return sprintf("%.2f%%", v * 100)
    a = (v < 0 ? -v : v)
    if (v == int(v) && a < 1e12) return sprintf("%d", v)
    if (a >= 1e6 || (a > 0 && a < 1e-3)) return sprintf("%.4e", v)
    if (a >= 100) return sprintf("%.1f", v)
    return sprintf("%.4g", v)
}

{
    if ($0 ~ /^[ \t]*$/) next
    if (!jparse($0)) { print "row.awk: " J_err > "/dev/stderr"; next }
    line = ""
    for (c = 1; c <= ncols; c++) {
        k = cols[c]; raw = ""; shown = "-"
        if (k in J_idx) {
            i = J_idx[k]
            if (J_type[i] == "s") { raw = jdecode(J_val[i]); shown = raw }
            else if (J_type[i] == "n") { raw = J_val[i]; shown = fmt(k, J_val[i] + 0) }
            else if (J_type[i] == "b") { raw = J_val[i]; shown = raw }
        }
        gsub(/[\t\n]/, " ", raw)
        line = line (c > 1 ? "\t" : "") raw
        printf "%s", cell(k, shown)
    }
    printf "\n"
    print line >> tsv
}
