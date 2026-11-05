# counters.awk - cut the counter output of rocprofv3 to the ROI and aggregate it
#
# Usage:
#   awk -v t0=<roi start ns> -v t1=<roi end ns> -v wmean=<c1,c2,...> -v prefix=<csv prefix> \
#       -v iterations=<n> -f counters.awk <dir>/<host>/<pid>_counter_collection.csv
#
# Input: the CSV of rocprofv3 (one row per kernel dispatch and counter; Kernel_Name
#        contains commas inside quotes, so the rows are parsed as real CSV).
# Window: only dispatches that lie completely inside [t0, t1] (monotonic clock, ns).
#
# Output:
#   META <dispatches> <kernel_time_s>       dispatches inside the ROI, their summed duration
#   ROW <csv row>                           one per counter:
#       prefix,"<counter>",<value>,"<agg>",<dispatches>,<iterations>,<kernel_time_s>,<roi_s>
#     value: sum over the dispatches (agg "sum"), or for the counters in `wmean` the
#            mean weighted by dispatch duration (ratios such as MeanOccupancyPerCU)
#   ERR <message>                           missing columns

BEGIN {
    n = split(wmean, w, ",")
    for (i = 1; i <= n; i++) if (w[i] != "") is_wmean[w[i]] = 1
    t0 += 0; t1 += 0
}

# split a CSV line into f[1..n] (RFC 4180: quoted fields, "" inside quotes); returns n
function csvsplit(line, f,    n, i, c, q, cur, len) {
    n = 0; cur = ""; q = 0; len = length(line)
    for (i = 1; i <= len; i++) {
        c = substr(line, i, 1)
        if (q) {
            if (c == "\"") {
                if (substr(line, i + 1, 1) == "\"") { cur = cur "\""; i++ }
                else q = 0
            } else cur = cur c
        } else if (c == "\"") q = 1
        else if (c == ",") { f[++n] = cur; cur = "" }
        else cur = cur c
    }
    f[++n] = cur
    return n
}

{ sub(/\r$/, "") }

NR == 1 {
    nh = csvsplit($0, h)
    for (i = 1; i <= nh; i++) col[h[i]] = i
    split("Dispatch_Id Counter_Name Counter_Value Start_Timestamp End_Timestamp", need, " ")
    for (i = 1; i <= 5; i++) if (!(need[i] in col)) { print "ERR column " need[i] " missing in the rocprofv3 output"; bad = 1 }
    next
}

bad || $0 == "" { next }

{
    csvsplit($0, f)
    s = f[col["Start_Timestamp"]] + 0
    e = f[col["End_Timestamp"]] + 0
    if (s < t0 || e > t1) next                       # dispatch not completely inside the ROI
    d = f[col["Dispatch_Id"]]
    c = f[col["Counter_Name"]]
    v = f[col["Counter_Value"]] + 0
    if (!(d in dur)) { dur[d] = e - s; ndisp++; ktime += e - s }
    if (!(c in sum)) { order[++nc] = c; sum[c] = 0; wsum[c] = 0; wtot[c] = 0 }
    sum[c] += v
    wsum[c] += v * (e - s); wtot[c] += (e - s)
}

END {
    if (bad) exit
    printf "META %d %.9f\n", ndisp, ktime / 1e9
    for (k = 1; k <= nc; k++) {
        c = order[k]
        if (c in is_wmean) { val = (wtot[c] > 0) ? wsum[c] / wtot[c] : ""; agg = "wmean" }
        else { val = sum[c]; agg = "sum" }
        printf "ROW %s,\"%s\",%s,\"%s\",%d,%s,%.9f,%.9f\n", prefix, c, (val == "" ? "" : sprintf("%.15g", val)),
               agg, ndisp, iterations, ktime / 1e9, (t1 - t0) / 1e9
    }
}
