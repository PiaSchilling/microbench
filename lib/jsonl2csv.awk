# jsonl2csv.awk - convert JSON Lines (flat objects) to CSV
#
# Usage:  awk -f json.awk -f jsonl2csv.awk results.jsonl > results.csv
#
# Columns are the union of all keys, in order of first appearance (the wrapper
# writes its own fields first, so they come first). Missing values are empty.
# Strings are always quoted (RFC 4180), numbers and booleans are written as is,
# null becomes an empty cell.

{
    if ($0 ~ /^[ \t]*$/) next
    if (!jparse($0)) { print "jsonl2csv: line " NR " skipped: " J_err > "/dev/stderr"; next }
    nrec++
    for (i = 1; i <= J_n; i++) {
        k = J_key[i]
        if (!(k in col)) { ncol++; col[k] = ncol; name[ncol] = k }
        cell[nrec, col[k]] = csvval(J_type[i], J_val[i])
    }
}

function csvval(t, v) {
    if (t == "z") return ""
    if (t != "s") return v
    v = jdecode(v)
    gsub(/"/, "\"\"", v)
    return "\"" v "\""
}

END {
    if (ncol == 0) exit
    for (c = 1; c <= ncol; c++) printf "%s%s", (c > 1 ? "," : ""), name[c]
    printf "\n"
    for (r = 1; r <= nrec; r++) {
        for (c = 1; c <= ncol; c++) printf "%s%s", (c > 1 ? "," : ""), ((r, c) in cell ? cell[r, c] : "")
        printf "\n"
    }
}
