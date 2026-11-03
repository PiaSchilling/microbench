# telemetry.awk - cut sampler data (telemetry.csv) to a time window and summarize it
#
# Usage:
#   awk -F, -v t0=<ns> -v t1=<ns> [-v timecol=mono_ns|unix_ns] [-v bdf=<pci address>]
#       [-v header=<header line>] [-v cut=<file>] -f json.awk -f telemetry.awk <telemetry rows>
#
#   t0, t1    window [t0, t1] in ns of the column timecol (default mono_ns)
#   bdf       only this GPU (default: every GPU found, one output line each)
#   header    header line of telemetry.csv, needed if the input is a tail of the file
#             without header (a header line in the input is recognized and skipped)
#   cut       write the raw rows of <bdf> that cover the window to this file
#
# Output: one flat JSON object per GPU: "bdf" and the m_* fields (contract section 13).
#   Counters (energy, throttling residencies) are interpolated linearly at t0 and t1,
#   so the result does not depend on where the samples happen to lie:
#     m_energy_j              E(t1) - E(t0)
#     m_avg_power_w           m_energy_j / (t1 - t0)
#     m_ppt_viol_pct          100 * delta ppt_residency_acc / delta acc_counter (power limit)
#     m_thm_viol_pct          same with thm_residency_acc (thermal limit)
#   Instantaneous values are averaged over the samples inside the window:
#     m_telemetry_samples, m_avg_power_sampled_w, m_avg_gfx_clk_mhz, m_min_gfx_clk_mhz,
#     m_max_gfx_clk_mhz, m_avg_mem_clk_mhz, m_max_temp_c, m_max_temp_mem_c,
#     m_avg_gfx_util_pct, m_avg_mem_util_pct, m_max_read_ms
#   Values that cannot be computed are null.

BEGIN {
    if (timecol == "") timecol = "mono_ns"
    t0 += 0; t1 += 0
    if (header != "") set_header(header)
    nb = 0
}

function set_header(h,    n, i, f) {
    n = split(h, f, ",")
    for (i = 1; i <= n; i++) col[f[i]] = i
    hdr = h
    have_header = 1
}

function val(name) { return (name in col) ? $(col[name]) : "" }

# remember, for window boundary k (0: t0, 1: t1), the last sample at or before it and
# the first sample at or after it, separately for every counter c
function track(b, c, tm, v,    k, t) {
    if (v == "") return
    for (k = 0; k <= 1; k++) {
        t = k ? t1 : t0
        if (tm <= t) { bt[b, c, k] = tm; bv[b, c, k] = v + 0 }
        if (tm >= t && !((b, c, k) in at)) { at[b, c, k] = tm; av[b, c, k] = v + 0 }
    }
}

# counter value at boundary k, linearly interpolated; "" if not enclosed by samples
function at_boundary(b, c, k,    t) {
    if (!((b, c, k) in bt) || !((b, c, k) in at)) return ""
    t = k ? t1 : t0
    if (at[b, c, k] == bt[b, c, k]) return bv[b, c, k]
    return bv[b, c, k] + (av[b, c, k] - bv[b, c, k]) * (t - bt[b, c, k]) / (at[b, c, k] - bt[b, c, k])
}

function delta(b, c,    a, z) {
    a = at_boundary(b, c, 0); z = at_boundary(b, c, 1)
    return (a == "" || z == "") ? "" : z - a
}

# running statistics of instantaneous values inside the window
function stat(b, c, v) {
    if (v == "") return
    v += 0
    sum[b, c] += v; cnt[b, c]++
    if (!((b, c) in mn) || v < mn[b, c]) mn[b, c] = v
    if (!((b, c) in mx) || v > mx[b, c]) mx[b, c] = v
}
function avg(b, c)  { return ((b, c) in cnt) ? sum[b, c] / cnt[b, c] : "" }
function vmin(b, c) { return ((b, c) in mn) ? mn[b, c] : "" }
function vmax(b, c) { return ((b, c) in mx) ? mx[b, c] : "" }

function num(x) { return x == "" ? "null" : sprintf("%.10g", x) }

{
    sub(/\r$/, "")
    if ($1 == "mono_ns") { if (!have_header) set_header($0); next }
    if (!have_header || NF < 3) next
    b = val("bdf")
    if (bdf != "" && b != bdf) next
    tm = val(timecol) + 0
    if (!(b in seen)) { seen[b] = 1; order[++ngpu] = b }

    track(b, "energy_j", tm, val("energy_j"))
    track(b, "acc_counter", tm, val("acc_counter"))
    track(b, "ppt_residency_acc", tm, val("ppt_residency_acc"))
    track(b, "thm_residency_acc", tm, val("thm_residency_acc"))

    if (tm >= t0 && tm <= t1) {
        n_in[b]++
        stat(b, "power_w", val("power_w"))
        stat(b, "gfx_clk_mhz", val("gfx_clk_mhz"))
        stat(b, "gfx_clk_min_mhz", val("gfx_clk_min_mhz"))
        stat(b, "gfx_clk_max_mhz", val("gfx_clk_max_mhz"))
        stat(b, "mem_clk_mhz", val("mem_clk_mhz"))
        stat(b, "temp_hotspot_c", val("temp_hotspot_c"))
        stat(b, "temp_mem_c", val("temp_mem_c"))
        stat(b, "gfx_util_pct", val("gfx_util_pct"))
        stat(b, "mem_util_pct", val("mem_util_pct"))
        stat(b, "read_ns", val("read_ns"))
    }
    if (cut != "") { nrow++; row[nrow] = $0; rowt[nrow] = tm }
}

END {
    if (t1 <= t0) exit
    for (g = 1; g <= ngpu; g++) {
        b = order[g]
        e = delta(b, "energy_j")
        acc = delta(b, "acc_counter")
        ppt = delta(b, "ppt_residency_acc")
        thm = delta(b, "thm_residency_acc")
        out = "{" jstr("bdf") ":" jstr(b)
        out = out ",\"m_energy_j\":" num(e)
        out = out ",\"m_avg_power_w\":" num(e == "" ? "" : e / ((t1 - t0) / 1e9))
        out = out ",\"m_ppt_viol_pct\":" num(ppt == "" || acc == "" || acc <= 0 ? "" : 100 * ppt / acc)
        out = out ",\"m_thm_viol_pct\":" num(thm == "" || acc == "" || acc <= 0 ? "" : 100 * thm / acc)
        out = out ",\"m_telemetry_samples\":" (n_in[b] + 0)
        out = out ",\"m_avg_power_sampled_w\":" num(avg(b, "power_w"))
        out = out ",\"m_avg_gfx_clk_mhz\":" num(avg(b, "gfx_clk_mhz"))
        out = out ",\"m_min_gfx_clk_mhz\":" num(vmin(b, "gfx_clk_min_mhz"))
        out = out ",\"m_max_gfx_clk_mhz\":" num(vmax(b, "gfx_clk_max_mhz"))
        out = out ",\"m_avg_mem_clk_mhz\":" num(avg(b, "mem_clk_mhz"))
        out = out ",\"m_max_temp_c\":" num(vmax(b, "temp_hotspot_c"))
        out = out ",\"m_max_temp_mem_c\":" num(vmax(b, "temp_mem_c"))
        out = out ",\"m_avg_gfx_util_pct\":" num(avg(b, "gfx_util_pct"))
        out = out ",\"m_avg_mem_util_pct\":" num(avg(b, "mem_util_pct"))
        out = out ",\"m_max_read_ms\":" num(vmax(b, "read_ns") == "" ? "" : vmax(b, "read_ns") / 1e6)
        print out "}"
    }

    # raw rows covering the window: from the last sample before t0 to the first after t1
    if (cut != "" && bdf != "" && ngpu) {
        lo = ((bdf, "energy_j", 0) in bt) ? bt[bdf, "energy_j", 0] : t0
        hi = ((bdf, "energy_j", 1) in at) ? at[bdf, "energy_j", 1] : t1
        print hdr > cut
        for (r = 1; r <= nrow; r++) if (rowt[r] >= lo && rowt[r] <= hi) print row[r] > cut
        close(cut)
    }
}
