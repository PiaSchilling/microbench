/*
 * sampler.c - telemetry sampler of the benchmark suite (contract section 13)
 *
 * Reads the GPU metrics table of every AMD GPU at a fixed interval and writes
 * one CSV line per GPU and sample. Started by the wrapper in the background,
 * stopped with SIGTERM.
 *
 * Usage:
 *   sampler [--interval-ms <n>] [--out <file>]  -> sample until SIGTERM/SIGINT
 *   sampler --probe <seconds> [--interval-ms <n>] -> measure how often the values change
 *
 * Build (see build.sh):
 *   cc -O2 -std=c11 -I$ROCM_PATH/include tools/sampler.c -L$ROCM_PATH/lib \
 *      -Wl,-rpath,$ROCM_PATH/lib -lamd_smi -o build/sampler
 *
 * Exit codes: 0 stopped normally, 1 error (arguments, amdsmi, no GPU, output file)
 *
 * Written against amdsmi.h of ROCm 7.2.4 (amdsmi 26.2.2, metrics table v1.x).
 */

#define _GNU_SOURCE
#include <amd_smi/amdsmi.h>

#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_GPUS 64

/* energy_accumulator counts in units of 15.259 uJ (amdsmi.h: "Energy (15.259uJ ...)") */
#define ENERGY_UNIT_J 15.259e-6

/* ------------------------------------------------------------------------- */
/* Devices                                                                    */
/* ------------------------------------------------------------------------- */

typedef struct {
    amdsmi_processor_handle handle;
    char bdf[32];                 /* PCI address, e.g. "0000:42:00.0" */
} gpu_t;

static gpu_t gpus[MAX_GPUS];
static int ngpus = 0;

static const char *status_str(amdsmi_status_t s) {
    const char *msg = NULL;
    if (amdsmi_status_code_to_string(s, &msg) != AMDSMI_STATUS_SUCCESS || msg == NULL)
        return "unknown amdsmi error";
    return msg;
}

/* Find all AMD GPUs: sockets -> processors -> keep processors of type GPU. */
static int discover_gpus(void) {
    uint32_t nsock = 0;
    amdsmi_status_t s = amdsmi_get_socket_handles(&nsock, NULL);
    if (s != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "sampler: amdsmi_get_socket_handles: %s\n", status_str(s));
        return -1;
    }
    amdsmi_socket_handle *socks = calloc(nsock ? nsock : 1, sizeof *socks);
    if (!socks) return -1;
    s = amdsmi_get_socket_handles(&nsock, socks);
    if (s != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "sampler: amdsmi_get_socket_handles: %s\n", status_str(s));
        free(socks);
        return -1;
    }

    for (uint32_t i = 0; i < nsock; ++i) {
        uint32_t nproc = 0;
        if (amdsmi_get_processor_handles(socks[i], &nproc, NULL) != AMDSMI_STATUS_SUCCESS) continue;
        amdsmi_processor_handle *procs = calloc(nproc ? nproc : 1, sizeof *procs);
        if (!procs) continue;
        if (amdsmi_get_processor_handles(socks[i], &nproc, procs) == AMDSMI_STATUS_SUCCESS) {
            for (uint32_t j = 0; j < nproc && ngpus < MAX_GPUS; ++j) {
                processor_type_t type;
                amdsmi_bdf_t bdf;
                if (amdsmi_get_processor_type(procs[j], &type) != AMDSMI_STATUS_SUCCESS ||
                    type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                    continue;
                if (amdsmi_get_gpu_device_bdf(procs[j], &bdf) != AMDSMI_STATUS_SUCCESS) {
                    fprintf(stderr, "sampler: warning: cannot read the PCI address of a GPU, skipped\n");
                    continue;
                }
                char name[32];
                snprintf(name, sizeof name, "%04" PRIx64 ":%02x:%02x.%x",
                         (uint64_t)bdf.domain_number, (unsigned)bdf.bus_number,
                         (unsigned)bdf.device_number, (unsigned)bdf.function_number);
                /* In partition modes (e.g. CPX) one GPU may appear several times;
                 * keep one handle per PCI address. */
                int dup = 0;
                for (int k = 0; k < ngpus; ++k) dup |= !strcmp(gpus[k].bdf, name);
                if (dup) continue;
                gpus[ngpus].handle = procs[j];
                strcpy(gpus[ngpus].bdf, name);
                ++ngpus;
            }
        }
        free(procs);
    }
    free(socks);
    return ngpus;
}

/* ------------------------------------------------------------------------- */
/* Time                                                                       */
/* ------------------------------------------------------------------------- */

static int64_t now_ns(clockid_t clk) {
    struct timespec t;
    clock_gettime(clk, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* ------------------------------------------------------------------------- */
/* One sample of one GPU                                                      */
/* ------------------------------------------------------------------------- */

/* All accesses to the metrics table are in this section. If field names or
 * units differ in another ROCm version, only this section has to change. */

typedef struct {
    int64_t mono_ns, unix_ns, read_ns;
    int ok;                       /* 0 if reading the metrics table failed */
    amdsmi_gpu_metrics_t m;
} sample_t;

static void read_sample(const gpu_t *g, sample_t *s) {
    s->unix_ns = now_ns(CLOCK_REALTIME);
    s->mono_ns = now_ns(CLOCK_MONOTONIC);
    s->ok = amdsmi_get_gpu_metrics_info(g->handle, &s->m) == AMDSMI_STATUS_SUCCESS;
    s->read_ns = now_ns(CLOCK_MONOTONIC) - s->mono_ns;
}

/* The metrics table marks unsupported values with the maximum of the type. */
#define VALID16(v) ((v) != UINT16_MAX)
#define VALID64(v) ((v) != UINT64_MAX)

/* append helpers: write the value, or nothing (empty CSV cell) if not valid */
static char *put_u16(char *p, uint16_t v)  { return VALID16(v) ? p + sprintf(p, ",%u", (unsigned)v) : p + sprintf(p, ","); }
static char *put_u64(char *p, uint64_t v)  { return VALID64(v) ? p + sprintf(p, ",%" PRIu64, v) : p + sprintf(p, ","); }

/* gfx clock: one value per XCD (current_gfxclks); mean, min and max of the valid ones */
static char *put_gfx_clocks(char *p, const amdsmi_gpu_metrics_t *m) {
    unsigned n = 0, sum = 0, mn = UINT16_MAX, mx = 0;
    for (int i = 0; i < AMDSMI_MAX_NUM_GFX_CLKS; ++i) {
        uint16_t c = m->current_gfxclks[i];
        if (!VALID16(c) || c == 0) continue;
        ++n; sum += c;
        if (c < mn) mn = c;
        if (c > mx) mx = c;
    }
    if (n == 0) {                  /* older tables: only one gfx clock */
        p = put_u16(p, m->current_gfxclk);
        return p + sprintf(p, ",,");
    }
    return p + sprintf(p, ",%.1f,%u,%u", (double)sum / n, mn, mx);
}

static const char *CSV_HEADER =
    "mono_ns,unix_ns,bdf,read_ns,fw_timestamp,power_w,energy_j,"
    "gfx_clk_mhz,gfx_clk_min_mhz,gfx_clk_max_mhz,mem_clk_mhz,"
    "temp_hotspot_c,temp_mem_c,gfx_util_pct,mem_util_pct,"
    "acc_counter,ppt_residency_acc,thm_residency_acc\n";

static void format_sample(char *p, const gpu_t *g, const sample_t *s) {
    p += sprintf(p, "%" PRId64 ",%" PRId64 ",%s,%" PRId64, s->mono_ns, s->unix_ns, g->bdf, s->read_ns);
    if (!s->ok) {                  /* reading failed: timestamps only, all values empty */
        sprintf(p, ",,,,,,,,,,,,,,\n");
        return;
    }
    const amdsmi_gpu_metrics_t *m = &s->m;
    p = put_u64(p, m->firmware_timestamp);              /* 10 ns resolution, only compared */
    p = put_u16(p, m->current_socket_power);            /* W */
    if (VALID64(m->energy_accumulator))                 /* J since start of the GPU */
        p += sprintf(p, ",%.6f", (double)m->energy_accumulator * ENERGY_UNIT_J);
    else
        p += sprintf(p, ",");
    p = put_gfx_clocks(p, m);                           /* MHz: mean, min, max over XCDs */
    p = put_u16(p, m->current_uclk);                    /* MHz, memory clock */
    p = put_u16(p, m->temperature_hotspot);             /* C */
    p = put_u16(p, m->temperature_mem);                 /* C */
    p = put_u16(p, m->average_gfx_activity);            /* % */
    p = put_u16(p, m->average_umc_activity);            /* % */
    p = put_u64(p, m->accumulation_counter);            /* reference for the residencies */
    p = put_u64(p, m->ppt_residency_acc);               /* power limit (PVIOL) */
    p = put_u64(p, m->socket_thm_residency_acc);        /* thermal limit (TVIOL) */
    sprintf(p, "\n");
}

/* ------------------------------------------------------------------------- */
/* Main loop                                                                  */
/* ------------------------------------------------------------------------- */

static volatile sig_atomic_t stop_requested = 0;
static void on_signal(int sig) { (void)sig; stop_requested = 1; }

static int run_sampling(FILE *out, int64_t interval_ns) {
    static sample_t s;
    char line[1024];
    int64_t next = now_ns(CLOCK_MONOTONIC);
    long samples = 0, missed = 0;

    fputs(CSV_HEADER, out);
    while (!stop_requested) {
        for (int i = 0; i < ngpus; ++i) {
            read_sample(&gpus[i], &s);
            format_sample(line, &gpus[i], &s);
            fputs(line, out);
        }
        /* flush after every sample: the wrapper waits for the first line, and
         * nothing is lost if the run is aborted */
        fflush(out);
        ++samples;

        /* sleep until the next absolute point in time: the interval does not
         * drift by the time spent reading. Missed points are skipped. */
        next += interval_ns;
        int64_t now = now_ns(CLOCK_MONOTONIC);
        while (next <= now) { next += interval_ns; ++missed; }
        struct timespec ts = { next / 1000000000LL, next % 1000000000LL };
        while (!stop_requested &&
               clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    }
    fprintf(stderr, "sampler: stopped after %ld samples (%ld intervals missed)\n", samples, missed);
    return 0;
}

/* --probe: sample as fast as the interval allows and report how often the
 * values change, i.e. how often the firmware refreshes the metrics table. */
static int run_probe(double seconds, int64_t interval_ns) {
    static sample_t s;
    long n[MAX_GPUS] = {0}, chg_fw[MAX_GPUS] = {0}, chg_pw[MAX_GPUS] = {0}, chg_en[MAX_GPUS] = {0};
    int64_t read_sum[MAX_GPUS] = {0}, t_first[MAX_GPUS] = {0}, t_last[MAX_GPUS] = {0};
    uint64_t fw0[MAX_GPUS], fw1[MAX_GPUS], en0[MAX_GPUS], en1[MAX_GPUS];
    uint16_t pw_prev[MAX_GPUS];
    double pw_sum[MAX_GPUS] = {0};

    int64_t start = now_ns(CLOCK_MONOTONIC), end = start + (int64_t)(seconds * 1e9), next = start;
    while (!stop_requested && now_ns(CLOCK_MONOTONIC) < end) {
        for (int i = 0; i < ngpus; ++i) {
            read_sample(&gpus[i], &s);
            if (!s.ok) continue;
            const amdsmi_gpu_metrics_t *m = &s.m;
            if (n[i] == 0) {
                fw0[i] = m->firmware_timestamp; en0[i] = m->energy_accumulator; t_first[i] = s.mono_ns;
            } else {
                chg_fw[i] += m->firmware_timestamp != fw1[i];
                chg_en[i] += m->energy_accumulator != en1[i];
                chg_pw[i] += m->current_socket_power != pw_prev[i];
            }
            fw1[i] = m->firmware_timestamp; en1[i] = m->energy_accumulator; pw_prev[i] = m->current_socket_power;
            t_last[i] = s.mono_ns;
            pw_sum[i] += m->current_socket_power;
            read_sum[i] += s.read_ns;
            ++n[i];
        }
        next += interval_ns;
        struct timespec ts = { next / 1000000000LL, next % 1000000000LL };
        while (!stop_requested &&
               clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    }

    for (int i = 0; i < ngpus; ++i) {
        printf("%s\n", gpus[i].bdf);
        if (n[i] < 2) { printf("  too few successful samples (%ld)\n", n[i]); continue; }
        double span = (t_last[i] - t_first[i]) / 1e9;
        printf("  samples           %ld in %.2f s, mean read time %.3f ms\n",
               n[i], span, read_sum[i] / (double)n[i] / 1e6);
        printf("  fw_timestamp      changed %ld times -> refreshed every %.1f ms\n",
               chg_fw[i], chg_fw[i] ? span * 1e3 / chg_fw[i] : 0.0);
        printf("  power_w           changed %ld times\n", chg_pw[i]);
        printf("  energy_j          changed %ld times\n", chg_en[i]);
        if (VALID64(fw0[i]) && fw1[i] > fw0[i])
            printf("  firmware time     %.3f s elapsed per fw_timestamp (10 ns units) vs %.3f s wall time\n",
                   (fw1[i] - fw0[i]) * 1e-8, span);
        if (VALID64(en0[i]) && en1[i] >= en0[i] && span > 0)
            printf("  energy check      dE/dt = %.1f W vs mean power_w = %.1f W "
                   "(should be similar if the energy unit is right)\n",
                   (en1[i] - en0[i]) * ENERGY_UNIT_J / span, pw_sum[i] / n[i]);
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Arguments and setup                                                        */
/* ------------------------------------------------------------------------- */

static void usage(FILE *f) {
    fprintf(f,
        "Usage: sampler [--interval-ms <n>] [--out <file>]\n"
        "       sampler --probe <seconds> [--interval-ms <n>]\n"
        "  --interval-ms <n>   sampling interval in ms (default 100; with --probe 1)\n"
        "  --out <file>        CSV output (default: stdout)\n"
        "  --probe <seconds>   measure how often the metrics are refreshed, then exit\n");
}

int main(int argc, char **argv) {
    long interval_ms = -1;
    double probe = 0;
    const char *out_path = NULL;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout); return 0; }
        if (i + 1 >= argc) { fprintf(stderr, "sampler: option %s requires a value\n", a); return 1; }
        const char *v = argv[++i];
        char *end;
        if (!strcmp(a, "--interval-ms")) {
            interval_ms = strtol(v, &end, 10);
            if (*end || end == v || interval_ms < 1 || interval_ms > 60000) {
                fprintf(stderr, "sampler: --interval-ms must be an integer between 1 and 60000\n");
                return 1;
            }
        } else if (!strcmp(a, "--probe")) {
            probe = strtod(v, &end);
            if (*end || end == v || !(probe > 0)) {
                fprintf(stderr, "sampler: --probe must be a number of seconds > 0\n");
                return 1;
            }
        } else if (!strcmp(a, "--out")) {
            out_path = v;
        } else {
            fprintf(stderr, "sampler: unknown option %s\n", a);
            usage(stderr);
            return 1;
        }
    }
    if (interval_ms < 0) interval_ms = probe > 0 ? 1 : 100;

    /* SIGTERM/SIGINT end the loop; without SA_RESTART the sleep is interrupted */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    amdsmi_status_t s = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (s != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "sampler: amdsmi_init: %s\n", status_str(s));
        return 1;
    }
    if (discover_gpus() <= 0) {
        fprintf(stderr, "sampler: no AMD GPU found\n");
        amdsmi_shut_down();
        return 1;
    }
    for (int i = 0; i < ngpus; ++i) fprintf(stderr, "sampler: GPU %d: %s\n", i, gpus[i].bdf);

    int rc;
    if (probe > 0) {
        rc = run_probe(probe, interval_ms * 1000000LL);
    } else {
        FILE *out = stdout;
        if (out_path && !(out = fopen(out_path, "w"))) {
            fprintf(stderr, "sampler: cannot open %s: %s\n", out_path, strerror(errno));
            amdsmi_shut_down();
            return 1;
        }
        rc = run_sampling(out, interval_ms * 1000000LL);
        if (out != stdout) fclose(out);
    }
    amdsmi_shut_down();
    return rc;
}
