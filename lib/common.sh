# common.sh - shared helpers for run_suite.sh and build.sh
#
# Sourced, not executed. Requires bash >= 4.3 (associative arrays, namerefs).
#
# Provides:
#   logging      info / notice / warn / err / die, mirrored to $LOG_FILE once set
#   errors       add_error / report_errors (collect all validation errors, report once)
#   key=value    kv_parse: parser for component config, counter config, metadata
#   metadata     meta_load: extracts and validates the metadata block of src/<name>.cpp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE=""          # set by the wrapper as soon as the result directory exists
SUITE_ERRORS=()      # validation errors collected by add_error

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

# log <text>: write to the log file only (no terminal output)
log() {
    [[ -n $LOG_FILE && -n $* ]] && printf '[%s] %s\n' "$(_ts)" "$*" >>"$LOG_FILE"
    return 0
}

# All terminal messages go to stderr; stdout is reserved for the result table.
info()   { printf '%s\n' "$*" >&2;          log "INFO   $*"; }
notice() { printf 'notice: %s\n' "$*" >&2;  log "NOTICE $*"; }
warn()   { printf 'warning: %s\n' "$*" >&2; log "WARN   $*"; }
err()    { printf 'error: %s\n' "$*" >&2;   log "ERROR  $*"; }
die()    { err "$*"; exit 1; }

add_error() { SUITE_ERRORS+=("$*"); }

# report_errors <what>: print all collected errors; returns 1 if there were any
report_errors() {
    local e
    ((${#SUITE_ERRORS[@]} == 0)) && return 0
    for e in "${SUITE_ERRORS[@]}"; do err "$e"; done
    err "${#SUITE_ERRORS[@]} error(s) found; $1"
    return 1
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# trim <var-name>: remove leading and trailing whitespace in place
trim() {
    local -n __t=$1
    __t="${__t#"${__t%%[![:space:]]*}"}"
    __t="${__t%"${__t##*[![:space:]]}"}"
}

is_uint()   { [[ $1 =~ ^[0-9]+$ ]]; }
is_number() { [[ $1 =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$ ]]; }   # non-negative

# in_list <item> <comma-separated list>
in_list() { [[ ",$2," == *",$1,"* ]]; }

# require_env: minimal check of the tools the suite relies on
require_env() {
    local t missing=()
    if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
        die "bash >= 4.3 required (found $BASH_VERSION)"
    fi
    for t in awk sed sort timeout date mkdir cp; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done
    ((${#missing[@]})) && die "required tools not found: ${missing[*]}"
    [[ $(date +%N) =~ ^[0-9]{9}$ ]] || die "'date +%N' does not provide nanoseconds"
    return 0
}

# ---------------------------------------------------------------------------
# key=value files
# ---------------------------------------------------------------------------

# kv_parse <assoc-name> <order-array-name> <context>   (reads stdin)
#   Line-based key=value; '#' starts a comment line; empty lines are ignored.
#   Leading/trailing whitespace around key and value is removed.
#   Keys: [A-Za-z0-9_.]+, each key at most once. Errors go to SUITE_ERRORS.
#   The file is parsed, never executed.
kv_parse() {
    local -n __kv=$1
    local -n __ord=$2
    local ctx=$3 line key val n=0
    while IFS= read -r line || [[ -n $line ]]; do
        n=$((n + 1))
        line=${line%$'\r'}
        trim line
        [[ -z $line || $line == \#* ]] && continue
        if [[ $line != *=* ]]; then
            add_error "$ctx:$n: expected 'key=value', got '$line'"
            continue
        fi
        key=${line%%=*}
        val=${line#*=}
        trim key
        trim val
        if [[ ! $key =~ ^[A-Za-z0-9_.]+$ ]]; then
            add_error "$ctx:$n: invalid key '$key'"
            continue
        fi
        if [[ -v __kv[$key] ]]; then
            add_error "$ctx:$n: duplicate key '$key'"
            continue
        fi
        __kv[$key]=$val
        __ord+=("$key")
    done
}

# ---------------------------------------------------------------------------
# Component config
# ---------------------------------------------------------------------------

declare -gA COMP=()
declare -ga COMP_ORDER=()

# comp_load <file>: parse and validate the component config into COMP
comp_load() {
    local file=$1 k
    if [[ ! -r $file ]]; then
        add_error "component config: cannot read '$file'"
        return 1
    fi
    kv_parse COMP COMP_ORDER "$file" <"$file"
    for k in arch gpus precisions; do
        [[ -n ${COMP[$k]:-} ]] || add_error "$file: required key '$k' missing or empty"
    done
    [[ -z ${COMP[arch]:-} || ${COMP[arch]} =~ ^gfx[0-9a-f]+$ ]] ||
        add_error "$file: arch '${COMP[arch]}' is not a gfx target (e.g. gfx942)"
    [[ -z ${COMP[gpus]:-} || ${COMP[gpus]} =~ ^[1-9][0-9]*$ ]] ||
        add_error "$file: gpus must be a positive integer"
    [[ -z ${COMP[precisions]:-} || ${COMP[precisions]} =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] ||
        add_error "$file: precisions must be a comma-separated list (e.g. fp64,fp32)"
    for k in "${COMP_ORDER[@]}"; do
        case $k in
            arch | gpus | precisions) ;;
            lds_bytes_per_cu | l1_bytes_per_cu | l2_bytes_per_xcd | llc_bytes | hbm_bytes)
                is_uint "${COMP[$k]}" || add_error "$file: $k must be an integer (bytes)" ;;
            *) add_error "$file: unknown key '$k'" ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Benchmark metadata (block at the top of src/<name>.cpp)
# ---------------------------------------------------------------------------

META_START='// ---- benchmark metadata ----'
META_END='// ---- end metadata ----'

declare -gA META_LOADED=()   # name -> 1 (valid) | 0 (invalid)

# meta_extract <file>: print the metadata block content (comment markers removed).
# Exit status: 0 ok, 2 no start marker in line 1, 3 non-comment line in block, 4 no end marker
meta_extract() {
    awk -v START="$META_START" -v STOP="$META_END" '
        { sub(/\r$/, "") }
        NR == 1 { if ($0 != START) { code = 2; exit } next }
        $0 == STOP { found = 1; exit }
        substr($0, 1, 2) != "//" { code = 3; exit }
        { s = substr($0, 3); sub(/^ /, "", s); print s }
        END { if (code) exit code; if (!found) exit 4; exit 0 }
    ' "$1"
}

# meta_load <name> <src-dir>
#   Loads and validates the metadata of benchmark <name> into the arrays
#   META_<name> (assoc) and METAORD_<name> (key order). Errors go to SUITE_ERRORS.
#   Returns 0 if the metadata is valid.
meta_load() {
    local name=$1 srcdir=$2 file block rc k v nerr
    [[ -n ${META_LOADED[$name]:-} ]] && { [[ ${META_LOADED[$name]} == 1 ]]; return; }
    nerr=${#SUITE_ERRORS[@]}
    file="$srcdir/$name.cpp"

    if [[ ! $name =~ ^[A-Za-z0-9_]+$ ]]; then
        add_error "benchmark name '$name' is invalid (allowed: A-Z a-z 0-9 _)"
        META_LOADED[$name]=0; return 1
    fi
    if [[ ! -r $file ]]; then
        add_error "benchmark '$name': source '$file' not found"
        META_LOADED[$name]=0; return 1
    fi

    declare -gA "META_$name=()"
    declare -ga "METAORD_$name=()"
    local -n m="META_$name"

    block=$(meta_extract "$file"); rc=$?
    case $rc in
        0) ;;
        2) add_error "$file: line 1 must be '$META_START'" ;;
        3) add_error "$file: every line of the metadata block must start with '//'" ;;
        *) add_error "$file: end marker '$META_END' not found" ;;
    esac
    if ((rc == 0)); then
        kv_parse "META_$name" "METAORD_$name" "$file (metadata)" <<<"$block"
        local -n ord="METAORD_$name"

        for k in benchmark level component description tags version; do
            [[ -n ${m[$k]:-} ]] || add_error "$file: required metadata key '$k' missing or empty"
        done
        [[ -z ${m[benchmark]:-} || ${m[benchmark]} == "$name" ]] ||
            add_error "$file: benchmark=${m[benchmark]} does not match the file name '$name'"
        [[ -z ${m[level]:-} || ${m[level]} =~ ^[123]$ ]] ||
            add_error "$file: level must be 1, 2 or 3"
        [[ -z ${m[version]:-} || ${m[version]} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
            add_error "$file: version must be MAJOR.MINOR.PATCH"
        [[ -z ${m[precisions]:-} || ${m[precisions]} =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] ||
            add_error "$file: precisions must be a comma-separated list"

        for k in "${ord[@]}"; do
            v=${m[$k]}
            case $k in
                benchmark | level | component | description | tags | version | precisions | \
                    extra_build_flags | expect_counters | expect_isa) ;;
                expect_isa.*)
                    [[ ${k#expect_isa.} =~ ^[a-z0-9]+$ ]] || add_error "$file: invalid key '$k'" ;;
                param.*)
                    [[ ${k#param.} =~ ^[a-z_][a-z0-9_]*$ ]] ||
                        add_error "$file: invalid parameter name in '$k'"
                    [[ $v =~ ^[^|]*\|[^|]*\|[^|]*$ ]] ||
                        add_error "$file: $k must have the form <default>|<valid values>|<effect>" ;;
                *) add_error "$file: unknown metadata key '$k'" ;;
            esac
        done
        [[ -v m[param.duration] ]] || add_error "$file: param.duration missing (--duration is mandatory)"
    fi

    if ((${#SUITE_ERRORS[@]} > nerr)); then META_LOADED[$name]=0; return 1; fi
    META_LOADED[$name]=1
    return 0
}

# meta_get <name> <key>: print a metadata value (empty if missing)
meta_get() {
    local -n __m="META_$1"
    printf '%s' "${__m[$2]:-}"
}
