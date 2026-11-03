#!/usr/bin/env bash
# build.sh - build the benchmarks (contract section 2)
#
# Usage: build.sh [--component <file>] [--src-dir <dir>] [--bin-dir <dir>] [<name> ...]
#
# Builds src/<name>.cpp -> build/<name> for all benchmarks and tools/sampler.c ->
# build/sampler, or only the names given ("sampler" for the sampler).
# Benchmarks: compiler $HIPCC (default: hipcc), architecture from the component config;
# the metadata of each benchmark is validated before it is built.
# Sampler: compiler $CC (default: cc), amdsmi from $ROCM_PATH (default: /opt/rocm).

set -uo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SUITE_DIR/lib/common.sh"

COMPONENT=$SUITE_DIR/config/component.cfg
SRC_DIR=$SUITE_DIR/src
BIN_DIR=$SUITE_DIR/build
HIPCC=${HIPCC:-hipcc}
CC=${CC:-cc}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
SAMPLER_SRC=$SUITE_DIR/tools/sampler.c
COMMON_FLAGS=(-O3 -std=c++17 -ffp-contract=fast)
NAMES=()

while (($#)); do
    case $1 in
        --component) COMPONENT=${2:?--component requires a value}; shift 2 ;;
        --src-dir)   SRC_DIR=${2:?--src-dir requires a value}; shift 2 ;;
        --bin-dir)   BIN_DIR=${2:?--bin-dir requires a value}; shift 2 ;;
        --help | -h)
            sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*) die "unknown option '$1'" ;;
        *)  NAMES+=("$1"); shift ;;
    esac
done

# build_sampler: tools/sampler.c -> build/sampler (C, against the amdsmi library of ROCm)
build_sampler() {
    if [[ ! -r $ROCM_PATH/include/amd_smi/amdsmi.h ]]; then
        err "amdsmi.h not found in $ROCM_PATH/include/amd_smi (set ROCM_PATH or load the ROCm module)"
        return 1
    fi
    command -v "$CC" >/dev/null 2>&1 || { err "C compiler '$CC' not found (set CC)"; return 1; }
    local cmd=("$CC" -O2 -std=c11 -Wall -Wextra "-I$ROCM_PATH/include" "$SAMPLER_SRC"
               "-L$ROCM_PATH/lib" "-Wl,-rpath,$ROCM_PATH/lib" -lamd_smi -o "$BIN_DIR/sampler")
    info "[build] ${cmd[*]}"
    "${cmd[@]}" || { err "build of sampler failed"; return 1; }
}

WITH_SAMPLER=0
if ((${#NAMES[@]} == 0)); then
    shopt -s nullglob
    for f in "$SRC_DIR"/*.cpp; do NAMES+=("$(basename -- "$f" .cpp)"); done
    shopt -u nullglob
    ((${#NAMES[@]})) || die "no benchmarks found in $SRC_DIR"
    [[ -r $SAMPLER_SRC ]] && WITH_SAMPLER=1
else
    rest=()
    for name in "${NAMES[@]}"; do
        if [[ $name == sampler ]]; then WITH_SAMPLER=1; else rest+=("$name"); fi
    done
    NAMES=("${rest[@]}")
fi

mkdir -p "$BIN_DIR" || die "cannot create '$BIN_DIR'"

failed=()
built=0
if ((${#NAMES[@]})); then
    command -v "$HIPCC" >/dev/null 2>&1 || die "compiler '$HIPCC' not found (set HIPCC or load the ROCm module)"
    comp_load "$COMPONENT"
    report_errors "build aborted" || exit 1
    ARCH=${COMP[arch]}
fi

for name in "${NAMES[@]}"; do
    SUITE_ERRORS=()
    if ! meta_load "$name" "$SRC_DIR"; then
        report_errors "skipping $name"
        failed+=("$name")
        continue
    fi
    read -r -a extra <<<"$(meta_get "$name" extra_build_flags)"
    cmd=("$HIPCC" "${COMMON_FLAGS[@]}" "--offload-arch=$ARCH" "${extra[@]}" \
         "$SRC_DIR/$name.cpp" -o "$BIN_DIR/$name")
    info "[build] ${cmd[*]}"
    if "${cmd[@]}"; then
        built=$((built + 1))
    else
        err "build of $name failed"
        failed+=("$name")
    fi
done

if ((WITH_SAMPLER)); then
    if build_sampler; then built=$((built + 1)); else failed+=(sampler); fi
fi

info "built $built of $((${#NAMES[@]} + WITH_SAMPLER)) target(s) into $BIN_DIR"
if ((${#failed[@]})); then
    err "failed: ${failed[*]}"
    exit 1
fi
