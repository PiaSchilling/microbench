#!/usr/bin/env bash
# build.sh - build the benchmarks (contract section 2)
#
# Usage: build.sh [--component <file>] [--src-dir <dir>] [--bin-dir <dir>] [<name> ...]
#
# Builds src/<name>.cpp -> build/<name> for all benchmarks, or only the ones given.
# Compiler: $HIPCC (default: hipcc). Architecture from the component config.
# The metadata of each benchmark is validated before it is built.

set -uo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SUITE_DIR/lib/common.sh"

COMPONENT=$SUITE_DIR/config/component.cfg
SRC_DIR=$SUITE_DIR/src
BIN_DIR=$SUITE_DIR/build
HIPCC=${HIPCC:-hipcc}
COMMON_FLAGS=(-O3 -std=c++17 -ffp-contract=fast)
NAMES=()

while (($#)); do
    case $1 in
        --component) COMPONENT=${2:?--component requires a value}; shift 2 ;;
        --src-dir)   SRC_DIR=${2:?--src-dir requires a value}; shift 2 ;;
        --bin-dir)   BIN_DIR=${2:?--bin-dir requires a value}; shift 2 ;;
        --help | -h)
            sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*) die "unknown option '$1'" ;;
        *)  NAMES+=("$1"); shift ;;
    esac
done

command -v "$HIPCC" >/dev/null 2>&1 || die "compiler '$HIPCC' not found (set HIPCC or load the ROCm module)"

comp_load "$COMPONENT"
report_errors "build aborted" || exit 1
ARCH=${COMP[arch]}

if ((${#NAMES[@]} == 0)); then
    shopt -s nullglob
    for f in "$SRC_DIR"/*.cpp; do NAMES+=("$(basename -- "$f" .cpp)"); done
    shopt -u nullglob
    ((${#NAMES[@]})) || die "no benchmarks found in $SRC_DIR"
fi

mkdir -p "$BIN_DIR" || die "cannot create '$BIN_DIR'"

failed=()
built=0
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

info "built $built of ${#NAMES[@]} benchmark(s) into $BIN_DIR"
if ((${#failed[@]})); then
    err "failed: ${failed[*]}"
    exit 1
fi
