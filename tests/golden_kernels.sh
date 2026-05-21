#!/usr/bin/env bash
# tests/golden_kernels.sh — generate + verify per-kernel golden fixtures
# end-to-end. v0 covers rmsnorm / silu / swiglu using the CPU primitives
# both sides; cross-backend checks land in golden_kernels_metal.sh /
# golden_kernels_cuda.sh on top of the same fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p tests/goldens

CFLAGS="-O2 -std=c11 -Wall -Wextra"

echo "[golden] compile gen_goldens + check_goldens"
cc $CFLAGS -Icommon tools/gen_goldens.c   -o qw36_gen_goldens   -lm
cc $CFLAGS -Icommon tools/check_goldens.c -o qw36_check_goldens -lm

echo "[golden] (re)generate fixtures"
./qw36_gen_goldens tests/goldens

for k in rmsnorm silu swiglu; do
    fix="tests/goldens/${k}.bin"
    if [ ! -f "$fix" ]; then
        echo "[golden] FAIL: missing fixture $fix"; exit 1
    fi
    echo "[golden] check $k"
    ./qw36_check_goldens "$fix"
done

echo "[golden] all kernels match within tol"
