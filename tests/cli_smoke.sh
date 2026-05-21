#!/usr/bin/env bash
# tests/cli_smoke.sh — exercises --doctor / --print-config / --help / --info
# without needing a model on disk. The model-dependent paths short-circuit
# politely and the script stays green on a fresh checkout.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick the freshest binary available. CPU is always there.
BIN=./qw36_cpu
if [ -x ./qw36_metal ] && [ "$(uname)" = "Darwin" ]; then
    BIN=./qw36_metal
fi

echo "[cli-smoke] using $BIN"

# 1. --help short-circuits (returns 1 from parse_args by convention)
set +e
$BIN --help 2>&1 | head -1 > /dev/null
set -e
echo "[cli-smoke] --help: OK"

# 2. --doctor (no model, no profile) must exit 0 with WARN
out=$($BIN --doctor 2>&1)
echo "$out" | grep -q "Result: 0 FAIL"
echo "$out" | grep -q "WARN.*model path"
echo "[cli-smoke] --doctor no-model: OK"

# 3. --doctor with bad model must exit 1 and FAIL
set +e
out=$($BIN --doctor -m /tmp/qw36_doctor_does_not_exist.gguf 2>&1)
rc=$?
set -e
[ "$rc" = "1" ] || { echo "[cli-smoke] expected exit 1 for bad model, got $rc"; exit 1; }
echo "$out" | grep -q "FAIL.*model file"
echo "[cli-smoke] --doctor bad-model: OK"

# 4. --doctor with QW36_PROFILE=garbage must FAIL
set +e
out=$(QW36_PROFILE=garbage $BIN --doctor 2>&1)
rc=$?
set -e
[ "$rc" = "1" ] || { echo "[cli-smoke] expected exit 1 for bad profile, got $rc"; exit 1; }
echo "$out" | grep -q "FAIL.*QW36_PROFILE"
echo "[cli-smoke] --doctor bad-profile: OK"

# 5. --print-config (no args) shows the defaults and exits 0
out=$($BIN --print-config 2>&1)
echo "$out" | grep -q "qw36 effective config"
echo "$out" | grep -q "QW36_PROFILE.*stable"
echo "$out" | grep -q "QW36_METAL_KV_TRANSPOSED"
echo "[cli-smoke] --print-config defaults: OK"

# 6. --print-config --fast shows the profile flags set
out=$($BIN --fast --print-config 2>&1)
echo "$out" | grep -q "profile.*= fast"
echo "$out" | grep -E "QW36_METAL_QUANT_GPU.*1" >/dev/null
echo "[cli-smoke] --print-config --fast: OK"

# 7. --doctor with --fast + a real model (if QW36_TEST_MODEL set) → 0 FAIL 0 WARN
if [ -n "${QW36_TEST_MODEL:-}" ] && [ -f "$QW36_TEST_MODEL" ]; then
    out=$($BIN --fast --doctor -m "$QW36_TEST_MODEL" 2>&1)
    echo "$out" | grep -q "all clear"
    echo "[cli-smoke] --doctor --fast real-model: OK"
fi

echo "[cli-smoke] all CLI checks passed"
