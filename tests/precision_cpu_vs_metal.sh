#!/usr/bin/env bash
# precision_cpu_vs_metal.sh — assert CPU and Metal builds produce identical
# top-K logits on a fixed (model, prompt). Used as a CI invariant after any
# change to forward, materialize, dequant, or backend dispatch.
#
# Usage:
#   tests/precision_cpu_vs_metal.sh [model.gguf]
# Exits 0 if logits match, non-zero otherwise.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
PROMPT='Hello'
N=3
TOP=5

if [ ! -f "$MODEL" ]; then
    echo "skip: model not found at $MODEL"
    exit 0
fi

if [ ! -x ./qw36_cpu ];   then make -s cpu;   fi
if [ ! -x ./qw36_metal ]; then make -s metal; fi

extract_logits() {
    grep -E "^\s+[0-9]+\.[0-9]+\s+id=" | awk '{print $1,$2}'
}

CPU_OUT=$(./qw36_cpu   -m "$MODEL" -p "$PROMPT" -n "$N" --no-special --debug-top "$TOP" 2>&1 | extract_logits)
MTL_OUT=$(./qw36_metal -m "$MODEL" -p "$PROMPT" -n "$N" --no-special --debug-top "$TOP" 2>&1 | extract_logits)

if [ "$CPU_OUT" = "$MTL_OUT" ]; then
    n_lines=$(echo "$CPU_OUT" | wc -l | tr -d ' ')
    echo "ok: CPU and Metal logits agree across $n_lines lines (prompt='$PROMPT', n=$N, top=$TOP)"
    exit 0
fi

echo "FAIL: CPU and Metal logits diverge."
echo "--- CPU ---"; echo "$CPU_OUT" | head -20
echo "--- METAL ---"; echo "$MTL_OUT" | head -20
echo "--- diff ---"
diff <(echo "$CPU_OUT") <(echo "$MTL_OUT") | head -30
exit 1
