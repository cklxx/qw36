#!/usr/bin/env bash
# precision_cpu_vs_metal.sh — assert the *first* decode step's top-K
# logits are bit-equal between the CPU reference and Metal builds.
#
# We only check step 0 because at step 0 the KV cache and DeltaNet
# state are both zero, so reduction order in the GPU softmax/scan
# can't accumulate divergence from the CPU's double-accumulator
# matmul. From step 1 onward sub-1e-3 logit drift is expected and
# accepted (different reduction order across simd lanes vs sequential
# CPU sum) — this is the standard fp32 GPU↔CPU contract.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
PROMPT='Hello'
N=1
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

# Force fp32 weights on Metal for the byte-equal comparison. The default
# Metal path is fp16-eligible (opt-in via QW36_METAL_FP16_WEIGHTS=1 in
# the engine open) which introduces sub-1e-3 drift.
CPU_OUT=$(QW36_METAL_FP16_WEIGHTS=0 ./qw36_cpu   -m "$MODEL" -p "$PROMPT" -n "$N" --no-special --debug-top "$TOP" 2>&1 | extract_logits)
MTL_OUT=$(QW36_METAL_FP16_WEIGHTS=0 ./qw36_metal -m "$MODEL" -p "$PROMPT" -n "$N" --no-special --debug-top "$TOP" 2>&1 | extract_logits)

if [ "$CPU_OUT" = "$MTL_OUT" ]; then
    n=$(echo "$CPU_OUT" | wc -l | tr -d ' ')
    echo "ok: CPU and Metal logits agree across $n lines (step 0 only, prompt='$PROMPT')"
    exit 0
fi

echo "FAIL: CPU and Metal step-0 logits diverge."
echo "--- CPU ---"; echo "$CPU_OUT"
echo "--- METAL ---"; echo "$MTL_OUT"
exit 1
