#!/usr/bin/env bash
# Metal-only 35B MoE validation/bench helper for Qwen3.6-35B-A3B.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${QW36_35B_MODEL:-/Users/bytedance/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
N="${N:-8}"
SEQ="${SEQ:-256}"
REPEAT="${REPEAT:-1}"
PROMPT="${PROMPT:-Benchmark:}"
PROFILE="${PROFILE:-fast}"
BIN="./qw36_metal"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "skip: Metal backend requires Darwin"
    exit 0
fi

if [[ ! -f "$MODEL" ]]; then
    echo "skip: 35B model not found at $MODEL"
    exit 0
fi

if [[ ! -x "$BIN" ]]; then
    make -s -C metal
fi

profile_args=()
case "$PROFILE" in
    fast) profile_args=(--fast) ;;
    reference|fp16|lowmem) profile_args=(--profile "$PROFILE") ;;
    none|"") profile_args=() ;;
    *)
        echo "error: PROFILE must be fast, reference, fp16, lowmem, or none"
        exit 2
        ;;
esac

echo "metal_35b_moe_perf model=$MODEL"
echo "metal_35b_moe_perf n=$N seq=$SEQ repeat=$REPEAT profile=${PROFILE:-none} perf=${QW36_METAL_PERF:-0}"

tmp=""
cleanup() {
    if [[ -n "$tmp" && -f "$tmp" ]]; then
        rm -f "$tmp"
    fi
}
trap cleanup EXIT

for ((run = 1; run <= REPEAT; run++)); do
    tmp="$(mktemp "${TMPDIR:-/tmp}/qw36_35b_moe_perf.XXXXXX")"
    echo "metal_35b_moe_perf run=$run start"

    set +e
    "$BIN" "${profile_args[@]}" -m "$MODEL" --no-special -p "$PROMPT" -n "$N" --seq "$SEQ" >"$tmp" 2>&1
    status=$?
    set -e

    if [[ "$status" -ne 0 ]]; then
        echo "metal_35b_moe_perf run=$run status=$status"
        sed -n '1,160p' "$tmp"
        exit "$status"
    fi

    generated_line="$(grep -E "qw36: generated [0-9]+ tokens in [0-9.]+s \\([0-9.]+ tok/s\\)" "$tmp" | tail -1 || true)"
    if [[ -n "$generated_line" ]]; then
        echo "$generated_line"
        tokens="$(printf '%s\n' "$generated_line" | sed -nE 's/.*generated ([0-9]+) tokens.*/\1/p')"
        seconds="$(printf '%s\n' "$generated_line" | sed -nE 's/.*tokens in ([0-9.]+)s.*/\1/p')"
        tok_s="$(printf '%s\n' "$generated_line" | sed -nE 's/.*\(([0-9.]+) tok\/s\).*/\1/p')"
        echo "metal_35b_moe_perf result run=$run tokens=$tokens seconds=$seconds tok_s=$tok_s"
    else
        echo "metal_35b_moe_perf result run=$run missing_generated_line=1"
    fi

    grep -E "^(qw36: prefill|\\[metal-perf\\])|affine32 repacked|qk repacked|Q6K" "$tmp" || true
    rm -f "$tmp"
    tmp=""
done
