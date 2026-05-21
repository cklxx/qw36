#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${QW36_35B_MODEL:-/Users/bytedance/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
AUDIT="${QW36_AUDIT_TENSOR_F16:-$ROOT/qw36_audit_tensor_f16}"

if [[ ! -f "$MODEL" ]]; then
  echo "skip: 35B model not found at $MODEL"
  exit 0
fi

if [[ ! -x "$AUDIT" || "$ROOT/tools/audit_tensor_f16.c" -nt "$AUDIT" ||
      "$ROOT/common/qw36_gguf.c" -nt "$AUDIT" ||
      "$ROOT/common/qw36_dequant.c" -nt "$AUDIT" ]]; then
  cc -O2 -std=c11 -I"$ROOT/common" \
    "$ROOT/tools/audit_tensor_f16.c" \
    "$ROOT/common/qw36_gguf.c" "$ROOT/common/qw36_dequant.c" \
    -lm -o "$AUDIT"
fi

"$AUDIT" "$MODEL" output.weight 248320
