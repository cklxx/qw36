#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${QW36_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
DUMP="${QW36_DUMP_TENSOR:-$ROOT/qw36_dump_tensor}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

if [[ ! -x "$DUMP" || "$ROOT/tools/dump_tensor.c" -nt "$DUMP" || "$ROOT/common/qw36_gguf.c" -nt "$DUMP" ]]; then
  cc -O2 -std=c11 -I"$ROOT/common" \
    "$ROOT/tools/dump_tensor.c" "$ROOT/common/qw36_gguf.c" \
    -lm -o "$DUMP"
fi

compare_dump() {
  local tensor="$1"
  local expected_csv="$2"
  local required="$3"
  local out err
  out="$(mktemp)"
  err="$(mktemp)"
  trap 'rm -f "$out" "$err"' RETURN

  if ! "$DUMP" "$MODEL" "$tensor" 16 >"$out" 2>"$err"; then
    if grep -q "tensor not found" "$err" && [[ "$required" == "optional" ]]; then
      echo "skip: $tensor not present in this model"
      return 0
    fi
    cat "$err" >&2
    return 1
  fi

  if [[ -z "$expected_csv" ]]; then
    echo "missing hard-coded golden values for existing tensor: $tensor" >&2
    return 1
  fi

  python3 - "$tensor" "$expected_csv" "$out" <<'PY'
import math
import re
import sys

tensor, expected_csv, path = sys.argv[1:4]
expected = [float(x) for x in expected_csv.split(",")]
actual = []
line_re = re.compile(r"\[\s*\d+\]\s+([-+0-9.eE]+)")
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        m = line_re.search(line)
        if m:
            actual.append(float(m.group(1)))

if len(actual) != len(expected):
    print(f"{tensor}: expected {len(expected)} values, got {len(actual)}", file=sys.stderr)
    sys.exit(1)

for i, (a, e) in enumerate(zip(actual, expected)):
    if not math.isclose(a, e, rel_tol=0.0, abs_tol=5e-6):
        print(f"{tensor}: mismatch at {i}: got {a:.9g}, expected {e:.9g}", file=sys.stderr)
        sys.exit(1)

print(f"ok: {tensor} row0 first {len(expected)} fp32 values match")
PY
}

ATTN_Q_ROW0_16="-0.002163,-0.002163,0.006924,-0.006707,-0.002163,0.011468,-0.002163,0.006924,0.006924,0.006924,-0.020338,0.029643,-0.002163,0.011468,0.006924,0.002381"

compare_dump "blk.3.attn_q.weight" "$ATTN_Q_ROW0_16" "required"

# Layer 3 is vanilla on Qwen3.5-0.8B-Q4_K_M, but keep the optional DN probe
# so the test catches the same kernel path on fixtures where blk.3 is DeltaNet.
compare_dump "blk.3.attn_qkv.weight" "${QW36_EXPECTED_L3_ATTN_QKV_ROW0_16:-}" "optional"
