#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${QW36_35B_MODEL:-/Users/bytedance/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
DUMP="${QW36_DUMP_TENSOR:-$ROOT/qw36_dump_tensor}"

if [[ ! -f "$MODEL" ]]; then
  echo "skip: 35B model not found at $MODEL"
  exit 0
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import gguf
import numpy
PY
then
  echo "skip: python gguf/numpy modules not available"
  exit 0
fi

if [[ ! -x "$DUMP" || "$ROOT/tools/dump_tensor.c" -nt "$DUMP" ||
      "$ROOT/common/qw36_gguf.c" -nt "$DUMP" ||
      "$ROOT/common/qw36_dequant.c" -nt "$DUMP" ]]; then
  cc -O2 -std=c11 -I"$ROOT/common" \
    "$ROOT/tools/dump_tensor.c" \
    "$ROOT/common/qw36_gguf.c" "$ROOT/common/qw36_dequant.c" \
    -lm -o "$DUMP"
fi

python3 - "$MODEL" "$DUMP" <<'PY'
import math
import re
import subprocess
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, dequantize

model, dump = sys.argv[1:3]
reader = GGUFReader(model)
tensors = {t.name: t for t in reader.tensors}

required = ["token_embd.weight", "output.weight"]
for name in required:
    if name not in tensors:
        raise SystemExit(f"missing tensor: {name}")
    if tensors[name].tensor_type != GGMLQuantizationType.Q8_0:
        raise SystemExit(f"{name}: expected Q8_0, got {tensors[name].tensor_type}")

out_rows = int(tensors["output.weight"].shape[1])
probes = [
    ("token_embd.weight", 9419),
    ("output.weight", 0),
    ("output.weight", out_rows - 1),
]

line_re = re.compile(r"\[\s*\d+\]\s+([-+0-9.eE]+)")

for name, row in probes:
    t = tensors[name]
    ref = dequantize(t.data[row:row + 1], t.tensor_type)[0].astype(np.float32)
    got_text = subprocess.check_output(
        [dump, model, name, str(ref.shape[0]), str(row)],
        text=True,
    )
    got = np.array([float(m.group(1)) for m in line_re.finditer(got_text)],
                   dtype=np.float32)
    if got.shape != ref.shape:
        raise SystemExit(f"{name} row {row}: got {got.shape}, expected {ref.shape}")
    diff = np.abs(got - ref)
    max_abs = float(diff.max()) if diff.size else math.inf
    if max_abs > 2.0e-6:
        idx = int(diff.argmax())
        raise SystemExit(
            f"{name} row {row}: mismatch at {idx}: "
            f"got {got[idx]:.9g}, expected {ref[idx]:.9g}, max_abs={max_abs:.3g}"
        )
    print(f"ok: {name} row {row} matches gguf-py Q8_0 reference (max_abs={max_abs:.3g})")
PY
