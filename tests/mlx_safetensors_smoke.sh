#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAP="${QW36_MLX_MODEL:-/Users/bytedance/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46}"
SHARD="$SNAP/model-00001-of-00004.safetensors"

if [ ! -f "$SHARD" ]; then
  echo "[mlx-safetensors] skip: $SHARD not found"
  exit 0
fi

TMP_C="${TMPDIR:-/tmp}/qw36_mlx_safetensors_smoke.c"
TMP_BIN="${TMPDIR:-/tmp}/qw36_mlx_safetensors_smoke"

cat >"$TMP_C" <<'C'
#include "qw36_safetensors.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int expect_tensor(const qw36_safetensors_file *f, const char *name,
                         const char *dtype, uint32_t n_dims,
                         const uint64_t *dims) {
    qw36_safetensors_tensor t;
    if (qw36_safetensors_get_tensor(f, name, &t)) {
        fprintf(stderr, "missing tensor: %s\n", name);
        return 1;
    }
    if (!t.data || !t.dtype_name || t.n_dims != n_dims) {
        fprintf(stderr, "bad metadata: %s\n", name);
        return 1;
    }
    if (dtype && strcmp(t.dtype_name, dtype) != 0) {
        fprintf(stderr, "bad dtype: %s got %s want %s\n",
                name, t.dtype_name, dtype);
        return 1;
    }
    for (uint32_t i = 0; i < n_dims; i++) {
        if (t.dims[i] != dims[i]) {
            fprintf(stderr, "bad dim[%u]: %s got %llu want %llu\n",
                    i, name, (unsigned long long)t.dims[i],
                    (unsigned long long)dims[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    char err[256];
    if (argc != 2) {
        fprintf(stderr, "usage: %s <shard.safetensors>\n", argv[0]);
        return 2;
    }
    qw36_safetensors_file *f = qw36_safetensors_open(argv[1], err, sizeof(err));
    if (!f) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    if (qw36_safetensors_tensor_count(f) < 700) {
        fprintf(stderr, "too few tensors: %zu\n",
                qw36_safetensors_tensor_count(f));
        qw36_safetensors_close(f);
        return 1;
    }
    const uint64_t embed[2] = {248320, 256};
    const uint64_t moe_w[3] = {256, 512, 256};
    const uint64_t moe_s[3] = {256, 512, 32};
    int rc = 0;
    rc |= expect_tensor(f, "language_model.model.embed_tokens.weight",
                        "U32", 2, embed);
    rc |= expect_tensor(f, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                        "U32", 3, moe_w);
    rc |= expect_tensor(f, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
                        "BF16", 3, moe_s);
    qw36_safetensors_close(f);
    return rc;
}
C

cc -O2 -std=c11 -Wall -Wextra -I"$ROOT/common" \
  "$TMP_C" "$ROOT/common/qw36_safetensors.c" -o "$TMP_BIN"
"$TMP_BIN" "$SHARD"
echo "[mlx-safetensors] ok: parsed $(basename "$SHARD")"
