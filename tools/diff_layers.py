#!/usr/bin/env python3
"""Diff qw36's --layer-trace output against an MLX golden JSON dump.

Run after capturing both sides:

    python3 tools/mlx_dump_intermediates.py --layer 3 --out golden.json
    ./qw36_cpu -m model.gguf -p Hello --layer-trace 3 --layer-trace-out qw.json
    python3 tools/diff_layers.py golden.json qw.json --rtol 1e-2

For each common tensor key, prints max-abs-error and the index of the worst
mismatch. The first tensor that exceeds --rtol is the bug site.
"""
import argparse, json, sys
import math


def load_dump(path):
    with open(path) as f:
        d = json.load(f)
    if "captured" in d:
        return d["captured"]
    return d


def compare_tensor(name, a, b, rtol):
    av = a["first_n"]
    bv = b["first_n"]
    n = min(len(av), len(bv))
    if n == 0:
        return ("EMPTY", 0.0, -1)
    worst = 0.0
    worst_i = 0
    for i in range(n):
        diff = abs(av[i] - bv[i])
        denom = max(abs(av[i]), abs(bv[i]), 1e-6)
        rel = diff / denom
        if rel > worst:
            worst = rel
            worst_i = i
    status = "OK " if worst < rtol else "BAD"
    return (status, worst, worst_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("qw36")
    ap.add_argument("--rtol", type=float, default=5e-2)
    args = ap.parse_args()

    g = load_dump(args.golden)
    q = load_dump(args.qw36)

    common = [k for k in g if k in q]
    only_g = [k for k in g if k not in q]
    only_q = [k for k in q if k not in g]

    print(f"# diff {args.golden} vs {args.qw36}  rtol={args.rtol}")
    if only_g:
        print(f"# in golden only: {only_g}")
    if only_q:
        print(f"# in qw36 only:   {only_q}")
    print()

    first_bad = None
    for k in common:
        status, worst, idx = compare_tensor(k, g[k], q[k], args.rtol)
        shape_g = g[k].get("shape", "?")
        shape_q = q[k].get("shape", "?")
        shape_warn = "" if shape_g == shape_q else f"  SHAPE MISMATCH {shape_g} vs {shape_q}"
        marker = "  ← FIRST DIVERGENCE" if status == "BAD" and first_bad is None else ""
        if status == "BAD" and first_bad is None:
            first_bad = k
        print(f"  [{status}] {k:32s} max_rel={worst:.3e} at i={idx}{shape_warn}{marker}")

    if first_bad:
        print()
        print(f"First diverging tensor: {first_bad}")
        g_first = g[first_bad]["first_n"][:8]
        q_first = q[first_bad]["first_n"][:8]
        print(f"  golden first 8: {[f'{v:+.4f}' for v in g_first]}")
        print(f"  qw36   first 8: {[f'{v:+.4f}' for v in q_first]}")
        sys.exit(1)
    else:
        print()
        print("All tensors within rtol — no divergence detected.")


if __name__ == "__main__":
    main()
