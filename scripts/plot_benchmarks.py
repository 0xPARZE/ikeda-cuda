#!/usr/bin/env python3
"""Scaling plot and speedup table from benchmarks/results/*.csv.

    python scripts/plot_benchmarks.py benchmarks/results/latest.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from common import FIGURES, SERIES, apply_style

# (impl, precision) -> legend label, colour slot, marker. The order and colours
# stay fixed, so a missing implementation never shifts the others.
IMPLS = [
    (("python_loop", "double"), "Python loop (coursework style)", 0, "o"),
    (("numpy", "double"), "NumPy, vectorised over B", 1, "s"),
    (("cpu_serial", "double"), "C++ 1 thread, FP64", 2, "^"),
    (("cpu_openmp", "double"), "C++ OpenMP, FP64", 3, "v"),
    (("cpu_openmp", "float"), "C++ OpenMP, FP32", 6, "<"),
    (("gpu", "double"), "CUDA, FP64", 4, "D"),
    (("gpu", "float"), "CUDA, FP32", 5, "P"),
]


def load(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_params", "iterations", "block_size", "threads"):
            r[k] = int(r[k])
        for k in ("kernel_ms", "transfer_ms", "total_ms", "giga_iter_per_s"):
            r[k] = float(r[k])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmarks/results/latest.csv")
    ap.add_argument("--out", default=str(FIGURES / "scaling.png"))
    ap.add_argument("--table", default="benchmarks/results/summary.md")
    args = ap.parse_args()
    rows = load(Path(args.csv))
    apply_style()

    scaling = defaultdict(dict)  # (impl, precision) -> {N: row}
    for r in rows:
        if r["section"] == "scaling":
            scaling[(r["impl"], r["precision"])][r["n_params"]] = r

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, slot, marker in IMPLS:
        if key not in scaling:
            continue
        ns = sorted(scaling[key])
        # Time the user waits for. For the GPU that is end to end, not just the kernel.
        ts = [scaling[key][n]["total_ms"] for n in ns]
        ax.loglog(ns, ts, marker=marker, markersize=6, color=SERIES[slot], label=label)
    ax.set_xlabel("Number of B values swept, $N$ (1000 round trips each)")
    ax.set_ylabel("Wall time per sweep (ms)")
    ax.set_title("Sweep time vs. problem size", loc="left")
    ax.legend(loc="upper left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")

    # Markdown summary. Speedups are against multithreaded C++ at the *same*
    # precision: FP32 GPU vs FP32 CPU, FP64 GPU vs FP64 CPU.
    lines = []
    n_big = max(scaling[("gpu", "float")]) if ("gpu", "float") in scaling else None
    if n_big:
        lines += [f"| Implementation (N = {n_big:,}) | GPU kernel (ms) | End-to-end (ms) | Throughput (G iter/s) "
                  "| Speedup vs. C++ OpenMP, same precision |",
                  "|---|---:|---:|---:|---:|"]
        for key, label, _, _ in IMPLS:
            r = scaling.get(key, {}).get(n_big)
            if r is None:
                continue
            base = scaling.get(("cpu_openmp", key[1]), {}).get(n_big)
            kern = f"{r['kernel_ms']:.1f}" if key[0] == "gpu" else "-"
            speed = f"**{base['total_ms'] / r['total_ms']:.1f}x**" if base and key[0] == "gpu" else (
                f"{base['total_ms'] / r['total_ms']:.2f}x" if base else "-")
            gips = r["n_params"] * r["iterations"] / (r["total_ms"] * 1e6)
            lines.append(f"| {label} | {kern} | {r['total_ms']:.1f} | {gips:.3f} | {speed} |")
        lines.append("")

    # Baselines too slow to run at n_big: compare at their largest measured N.
    for key, label, _, _ in IMPLS:
        if key in scaling and n_big not in scaling[key]:
            n = max(scaling[key])
            r = scaling[key][n]
            parts = [f"- {label}, N = {n:,}: {r['total_ms']:,.1f} ms"]
            for gk, gl in ((("gpu", "double"), "CUDA FP64"), (("gpu", "float"), "CUDA FP32")):
                g = scaling.get(gk, {}).get(n)
                if g:
                    parts.append(f"{gl} is {r['total_ms'] / g['total_ms']:,.0f}x faster")
            lines.append("; ".join(parts))
    lines.append("")

    work = [r for r in rows if r["section"] == "workflow"]
    if work:
        lines += ["### Workflow: README bifurcation diagram (4000 B values, 2000 transient + 300 stored, output included)",
                  "", "| Implementation | Kernel (ms) | D2H copy (ms) | End-to-end (ms) |", "|---|---:|---:|---:|"]
        for r in work:
            label = next(lb for k, lb, _, _ in IMPLS if k == (r["impl"], r["precision"]))
            gpu = r["impl"] == "gpu"
            lines.append(f"| {label} | {r['kernel_ms']:.1f} | {r['transfer_ms']:.1f} | {r['total_ms']:.1f} |"
                         if gpu else f"| {label} | - | - | {r['total_ms']:.1f} |")
        lines.append("")

    blocks = [r for r in rows if r["section"] == "blocksize"]
    if blocks:
        lines += ["### Block size (N = 1,048,576, kernel time in ms)", "", "| Block | FP64 | FP32 |", "|---:|---:|---:|"]
        by = defaultdict(dict)
        for r in blocks:
            by[r["block_size"]][r["precision"]] = r["kernel_ms"]
        for b in sorted(by):
            lines.append(f"| {b} | {by[b].get('double', float('nan')):.1f} | {by[b].get('float', float('nan')):.1f} |")
        lines.append("")

    Path(args.table).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
