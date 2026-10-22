#!/usr/bin/env python3
"""Frames for the "CPU vs. GPU architectures" animation on the HPC slide.

A schematic, not a measurement: 400 independent tasks (one per B value)
processed by a latency-optimised CPU (8 cores, a few tasks at a time) and a
throughput-optimised GPU (every task in flight at once, each progressing
more slowly). The measured numbers are on the "Measured Performance" slide.

    python talk/animation.py talk/figures/animation
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from slide_style import ACCENT, BLUEGRAY, DARKGRAY, LIGHTGRAY, SECTIONBOX

GRID = 20  # 20 x 20 = 400 tasks
TASKS = GRID * GRID
CPU_CORES = 8  # tasks finished per CPU cycle
GPU_RATE = 0.03  # fraction of every task completed per GPU cycle
N_FRAMES = 120
DPI = 120


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmap_cpu = LinearSegmentedColormap.from_list("cpu", [LIGHTGRAY, BLUEGRAY, SECTIONBOX])
    cmap_gpu = LinearSegmentedColormap.from_list("gpu", [LIGHTGRAY, "#e07a82", ACCENT])
    X, Y = np.meshgrid(np.arange(GRID + 1), np.arange(GRID + 1))
    cpu_done_at = math.ceil(TASKS / CPU_CORES)
    gpu_done_at = math.ceil(1.0 / GPU_RATE)

    for frame in range(N_FRAMES):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
        fig.patch.set_facecolor("white")

        cpu = np.zeros(TASKS)
        done = min(frame * CPU_CORES, TASKS)
        cpu[:done] = 1.0
        cpu[done : min(done + CPU_CORES, TASKS)] = 0.5  # tasks on the cores right now
        gpu_progress = min(frame * GPU_RATE, 1.0)
        gpu = np.full(TASKS, gpu_progress)

        for ax, data, cmap in ((ax1, cpu, cmap_cpu), (ax2, gpu, cmap_gpu)):
            ax.pcolormesh(X, Y, data.reshape(GRID, GRID), cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=0.5)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")

        cpu_status = f"DONE ({cpu_done_at} cycles)" if done >= TASKS else f"Processing... ({done}/{TASKS})"
        gpu_status = f"DONE ({gpu_done_at} cycles)" if gpu_progress >= 1.0 else f"Processing... ({gpu_progress:.0%})"
        ax1.set_title(f"CPU Architecture\n(Latency-Optimized: {CPU_CORES} Threads)\n\nSTATUS: {cpu_status}",
                      fontsize=12, fontweight="bold", color=SECTIONBOX if done >= TASKS else DARKGRAY)
        ax2.set_title(f"GPU Architecture\n(Throughput-Optimized: {TASKS} Threads)\n\nSTATUS: {gpu_status}",
                      fontsize=12, fontweight="bold", color=ACCENT if gpu_progress >= 1.0 else DARKGRAY)
        fig.text(0.5, 0.05, f"Execution Cycle: {frame}", ha="center", fontsize=14, fontweight="bold", color=DARKGRAY)
        fig.tight_layout()
        fig.savefig(out / f"frame_{frame:03d}.png", facecolor="white")
        plt.close(fig)
    print(f"wrote {N_FRAMES} frames to {out}/")


if __name__ == "__main__":
    main()
