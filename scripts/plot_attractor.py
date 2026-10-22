#!/usr/bin/env python3
"""Density map of the strange attractor, from a single-B, long-record sweep.

    python scripts/plot_attractor.py out/attractor
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import DENSITY_CMAP, FIGURES, INK_SECONDARY, Sweep, apply_style


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="output directory of ikeda_sweep (n_params = 1)")
    ap.add_argument("--bins", type=int, default=1000)
    ap.add_argument("--out", default=str(FIGURES / "attractor.png"))
    args = ap.parse_args()

    apply_style()
    s = Sweep(args.sweep)
    x, y = np.asarray(s.x[:, 0]), np.asarray(s.y[:, 0])
    B, lam = s.B[0], s.lyapunov[0]

    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    h, xe, ye = np.histogram2d(x, y, bins=args.bins)
    ax.imshow(
        h.T,
        origin="lower",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        cmap=DENSITY_CMAP,
        norm=LogNorm(vmin=1, vmax=h.max()),
        interpolation="antialiased",
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(r"Re $z_n$")
    ax.set_ylabel(r"Im $z_n$")
    ax.set_title(rf"Strange attractor, $B = {B:g}$", loc="left")
    ax.text(0.98, 0.52, f"{len(x):,} round trips\n" + rf"$\lambda_1 = {lam:.3f}$", transform=ax.transAxes,
            ha="right", va="center", color=INK_SECONDARY)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
