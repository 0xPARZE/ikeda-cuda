#!/usr/bin/env python3
"""Deterministic vs. noisy bifurcation diagram, side by side.

Noise smears the fine periodic windows and blurs the period-doubling cascade
before the deterministic B_inf: the "noisy precursors" of bifurcations.

    python scripts/plot_noise.py out/bifurcation out/stochastic
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import DENSITY_CMAP, FIGURES, Sweep, apply_style
from plot_bifurcation import density_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deterministic")
    ap.add_argument("stochastic")
    ap.add_argument("--b-min", type=float, default=0.35)
    ap.add_argument("--b-max", type=float, default=0.75)
    ap.add_argument("--out", default=str(FIGURES / "noise.png"))
    args = ap.parse_args()

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, path in zip(axes, (args.deterministic, args.stochastic)):
        s = Sweep(path)
        cols = (s.B >= args.b_min) & (s.B <= args.b_max)
        r = np.hypot(np.asarray(s.x[:, cols]), np.asarray(s.y[:, cols]))
        y_range = (0.0, 1.6)
        img = density_image(s.B[cols], r, 600, y_range)
        ax.imshow(img, origin="lower", aspect="auto",
                  extent=(s.B[cols][0], s.B[cols][-1], *y_range), cmap=DENSITY_CMAP,
                  norm=LogNorm(vmin=1.0 / r.shape[0], vmax=img.max()), interpolation="antialiased", rasterized=True)
        ax.grid(False)
        sigma = s.meta["noise_sigma"]
        ax.set_title("Deterministic" if sigma == 0 else rf"With noise, $\sigma = {sigma:g}$", loc="left")
        ax.set_xlabel("Mirror reflectivity $B$")
    axes[0].set_ylabel(r"Field amplitude $|z_n|$")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
