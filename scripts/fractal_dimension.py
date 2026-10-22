#!/usr/bin/env python3
"""Box-counting dimension of the attractor, cross-checked against Kaplan-Yorke.

Two independent estimates of the attractor's fractal dimension:

* Box counting (D_0). Cover the attractor with boxes of side eps, count the
  occupied boxes N(eps), and fit the slope of log N against log(1/eps) over
  the scaling range.
* Kaplan-Yorke (D_KY). For a 2D map, D_KY = 1 + lambda_1 / |lambda_2|. The
  sweep gives lambda_1, and since det J = B^2 everywhere, lambda_2 = 2 ln B - lambda_1
  exactly, so no second tangent vector is needed.

The Kaplan-Yorke conjecture identifies D_KY with the information dimension
D_1, and D_1 <= D_0. For the Ikeda attractor the two are known to be close, so
agreement between them is an end-to-end check on both the orbit and the
Lyapunov exponent.

    python scripts/fractal_dimension.py out/attractor
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, INK_SECONDARY, SERIES, Sweep, apply_style


def box_counts(x: np.ndarray, y: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    x0, y0 = x.min(), y.min()
    counts = []
    for eps in sizes:
        ix = np.floor((x - x0) / eps).astype(np.int64)
        iy = np.floor((y - y0) / eps).astype(np.int64)
        counts.append(np.unique(ix * (iy.max() + 1) + iy).size)
    return np.array(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="output directory of ikeda_sweep (n_params = 1)")
    ap.add_argument("--eps-min", type=float, default=2e-3, help="smallest box (limited by sample count)")
    ap.add_argument("--eps-max", type=float, default=1e-1, help="largest box in the fit")
    ap.add_argument("--out", default=str(FIGURES / "box_counting.png"))
    args = ap.parse_args()

    apply_style()
    s = Sweep(args.sweep)
    x, y = np.asarray(s.x[:, 0]), np.asarray(s.y[:, 0])
    B, lam1 = float(s.B[0]), float(s.lyapunov[0])
    lam2 = 2.0 * np.log(B) - lam1
    d_ky = 1.0 + lam1 / abs(lam2)

    sizes = np.geomspace(args.eps_min, args.eps_max, 16)
    n = box_counts(x, y, sizes)
    slope, intercept = np.polyfit(np.log(1.0 / sizes), np.log(n), 1)

    print(f"B = {B:g}, {len(x):,} points")
    print(f"lambda_1 = {lam1:.4f}, lambda_2 = 2 ln B - lambda_1 = {lam2:.4f}")
    print(f"box-counting D_0 = {slope:.3f}")
    print(f"Kaplan-Yorke D_KY = {d_ky:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(1.0 / sizes, n, "o", color=SERIES[0], markersize=6, label="occupied boxes $N(\\varepsilon)$")
    fit = np.exp(intercept) * (1.0 / sizes) ** slope
    ax.loglog(1.0 / sizes, fit, color=SERIES[1], linewidth=1.5, label=f"fit: $D_0 = {slope:.3f}$")
    ax.set_xlabel(r"$1/\varepsilon$")
    ax.set_ylabel(r"$N(\varepsilon)$")
    ax.set_title(f"Box-counting dimension, $B = {B:g}$", loc="left")
    ax.text(0.98, 0.04, f"Kaplan–Yorke: $D_{{KY}} = {d_ky:.3f}$", transform=ax.transAxes, ha="right",
            color=INK_SECONDARY)
    ax.legend(loc="upper left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
