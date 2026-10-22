#!/usr/bin/env python3
"""Bifurcation diagram with the Lyapunov spectrum underneath (the README hero figure).

The top panel is a density image, not a scatter plot. Each column is a
histogram of the recorded |z| for one B, so millions of points render in
milliseconds, and the dense chaotic bands and the sharp periodic branches are
visible together.

    python scripts/plot_bifurcation.py out/bifurcation --thresholds out/thresholds.json
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import DENSITY_CMAP, FIGURES, INK_MUTED, INK_SECONDARY, SERIES, Sweep, apply_style, load_thresholds

# name -> (function of (x, y), axis label, default y-range). The y-ranges frame
# the cascade and the chaotic attractor. For B > 0.9 the orbit from z_0 = 0
# jumps to a coexisting high-intensity focus (|z| ~ 5), which would otherwise
# squash the whole diagram.
COMPONENTS = {
    "magnitude": (lambda x, y: np.hypot(x, y), r"Field amplitude $|z_n|$", (0.0, 2.2)),
    "real": (lambda x, y: x, r"Re $z_n$", (-0.7, 2.0)),
    "imag": (lambda x, y: y, r"Im $z_n$", (-2.0, 1.0)),
}


def density_image(B: np.ndarray, values: np.ndarray, n_bins: int, y_range: tuple[float, float]) -> np.ndarray:
    """Per-column histogram, normalised so that each B carries equal ink.

    Values outside y_range are dropped, not piled up on the edge rows."""
    n_rec, n_par = values.shape
    edges = np.linspace(*y_range, n_bins + 1)
    idx = np.searchsorted(edges, values, side="right") - 1
    cols = np.broadcast_to(np.arange(n_par), (n_rec, n_par))
    inside = (idx >= 0) & (idx < n_bins)
    img = np.zeros((n_bins, n_par))
    np.add.at(img, (idx[inside], cols[inside]), 1.0)
    return img / n_rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="output directory of ikeda_sweep")
    ap.add_argument("--thresholds", default="out/thresholds.json", help="ikeda_thresholds output (optional)")
    ap.add_argument("--component", choices=COMPONENTS, default="magnitude")
    ap.add_argument("--bins", type=int, default=900)
    ap.add_argument("--b-min", type=float, default=0.35, help="left edge of the plot")
    ap.add_argument("--b-max", type=float, default=0.92, help="right edge of the plot")
    ap.add_argument("--out", default=None, help="output PNG (default docs/figures/bifurcation_<component>.png)")
    args = ap.parse_args()

    apply_style()
    s = Sweep(args.sweep)
    fn, label, y_range = COMPONENTS[args.component]
    keep = (s.B >= args.b_min) & (s.B <= args.b_max)
    B = s.B[keep]
    values = fn(np.asarray(s.x[:, keep]), np.asarray(s.y[:, keep]))
    img = density_image(B, values, args.bins, y_range)

    fig, (ax, axl) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.08}
    )
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=(B[0], B[-1], *y_range),
        cmap=DENSITY_CMAP,
        norm=LogNorm(vmin=1.0 / values.shape[0], vmax=img.max()),
        interpolation="antialiased",
        rasterized=True,
    )
    ax.grid(False)
    ax.set_ylabel(label)
    ax.set_title(
        rf"Ikeda map: route to optical turbulence  ($A={s.meta['A']:g},\ p={s.meta['p']:g},\ \phi={s.meta['phi']:g}$)",
        loc="left",
    )

    lam = np.where(np.isfinite(s.lyapunov[keep]), s.lyapunov[keep], np.nan)
    axl.axhline(0, color=INK_MUTED, linewidth=0.8)
    axl.fill_between(B, 0, lam, where=lam > 0, color=SERIES[1], alpha=0.35, linewidth=0, label=r"$\lambda > 0$ (chaos)")
    axl.plot(B, lam, color=SERIES[0], linewidth=1.0, label=r"largest Lyapunov exponent $\lambda$")
    axl.set_ylim(-1.0, np.nanmax(lam) * 1.3)
    axl.set_ylabel(r"$\lambda$")
    axl.set_xlabel("Mirror reflectivity $B$")
    axl.legend(loc="upper left", ncols=2)

    th = load_thresholds(args.thresholds)
    if th:
        for k, b in enumerate(th["period_doubling"][:3], start=1):
            for a in (ax, axl):
                a.axvline(b, color=INK_MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
            ax.annotate(f"$B_{k}$", (b, y_range[1]), xytext=(3, -12), textcoords="offset points",
                        color=INK_SECONDARY, fontsize=9)
        b_cw = th["cw_bistability"]
        for a in (ax, axl):
            a.axvline(b_cw, color=SERIES[6], linewidth=1.0, linestyle=(0, (5, 2)))
        ax.annotate("CW bistability", (b_cw, y_range[1]), xytext=(3, -12), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9)

    axl.set_xlim(B[0], B[-1])
    out = args.out or FIGURES / f"bifurcation_{args.component}.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
