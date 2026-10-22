#!/usr/bin/env python3
"""Every data figure in the slide deck, regenerated from the repository's own output.

Inputs come from the same programs as the README figures (see talk/build.sh),
so the numbers on the slides and in the repository can't drift apart.

    python talk/make_slide_figures.py --out talk/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from common import Sweep  # noqa: E402
from fractal_dimension import box_counts  # noqa: E402
from plot_bifurcation import density_image  # noqa: E402

import slide_style as st  # noqa: E402

B_RANGE = (0.30, 0.92)  # B > 0.903: the orbit from z_0 = 0 jumps to a coexisting focus
ZOOM = (0.12, 0.42, -0.66, -0.26)
BOX_EPS = 0.125  # box size drawn in the box-counting illustration  # Re min, Re max, Im min, Im max of the zoomed attractor panel


def columns(s: Sweep, lo: float, hi: float) -> np.ndarray:
    return (s.B >= lo) & (s.B <= hi)


def density_panel(ax, B, values, y_range, bins=700):
    img = density_image(B, values, bins, y_range)
    ax.imshow(img, origin="lower", aspect="auto", extent=(B[0], B[-1], *y_range), cmap=st.DENSITY,
              norm=LogNorm(vmin=1.0 / values.shape[0], vmax=img.max()), interpolation="antialiased", rasterized=True)


def bifurcation(s: Sweep, out: Path) -> None:
    keep = columns(s, *B_RANGE)
    B = s.B[keep]
    x, y = np.asarray(s.x[:, keep]), np.asarray(s.y[:, keep])
    panels = {
        "Bifurcation_Magnitude": (np.hypot(x, y), r"Field amplitude $|z_n|$", (0.0, 2.2)),
        "Bifurcation_Real": (x, r"Re$(z_n)$", (-0.7, 2.0)),
        "Bifurcation_Imaginary": (y, r"Im$(z_n)$", (-2.0, 1.0)),
    }
    for name, (vals, label, yr) in panels.items():
        fig, ax = plt.subplots(figsize=(0.95 * st.TEXT_WIDTH, 5.0 * st.CM))
        density_panel(ax, B, vals, yr)
        ax.set_xlabel(r"Reflection coefficient $B$")
        ax.set_ylabel(label)
        fig.savefig(out / f"{name}.png")
        plt.close(fig)


def lyapunov(s: Sweep, th: dict, out: Path) -> None:
    keep = columns(s, *B_RANGE)
    B, lam = s.B[keep], s.lyapunov[keep]
    b_inf = feigenbaum_limit(th["period_doubling"])
    fig, ax = plt.subplots(figsize=(0.95 * st.TEXT_WIDTH, 5.0 * st.CM))
    ax.axhline(0, color=st.BLUEGRAY, linewidth=0.6)
    ax.fill_between(B, 0, lam, where=lam > 0, color=st.ACCENT, alpha=0.3, linewidth=0, label=r"$\lambda > 0$: chaos")
    ax.plot(B, lam, color=st.DARKGRAY, linewidth=0.6, label=r"maximal Lyapunov exponent $\lambda$")
    for k, b in enumerate(th["period_doubling"][:3], start=1):
        ax.axvline(b, color=st.BLUEGRAY, linewidth=0.6, linestyle=(0, (2, 2)))
        ax.text(b + 0.004, -0.92, f"$B_{k}$", color=st.DARKGRAY, fontsize=6.5)
    ax.axvline(b_inf, color=st.ACCENT, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.text(b_inf + 0.004, 0.55, rf"$B_\infty \approx {b_inf:.4f}$", color=st.ACCENT, fontsize=6.5)
    ax.set_xlim(*B_RANGE)
    ax.set_ylim(-1.0, 0.7)
    ax.set_xlabel(r"Reflection coefficient $B$")
    ax.set_ylabel(r"$\lambda$")
    ax.legend(loc="upper left")
    fig.savefig(out / "Lyapunov_Spectrum.png")
    plt.close(fig)


def bifurcation_slim(s: Sweep, th: dict, out: Path) -> None:
    keep = columns(s, 0.36, 0.665)
    B = s.B[keep]
    r = np.hypot(np.asarray(s.x[:, keep]), np.asarray(s.y[:, keep]))
    fig, ax = plt.subplots(figsize=(0.46 * st.TEXT_WIDTH, 4.6 * st.CM))
    yr = (0.45, 1.08)
    density_panel(ax, B, r, yr, bins=500)
    for k, b in enumerate(th["period_doubling"][:3], start=1):
        ax.axvline(b, color=st.ACCENT, linewidth=0.6, linestyle=(0, (2, 2)))
        ax.text(b - 0.004, yr[1] - 0.06, f"$B_{k}$", color=st.ACCENT, fontsize=6.5, ha="right")
    ax.set_xlabel(r"Reflection coefficient $B$")
    ax.set_ylabel(r"$|z_n|$")
    fig.savefig(out / "Bifurcation_Magnitude_Slim.png")
    plt.close(fig)


def attractor_density(ax, x, y, extent, bins, aspect="auto"):
    """Log-density image of an orbit. The phase-space panels use square axes
    (aspect="auto"), as in the original deck. The attractor is 1.5x taller than
    it is wide, so its x-axis is stretched to fill the square."""
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[extent[:2], extent[2:]])
    ax.imshow(h.T, origin="lower", extent=extent, cmap=st.DENSITY, norm=LogNorm(vmin=1, vmax=max(h.max(), 2)),
              interpolation="antialiased", rasterized=True, aspect=aspect)


def phase_space(s: Sweep, out: Path) -> None:
    x, y = np.asarray(s.x[:, 0]), np.asarray(s.y[:, 0])
    full = (-0.45, 1.75, -2.35, 0.95)
    size = (5.4 * st.CM, 5.3 * st.CM)  # square plotting area

    fig, ax = plt.subplots(figsize=size)
    attractor_density(ax, x, y, full, 900)
    ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False,
                           edgecolor=st.ACCENT, linewidth=0.8))
    ax.set_xlabel(r"Re$(z_n)$")
    ax.set_ylabel(r"Im$(z_n)$")
    fig.savefig(out / "Phase_Space.png")
    plt.close(fig)

    inside = (x > ZOOM[0]) & (x < ZOOM[1]) & (y > ZOOM[2]) & (y < ZOOM[3])
    fig, ax = plt.subplots(figsize=size)
    attractor_density(ax, x[inside], y[inside], ZOOM, 900)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(st.ACCENT)
    ax.set_xlabel(r"Re$(z_n)$")
    ax.set_ylabel(r"Im$(z_n)$")
    fig.savefig(out / "Phase_Space_Zoom.png")
    plt.close(fig)

    # Section-divider artwork: the attractor alone, in the deck's navy, on a transparent background.
    fig = plt.figure(figsize=(5 * st.CM * 1.6, 5 * st.CM * 1.6))
    ax = fig.add_axes((0, 0, 1, 1))
    h, _, _ = np.histogram2d(x, y, bins=700, range=[full[:2], full[2:]])
    alpha = np.clip(np.log1p(h.T) / np.log1p(np.percentile(h[h > 0], 99)), 0, 1)
    rgba = np.zeros((*alpha.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(st.SECTIONBOX)
    rgba[..., 3] = alpha
    ax.imshow(rgba, origin="lower", extent=full, interpolation="antialiased", aspect="auto")
    ax.axis("off")
    fig.savefig(out / "section_chaos.png", transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def box_grid(s: Sweep, out: Path, eps: float = BOX_EPS) -> int:
    """Box counting illustrated: the attractor covered by a grid of eps x eps
    boxes, with occupied boxes shaded. Uses the same square axes as the other
    phase-space panels, so Re is stretched and the boxes look slightly wide."""
    x, y = np.asarray(s.x[:, 0]), np.asarray(s.y[:, 0])
    x0, y0 = x.min(), y.min()  # same grid origin as box_counts()
    ix = np.floor((x - x0) / eps).astype(int)
    iy = np.floor((y - y0) / eps).astype(int)
    occupied = np.zeros((ix.max() + 1, iy.max() + 1), dtype=bool)
    occupied[ix, iy] = True
    nx, ny = occupied.shape

    fig, ax = plt.subplots(figsize=(5.4 * st.CM, 5.3 * st.CM))  # same as Phase_Space.png
    for i, j in zip(*np.nonzero(occupied)):
        ax.add_patch(Rectangle((x0 + i * eps, y0 + j * eps), eps, eps, facecolor=st.ACCENT, alpha=0.3,
                               edgecolor="none", zorder=1))
    for i in range(nx + 1):
        ax.plot([x0 + i * eps] * 2, [y0, y0 + ny * eps], color=st.BLUEGRAY, linewidth=0.25, alpha=0.6, zorder=2)
    for j in range(ny + 1):
        ax.plot([x0, x0 + nx * eps], [y0 + j * eps] * 2, color=st.BLUEGRAY, linewidth=0.25, alpha=0.6, zorder=2)
    extent = (x0, x0 + nx * eps, y0, y0 + ny * eps)
    h, _, _ = np.histogram2d(x, y, bins=(nx * 24, ny * 24), range=[extent[:2], extent[2:]])
    shown = np.ma.masked_equal(h.T, 0)
    see_through = st.DENSITY.with_extremes(bad=(0, 0, 0, 0), under=(0, 0, 0, 0))  # empty cells show the grid
    ax.imshow(shown, origin="lower", extent=extent, cmap=see_through, norm=LogNorm(vmin=1, vmax=h.max()),
              interpolation="antialiased", rasterized=True, zorder=3, aspect="auto")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("auto")
    ax.set_xlabel(r"Re$(z_n)$")
    ax.set_ylabel(r"Im$(z_n)$")
    fig.savefig(out / "Box_Counting_Grid.png")
    plt.close(fig)
    return int(occupied.sum())


def fractal(s: Sweep, out: Path) -> tuple[float, float]:
    x, y = np.asarray(s.x[:, 0]), np.asarray(s.y[:, 0])
    B, lam1 = float(s.B[0]), float(s.lyapunov[0])
    d_ky = 1.0 + lam1 / abs(2.0 * np.log(B) - lam1)
    sizes = np.geomspace(2e-3, 1e-1, 16)  # same fit as scripts/fractal_dimension.py
    n = box_counts(x, y, sizes)
    slope, intercept = np.polyfit(np.log(1 / sizes), np.log(n), 1)

    fig, ax = plt.subplots(figsize=(0.46 * st.TEXT_WIDTH, 5.3 * st.CM))
    ax.loglog(1 / sizes, n, "o", color=st.SECTIONBOX, markersize=3.5, label=r"occupied boxes $N(\varepsilon)$")
    ax.loglog(1 / sizes, np.exp(intercept) * (1 / sizes) ** slope, color=st.ACCENT, linewidth=1.0,
              label=f"fit: $D_0 = {slope:.3f}$")
    ax.set_xlabel(r"$1/\varepsilon$")
    ax.set_ylabel(r"$N(\varepsilon)$")
    ax.text(0.97, 0.05, f"Kaplan–Yorke: $D_{{KY}} = {d_ky:.3f}$", transform=ax.transAxes, ha="right",
            color=st.DARKGRAY, fontsize=6.5)
    ax.legend(loc="upper left")
    fig.savefig(out / "Fractal_Dimension_LogLog.png")
    plt.close(fig)
    return slope, d_ky


def title_cascade(s: Sweep, out: Path) -> None:
    """Vertical white cascade for the dark right-hand side of the title slide."""
    keep = columns(s, 0.30, 0.902)
    B = s.B[keep]
    r = np.hypot(np.asarray(s.x[:, keep]), np.asarray(s.y[:, keep]))
    z_range = (0.05, 1.55)
    img = density_image(B, r, 900, z_range)  # [z_bin, B]
    counts = np.maximum(img * r.shape[0], 1.0)  # visits per bin
    alpha = np.where(img > 0, 0.35 + 0.65 * np.clip(np.log(counts) / np.log(r.shape[0]), 0, 1), 0)
    rgba = np.ones((*alpha.T.shape, 4))
    rgba[..., 3] = alpha.T  # rows = B (top: small B), columns = |z|
    fig = plt.figure(figsize=(6.5 * st.CM, 11.5 * st.CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgba, aspect="auto", interpolation="antialiased")
    ax.axis("off")
    fig.savefig(out / "ikeda_cascade.png", transparent=True, bbox_inches=None, pad_inches=0, dpi=400)
    plt.close(fig)


def scaling(bench_csv: Path, out: Path) -> dict:
    rows = list(csv.DictReader(bench_csv.open()))
    series = {
        ("numpy", "double"): ("NumPy", st.BLUEGRAY, "s", (0, (3, 1.5))),
        ("cpu_openmp", "double"): ("C++ OpenMP, FP64", st.DARKGRAY, "v", "-"),
        ("gpu", "double"): ("CUDA, FP64", st.SECTIONBOX, "D", "-"),
        ("gpu", "float"): ("CUDA, FP32", st.ACCENT, "P", "-"),
    }
    fig, ax = plt.subplots(figsize=(0.55 * st.TEXT_WIDTH, 5.4 * st.CM))
    times = {}
    for key, (label, colour, marker, ls) in series.items():
        pts = sorted((int(r["n_params"]), float(r["total_ms"])) for r in rows
                     if r["section"] == "scaling" and (r["impl"], r["precision"]) == key)
        if not pts:
            continue
        n, t = zip(*pts)
        times[key] = dict(pts)
        ax.loglog(n, t, marker=marker, markersize=3, color=colour, linestyle=ls, linewidth=1.0)
        ax.annotate(label, (n[-1], t[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6, color=colour)
    ax.set_xlim(right=ax.get_xlim()[1] * 9)
    ax.set_xlabel(r"Number of $B$ values, $N$")
    ax.set_ylabel("Wall time (ms)")
    fig.savefig(out / "scaling.png")
    plt.close(fig)
    return times


def feigenbaum_limit(t: list[float], delta: float = 4.669201609102990) -> float:
    """Accumulation point B_inf, extrapolated geometrically from the last two thresholds."""
    return t[-1] + (t[-1] - t[-2]) / (delta - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bifurcation", default=str(REPO / "out/bifurcation"))
    ap.add_argument("--attractor", default=str(REPO / "out/attractor"))
    ap.add_argument("--thresholds", default=str(REPO / "out/thresholds.json"))
    ap.add_argument("--bench", default=str(REPO / "benchmarks/results/latest.csv"))
    ap.add_argument("--out", default=str(REPO / "talk/figures"))
    args = ap.parse_args()

    st.apply()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    th = json.loads(Path(args.thresholds).read_text())
    bif, att = Sweep(args.bifurcation), Sweep(args.attractor)

    bifurcation(bif, out)
    lyapunov(bif, th, out)
    bifurcation_slim(bif, th, out)
    title_cascade(bif, out)
    phase_space(att, out)
    d0, dky = fractal(att, out)
    n_boxes = box_grid(att, out)
    scaling(Path(args.bench), out)
    print(f"B_inf ~ {feigenbaum_limit(th['period_doubling']):.6f}, D0 = {d0:.3f}, D_KY = {dky:.3f}, "
          f"N(eps={BOX_EPS}) = {n_boxes}")
    print(f"wrote slide figures to {out}/")


if __name__ == "__main__":
    main()
