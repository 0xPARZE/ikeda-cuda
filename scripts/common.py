"""Shared helpers for the plotting scripts: loading sweep output and figure style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures"

# Palette: categorical slots in a fixed, colour-vision-deficiency-checked order,
# plus a single-hue sequential ramp for densities. Text always uses the ink
# colours, never a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Densities start at a visible mid-blue rather than at the surface colour, so
# the sparsest parts of the attractor still read. Empty bins (masked by the log
# norm) are painted with the surface colour, so anti-aliasing blends them
# cleanly instead of into grey.
DENSITY_CMAP = LinearSegmentedColormap.from_list("density_blue", SEQUENTIAL_BLUE[2:]).with_extremes(
    bad=SURFACE, under=SURFACE
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


class Sweep:
    """Output directory written by `ikeda_sweep` (see include/ikeda/io.hpp)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.B = np.load(self.path / "B.npy")
        self.lyapunov = np.load(self.path / "lyapunov.npy")
        has_orbit = (self.path / "x.npy").exists()
        # Arrays are [n_record, n_params]; memory-map them so large sweeps are cheap to open.
        self.x = np.load(self.path / "x.npy", mmap_mode="r") if has_orbit else None
        self.y = np.load(self.path / "y.npy", mmap_mode="r") if has_orbit else None


def load_thresholds(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
