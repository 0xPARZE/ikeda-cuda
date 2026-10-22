"""Colours and matplotlib style shared by the slide figures (matches main.tex)."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

DARKGRAY = "#2b2d42"  # \definecolor{darkgray}
BLUEGRAY = "#8d99ae"  # \definecolor{bluegray}
LIGHTGRAY = "#edf2f4"  # \definecolor{lightgray}
SECTIONBOX = "#303861"  # \definecolor{sectionbox}
ACCENT = "#e63946"  # highlight colour used for "chaos" and the GPU

# Density ramp for bifurcation diagrams and attractors: blue-grey for rarely
# visited regions up to near-black navy for the densest. It starts dark enough
# to survive a projector.
DENSITY = LinearSegmentedColormap.from_list("deck_density", [BLUEGRAY, "#4a5578", SECTIONBOX, "#14182e"]).with_extremes(
    bad="white", under="white"
)

# Slides are 16 cm x 9 cm with 1.5 cm side margins, so the text block is 13 cm.
# Figures are drawn at their final printed size so 7-8 pt text matches the slide.
CM = 1 / 2.54
TEXT_WIDTH = 13.0 * CM


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlecolor": DARKGRAY,
            "axes.labelcolor": DARKGRAY,
            "axes.edgecolor": BLUEGRAY,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": DARKGRAY,
            "ytick.color": DARKGRAY,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "mathtext.fontset": "dejavusans",
        }
    )
