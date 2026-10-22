#!/usr/bin/env bash
# Rebuilds the slide deck from scratch: runs the sweeps, draws every figure,
# renders the animation frames, and compiles slides.pdf.
#
#   ./talk/build.sh [build-dir]
#
# Needs a CUDA build of the repository (see the README), a Python environment
# with scripts/requirements.txt, and a TeX distribution with pdflatex and
# latexmk (TeX Live: texlive-latex-extra, texlive-pictures).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build}
PY=${PYTHON:-python3}

"$BUILD/ikeda_thresholds" out/thresholds.json
[[ -f out/bifurcation/meta.json ]] || "$BUILD/ikeda_sweep" --config configs/bifurcation.cfg
[[ -f out/attractor/meta.json ]] || "$BUILD/ikeda_sweep" --config configs/attractor.cfg

$PY talk/make_slide_figures.py --out talk/figures
$PY talk/animation.py talk/figures/animation

cd talk
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=slides main.tex
cp build/slides.pdf slides.pdf
echo "wrote talk/slides.pdf"
