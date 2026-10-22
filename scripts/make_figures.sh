#!/usr/bin/env bash
# Regenerates every figure in docs/figures from scratch.
#
#   ./scripts/make_figures.sh [build-dir]
#
# The bifurcation and noise sweeps use the GPU. Without one, set
# `device = cpu` in configs/bifurcation.cfg and configs/stochastic.cfg
# (it takes about 10x longer, but the output is the same).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build}
PY=${PYTHON:-python3}

"$BUILD/ikeda_thresholds" out/thresholds.json
"$BUILD/ikeda_sweep" --config configs/bifurcation.cfg
"$BUILD/ikeda_sweep" --config configs/attractor.cfg
"$BUILD/ikeda_sweep" --config configs/stochastic.cfg

$PY scripts/plot_bifurcation.py out/bifurcation --component magnitude
$PY scripts/plot_bifurcation.py out/bifurcation --component real
$PY scripts/plot_bifurcation.py out/bifurcation --component imag
$PY scripts/plot_noise.py out/bifurcation out/stochastic
$PY scripts/plot_attractor.py out/attractor
$PY scripts/fractal_dimension.py out/attractor
if [[ -f benchmarks/results/latest.csv ]]; then
    $PY scripts/plot_benchmarks.py benchmarks/results/latest.csv
fi
