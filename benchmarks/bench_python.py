#!/usr/bin/env python3
"""Python baselines for the benchmark table.

Times the two Python approaches that came before the C++/CUDA code, on the
same workload as the C++ ``[scaling]`` benchmark (1000 iterations per B value,
tangent-vector Lyapunov exponent included):

* ``python_loop``: a pure-Python double loop, in the style of the original
  PHYS 4250 notebook;
* ``numpy``: NumPy vectorised across the B axis, the best you can reasonably
  do in plain Python.

Results are appended to the same CSV that ``ikeda_bench`` writes, so
``scripts/plot_benchmarks.py`` can plot every implementation on one chart.

    python benchmarks/bench_python.py benchmarks/results/latest.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np

A, P, PHI = 1.0, 6.0, 0.4
N_TRANSIENT, N_RECORD = 500, 500  # must match benchmarks/bench_sweep.cpp


def sweep_python_loop(bs: list[float]) -> list[float]:
    lyap = []
    for b in bs:
        x = y = 0.0
        vx, vy, acc = 1.0, 0.0, 0.0
        for k in range(N_TRANSIENT + N_RECORD):
            d = 1.0 + x * x + y * y
            t = PHI - P / d
            c, s = math.cos(t), math.sin(t)
            xn = A + b * (x * c - y * s)
            yn = b * (x * s + y * c)
            if k >= N_TRANSIENT:
                kk = 2.0 * P / (d * d)
                dtx, dty = kk * x, kk * y
                u = xn - A
                wx = (b * c - yn * dtx) * vx + (-b * s - yn * dty) * vy
                wy = (b * s + u * dtx) * vx + (b * c + u * dty) * vy
                nrm = math.hypot(wx, wy)
                if nrm > 0:
                    acc += math.log(nrm)
                    vx, vy = wx / nrm, wy / nrm
                else:
                    acc = -math.inf
            x, y = xn, yn
        lyap.append(acc / N_RECORD)
    return lyap


def sweep_numpy(bs: np.ndarray) -> np.ndarray:
    x = np.zeros_like(bs)
    y = np.zeros_like(bs)
    vx, vy = np.ones_like(bs), np.zeros_like(bs)
    acc = np.zeros_like(bs)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(N_TRANSIENT + N_RECORD):
            d = 1.0 + x * x + y * y
            t = PHI - P / d
            c, s = np.cos(t), np.sin(t)
            xn = A + bs * (x * c - y * s)
            yn = bs * (x * s + y * c)
            if k >= N_TRANSIENT:
                kk = 2.0 * P / (d * d)
                dtx, dty = kk * x, kk * y
                u = xn - A
                wx = (bs * c - yn * dtx) * vx + (-bs * s - yn * dty) * vy
                wy = (bs * s + u * dtx) * vx + (bs * c + u * dty) * vy
                nrm = np.hypot(wx, wy)
                acc += np.log(nrm)
                ok = nrm > 0
                vx = np.where(ok, wx / nrm, vx)
                vy = np.where(ok, wy / nrm, vy)
            x, y = xn, yn
    return acc / N_RECORD


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="benchmarks/results/latest.csv")
    ap.add_argument("--max-loop-n", type=int, default=4096, help="largest N for the pure-Python loop")
    ap.add_argument("--max-numpy-n", type=int, default=1 << 20, help="largest N for NumPy")
    args = ap.parse_args()

    iters = N_TRANSIENT + N_RECORD
    rows = []
    n = 256
    while n <= args.max_numpy_n:
        bs = np.linspace(0.0, 1.0, n)
        impls = [("numpy", lambda: sweep_numpy(bs))]
        if n <= args.max_loop_n:
            impls.insert(0, ("python_loop", lambda: sweep_python_loop(bs.tolist())))
        for name, fn in impls:
            if n <= 4096:  # warm-up run, only where it is cheap
                fn()
            t0 = time.perf_counter()
            fn()
            ms = (time.perf_counter() - t0) * 1e3
            gips = n * iters / (ms * 1e6)
            print(f"  scaling   {name:<12} double N={n:<8} total {ms:10.1f} ms  {gips:7.4f} Giter/s")
            rows.append(["scaling", name, "double", n, iters, 0, 0, 1, ms, 0.0, ms, gips])
        n *= 4

    with open(args.csv, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
