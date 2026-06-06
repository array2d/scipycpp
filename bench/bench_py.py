#!/usr/bin/env python3
"""
bench/bench_py.py — Python-side performance comparison: scipycpp vs scipy.

Benchmarks key scipycpp APIs against their scipy equivalents on the same
hardware. Run from the project root after building tests/:

    make -C tests          # build the pybind11 module
    python3 bench/bench_py.py

Output: wall-time table (median over REPEAT calls) for each API.
"""

import sys, os, time
import numpy as np

# Locate the built .so next to tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
try:
    import scipycpp as cpp
except ImportError as e:
    sys.exit(
        f"ERROR: cannot import scipycpp — did you run 'make -C tests'?\n  {e}"
    )

import scipy.stats as sp_stats
import scipy.integrate as sp_integrate
import scipy.linalg as sp_linalg
from scipy.spatial.transform import Rotation as sp_Rotation

REPEAT = 200
WARMUP  = 20

def bench(label, fn, *, n=REPEAT):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return label, float(np.median(times))

def fmt(t):
    if t < 1e-6:
        return f"{t*1e9:.1f} ns"
    if t < 1e-3:
        return f"{t*1e6:.2f} µs"
    return f"{t*1e3:.3f} ms"

results = []

# ------------------------------------------------------------------
# stats.norm.pdf  (batch, float64)
# ------------------------------------------------------------------
for N in (1_000, 100_000):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N)

    results.append(bench(
        f"stats.norm.pdf   N={N:>7}  [scipy ]",
        lambda: sp_stats.norm.pdf(x)))
    results.append(bench(
        f"stats.norm.pdf   N={N:>7}  [cpp   ]",
        lambda: cpp.stats.norm.pdf(x)))
    results.append(("", None))

# ------------------------------------------------------------------
# stats.norm.cdf  (batch, float64)
# ------------------------------------------------------------------
for N in (1_000, 100_000):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N)

    results.append(bench(
        f"stats.norm.cdf   N={N:>7}  [scipy ]",
        lambda: sp_stats.norm.cdf(x)))
    results.append(bench(
        f"stats.norm.cdf   N={N:>7}  [cpp   ]",
        lambda: cpp.stats.norm.cdf(x)))
    results.append(("", None))

# ------------------------------------------------------------------
# integrate.trapezoid
# ------------------------------------------------------------------
for N in (1_000, 1_000_000):
    rng = np.random.default_rng(13)
    y = rng.uniform(1, 100, N)

    results.append(bench(
        f"integrate.trapezoid N={N:>7}  [scipy ]",
        lambda: sp_integrate.trapezoid(y)))
    results.append(bench(
        f"integrate.trapezoid N={N:>7}  [cpp   ]",
        lambda: cpp.trapezoid(y)))
    results.append(("", None))

# ------------------------------------------------------------------
# integrate.simpson
# ------------------------------------------------------------------
for N in (1_001, 999_999):   # must be odd for scipy.integrate.simpson
    rng = np.random.default_rng(21)
    y = rng.uniform(1, 100, N)

    results.append(bench(
        f"integrate.simpson   N={N:>7}  [scipy ]",
        lambda: sp_integrate.simpson(y)))
    results.append(bench(
        f"integrate.simpson   N={N:>7}  [cpp   ]",
        lambda: cpp.simpson(y)))
    results.append(("", None))

# ------------------------------------------------------------------
# linalg.solve  (2×2, well-conditioned)
# ------------------------------------------------------------------
rng = np.random.default_rng(99)
A2 = rng.standard_normal((2, 2)) + 3 * np.eye(2)
b2 = rng.standard_normal(2)

results.append(bench("linalg.solve 2×2        [scipy ]",
                     lambda: sp_linalg.solve(A2, b2)))
results.append(bench("linalg.solve 2×2        [cpp   ]",
                     lambda: cpp.linalg.solve(A2, b2)))
results.append(("", None))

# ------------------------------------------------------------------
# Rotation.from_euler + as_matrix  (single-axis z — typical ego yaw)
# ------------------------------------------------------------------
yaw = 0.3
results.append(bench(
    "Rotation.from_euler('z')+as_matrix  [scipy ]",
    lambda: sp_Rotation.from_euler('z', yaw).as_matrix()))
results.append(bench(
    "Rotation.from_euler('z')+as_matrix  [cpp   ]",
    lambda: cpp.spatial.transform.Rotation.from_euler('z', yaw).as_matrix()))
results.append(("", None))

# Rotation.from_euler + as_matrix  (multi-axis xyz)
angles = np.array([0.1, 0.2, 0.3])
results.append(bench(
    "Rotation.from_euler('xyz')+as_matrix  [scipy ]",
    lambda: sp_Rotation.from_euler('xyz', angles).as_matrix()))
results.append(bench(
    "Rotation.from_euler('xyz')+as_matrix  [cpp   ]",
    lambda: cpp.spatial.transform.Rotation.from_euler('xyz', angles).as_matrix()))

# ------------------------------------------------------------------
# Print table
# ------------------------------------------------------------------
print()
print(f"  {'Benchmark':<56}  {'Median / call':>14}")
print("  " + "-" * 74)
for label, t in results:
    if t is None:
        print()
    else:
        print(f"  {label:<56}  {fmt(t):>14}")
print()
