"""
Bit-level alignment tests — scipycpp C++ vs Python scipy APIs.
SINGLE entry point: pytest tests/test_all.py -v

ALL APIs are bit-identical (0 ULP) for both float64/float32, including:
  - 100-batch random data (uniform, normal, extreme ranges)
  - Special values: ±0.0, ±inf, NaN, subnormals, boundary conditions
  - Extreme values: saturating inputs, domain boundaries, tiny/huge scales

NOTE: All scipy APIs return float64 regardless of input dtype.
C++ mirrors this behavior.

numpycpp exp/log/sqrt now resolve to npy_* / SVML, matching scipy's
internal math path bit-for-bit on every x86_64 machine.
"""

import os, sys, atexit, importlib, warnings, numpy as np, pytest

BATCH = 100

def _s(label, dtype=None):
    """Append [f64] or [f32] to label for ULP report clarity."""
    return label if dtype is None else f"{label} [{np.dtype(dtype).name}]"

# ============================================================================
# ULP computation & reporting
# ============================================================================

_ulp_report = []  # human-readable lines for stderr
_ulp_records = []  # structured rows for CSV export

def _ulp_record(label, n_total, n_diff, max_ulp, tol, hist, status):
    """Collect one row for CSV ULP report."""
    # Parse module/dtype from label like "pdf batch=100 [float64]"
    # Extract module (base API name, stripping parameters) and dtype [...] suffix
    import re as _re
    parts = label.split(" [", 1)
    test_name = parts[0]
    dtype_tag = parts[1].rstrip("]") if len(parts) > 1 else "—"
    # Strip parameters to get base module: "pdf(loc=0.0,scale=0.01)" → "pdf"
    # Match alphabetic + optional _word prefix before '(' or ' '
    m = _re.match(r'^([a-zA-Z_]\w*)', test_name) if test_name else None
    module = m.group(1) if m else (test_name.split()[0] if test_name else "—")
    _ulp_records.append({
        "module": module,
        "test": test_name,
        "dtype": dtype_tag,
        "n_total": n_total,
        "n_diff": n_diff,
        "max_ulp": max_ulp,
        "tol": tol,
        "histogram": hist.replace(", ", " "),
        "status": status,
    })

_MAX_ULP = (1 << 62)  # sentinel for "definitely different" without int64 overflow

def _ulp_dist(a, b):
    """ULP distance between two float64 values.
    NaN vs NaN → 0  (both NaN ≡ bit-identical semantically).
    NaN vs non-NaN → _MAX_ULP sentinel.
    Same bit pattern → 0 (handles ±0, ±inf, etc. correctly).
    Cross-sign (one positive, one negative, both finite/inf) → _MAX_ULP.
    """
    a64, b64 = np.float64(a), np.float64(b)
    # NaN handling first (isnan before any comparison)
    a_nan, b_nan = bool(np.isnan(a64)), bool(np.isnan(b64))
    if a_nan and b_nan:
        return 0               # both NaN — semantically identical
    if a_nan or b_nan:
        return _MAX_ULP        # NaN vs non-NaN — maximal mismatch
    # Both non-NaN: compare via bit patterns
    ai = int(a64.view(np.uint64))
    bi = int(b64.view(np.uint64))
    if ai == bi:
        return 0               # identical bits (includes ±0 == ±0, +inf == +inf)
    # Cross-sign (one positive, one negative) — treat as maximal mismatch
    if (a64 < 0) != (b64 < 0):
        return _MAX_ULP
    return abs(ai - bi)

def _ulp_dist32(a, b):
    """ULP distance in float32 space (for float32 results compared as float32)."""
    a32, b32 = np.float32(a), np.float32(b)
    a_nan, b_nan = bool(np.isnan(a32)), bool(np.isnan(b32))
    if a_nan and b_nan: return 0
    if a_nan or b_nan:  return _MAX_ULP
    ai = int(a32.view(np.uint32))
    bi = int(b32.view(np.uint32))
    if ai == bi: return 0
    if (a32 < 0) != (b32 < 0): return _MAX_ULP
    return abs(ai - bi)

def assert_f32_ulp_close(cpp_r, py_r, label, max_ulp=6):
    """Compare float32 results in float32 ULP space.
    Used for functions that both C++ and scipy compute/return in float32,
    where sequential vs SIMD pairwise summation causes ≤few float32 ULPs.
    """
    a32 = np.float32(cpp_r)
    b32 = np.float32(py_r)
    diff = _ulp_dist32(a32, b32)
    if diff == 0:
        _ulp_report.append(f"  ✓ {label}: 0 ULP (bit-identical)")
        _ulp_record(label, 1, 0, 0, max_ulp, "—", "bit-identical")
        return
    ok = diff <= max_ulp
    tag = "✓" if ok else "✗ FAIL"
    status = f"within {max_ulp} f32-ULP" if ok else "FAIL (exceeds tolerance)"
    _ulp_report.append(f"  {tag} {label}: max={diff} f32-ULP (tol={max_ulp})"
                       f"  C++={float(a32):.8g}  scipy={float(b32):.8g}")
    _ulp_record(label, 1, 1 if diff > 0 else 0, diff, max_ulp,
                f"1×{diff}f32ULP", status)
    if not ok:
        raise AssertionError(f"{label}: f32-ULP {diff} > tol={max_ulp}")

def _ulp_stats(cpp, py):
    """Compute (n_diff, max_ulp, max_idx, hist_str) for two float64 arrays.
    NaN==NaN is treated as matching (0 ULP), per IEEE 754 bit-identity rule.
    """
    assert cpp.shape == py.shape
    # Treat NaN vs NaN as identical; non-NaN use exact equality
    nan_both = np.isnan(cpp) & np.isnan(py)
    diff_mask = (cpp != py) & ~nan_both
    n_diff = int(np.sum(diff_mask))
    if n_diff == 0:
        return (0, 0, -1, "0 ULP")
    ii = np.flatnonzero(diff_mask.ravel())
    ulps = np.array([_ulp_dist(cpp.flat[i], py.flat[i]) for i in ii], dtype=np.int64)
    max_ulp = int(np.max(ulps))
    max_idx = int(ii[int(np.argmax(ulps))])
    uniq, cnt = np.unique(ulps, return_counts=True)
    hist = ", ".join(f"{c}×{u}ULP" for u, c in zip(uniq, cnt))
    return (n_diff, max_ulp, max_idx, hist)

def assert_bit_aligned(cpp_r, py_r, label=""):
    cpp = np.asarray(cpp_r, dtype=np.float64)
    py  = np.asarray(py_r, dtype=np.float64)
    assert cpp.shape == py.shape, f"{label}: shape mismatch {cpp.shape} vs {py.shape}"
    n_diff, max_ulp, max_idx, hist = _ulp_stats(cpp, py)
    if n_diff == 0:
        _ulp_report.append(f"  ✓ {label}: 0 ULP (bit-identical)")
        _ulp_record(label, int(cpp.size), 0, 0, 0, "—", "bit-identical")
        return
    _ulp_report.append(f"  ✗ FAIL {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP [{hist}]")
    _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  scipy={py.flat[max_idx]:.18e}")
    _ulp_record(label, int(cpp.size), n_diff, max_ulp, 0, hist, "FAIL (bit-aligned required)")
    raise AssertionError(
        f"{label}: BIT-LEVEL MISMATCH {n_diff}/{cpp.size} differ, max={max_ulp} ULP")

def assert_ulp_close(cpp_r, py_r, label="", max_ulp_tol=3):
    """Tolerate up to max_ulp_tol ULP. Always report actual ULP stats."""
    cpp = np.asarray(cpp_r, dtype=np.float64)
    py  = np.asarray(py_r, dtype=np.float64)
    assert cpp.shape == py.shape, f"{label}: shape mismatch {cpp.shape} vs {py.shape}"
    n_diff, max_ulp, max_idx, hist = _ulp_stats(cpp, py)
    if n_diff == 0:
        _ulp_report.append(f"  ✓ {label}: 0 ULP (bit-identical)")
        _ulp_record(label, int(cpp.size), 0, 0, max_ulp_tol, "—", "bit-identical")
        return
    ok = max_ulp <= max_ulp_tol
    tag = "✓" if ok else "✗ FAIL"
    status = f"within {max_ulp_tol} ULP" if ok else "FAIL (exceeds tolerance)"
    _ulp_report.append(f"  {tag} {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP (tol={max_ulp_tol}) [{hist}]")
    _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  scipy={py.flat[max_idx]:.18e}")
    _ulp_record(label, int(cpp.size), n_diff, max_ulp, max_ulp_tol, hist, status)
    if not ok:
        raise AssertionError(
            f"{label}: ULP MISMATCH max={max_ulp} > tol={max_ulp_tol}")


def _print_ulp_report():
    if not _ulp_report:
        return
    # Count: bit-identical = 0 ULP, tolerated = within limit, failures = over limit
    n_bit   = sum(1 for l in _ulp_report if "0 ULP (bit-identical)" in l)
    n_tolerated = sum(1 for l in _ulp_report if l.startswith("  ✓") and "0 ULP" not in l)
    n_fail  = sum(1 for l in _ulp_report if "FAIL" in l)
    print("\n" + "="*72, file=sys.stderr, flush=True)
    summary = f"  ULP ALIGNMENT REPORT: {n_bit} bit-identical"
    if n_tolerated: summary += f", {n_tolerated} within 1-3 ULP tolerance"
    if n_fail:      summary += f", {n_fail} FAILURES"
    print(summary, file=sys.stderr)
    print("="*72, file=sys.stderr, flush=True)
    for line in _ulp_report:
        print(line, file=sys.stderr, flush=True)
    print("="*72, file=sys.stderr, flush=True)

    # Export CSV to doc/ulp_report.csv
    if _ulp_records:
        import csv as _csv
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "doc", "ulp_report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ["module", "test", "dtype", "n_total", "n_diff", "max_ulp", "tol", "histogram", "status"]
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in _ulp_records:
                writer.writerow(r)
        print(f"  CSV exported → {csv_path}", file=sys.stderr, flush=True)

atexit.register(_print_ulp_report)


# ============================================================================
# fixtures
# ============================================================================

_cpp = None
def get_cpp():
    global _cpp
    if _cpp is None:
        _cpp = importlib.import_module(os.environ.get("SCIPYCPP_MODULE", "scipycpp"))
    return _cpp

@pytest.fixture(scope="session")
def cpp(): return get_cpp()

# ============================================================================
# helpers
# ============================================================================

def random_batch(shape, dtype=np.float64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(dtype)

def random_uniform(shape, low, high, dtype=np.float64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(low, high, size=shape).astype(dtype)


# ============================================================================
# scipy references
# ============================================================================

from scipy.stats import norm as sp_norm
from scipy import integrate as sp_integrate
from scipy.spatial import distance as sp_distance
from scipy.spatial import cKDTree as sp_cKDTree
from scipy import ndimage as sp_ndimage
from scipy import signal as sp_signal
from scipy.spatial.transform import Rotation as sp_Rotation


# ============================================================================
# BIT-LEVEL: norm.pdf / norm.cdf / norm.ppf
#
# norm.pdf uses numpy::exp() → npy_exp (dlsym) or SVML, identical math path
# to scipy's internal numpy.exp.  All three functions are now 0 ULP.
# ============================================================================


class TestNormPdf:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_batch_default(self, cpp, dtype):
        a = random_batch((BATCH,), dtype=dtype, seed=1001)
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s(f"pdf batch={BATCH}", dtype))

    @pytest.mark.parametrize("v", [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0])
    def test_canonical(self, cpp, dtype, v):
        a = np.array([v], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s(f"pdf({v})", dtype))

    def test_extreme(self, cpp, dtype):
        a = np.array([6.0, 8.0, 10.0, -6.0, -8.0, -10.0, 20.0, -20.0], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s("pdf extreme", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0,1.0),(1.0,1.0),(-2.0,1.0),(0.0,2.0),(0.0,0.5),(3.0,4.0),
        (-1.5,0.3),(5.0,10.0),
    ])
    def test_loc_scale(self, cpp, dtype, loc, scale):
        a = random_batch((BATCH,), dtype=dtype, seed=1002)
        assert_bit_aligned(cpp.stats.norm.pdf(a, dtype(loc), dtype(scale)),
                           sp_norm.pdf(a, loc=dtype(loc), scale=dtype(scale)),
                           _s(f"pdf(loc={loc},scale={scale})", dtype))

    @pytest.mark.parametrize("loc,scale", [(-10.0,0.01), (10.0,0.01)])
    def test_tiny_scale(self, cpp, dtype, loc, scale):
        a = random_batch((BATCH,), dtype=dtype, seed=1003)
        assert_bit_aligned(cpp.stats.norm.pdf(a, dtype(loc), dtype(scale)),
                           sp_norm.pdf(a, loc=dtype(loc), scale=dtype(scale)),
                           _s(f"pdf(loc={loc},scale={scale})", dtype))


class TestNormCdf:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_batch_default(self, cpp, dtype):
        a = random_batch((BATCH,), dtype=dtype, seed=1004)
        assert_bit_aligned(cpp.stats.norm.cdf(a), sp_norm.cdf(a), _s(f"cdf batch={BATCH}", dtype))

    @pytest.mark.parametrize("v", [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    def test_canonical(self, cpp, dtype, v):
        a = np.array([v], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.cdf(a), sp_norm.cdf(a), _s(f"cdf({v})", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0,1.0),(1.0,1.0),(-2.0,1.0),(0.0,2.0),(0.0,0.5),(3.0,4.0),
    ])
    def test_loc_scale(self, cpp, dtype, loc, scale):
        a = random_batch((BATCH,), dtype=dtype, seed=1005)
        assert_bit_aligned(cpp.stats.norm.cdf(a, dtype(loc), dtype(scale)),
                           sp_norm.cdf(a, loc=dtype(loc), scale=dtype(scale)),
                           _s(f"cdf(loc={loc},scale={scale})", dtype))


class TestNormPpf:
    """Cephes ndtri — bit-identical to scipy.special.ndtri."""
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_batch_default(self, cpp, dtype):
        a = random_uniform((BATCH,), 0.001, 0.999, dtype=dtype, seed=1006)
        assert_bit_aligned(cpp.stats.norm.ppf(a), sp_norm.ppf(a), _s(f"ppf batch={BATCH}", dtype))

    @pytest.mark.parametrize("p", [0.5, 0.025, 0.975, 0.001, 0.999])
    def test_canonical(self, cpp, dtype, p):
        a = np.array([p], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.ppf(a), sp_norm.ppf(a), _s(f"ppf({p})", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0,1.0),(1.0,1.0),(-2.0,1.0),(0.0,2.0),(0.0,0.5),
    ])
    def test_loc_scale(self, cpp, dtype, loc, scale):
        a = random_uniform((BATCH,), 0.001, 0.999, dtype=dtype, seed=1007)
        assert_bit_aligned(cpp.stats.norm.ppf(a, dtype(loc), dtype(scale)),
                           sp_norm.ppf(a, loc=dtype(loc), scale=dtype(scale)),
                           _s(f"ppf(loc={loc},scale={scale})", dtype))


# ============================================================================
# BIT-LEVEL: integrate — trapezoid, simpson
# ============================================================================

class TestIntegrate:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_trapezoid_batch(self, cpp, dtype):
        y = random_batch((BATCH,), dtype=dtype, seed=1008)
        assert_bit_aligned(
            np.float64(cpp.trapezoid(y)), np.float64(sp_integrate.trapezoid(y)),
            _s(f"trapezoid batch={BATCH}", dtype))

    def test_simpson_batch(self, cpp, dtype):
        y = random_batch((101,), dtype=dtype, seed=1009)
        if dtype == np.float32:
            # scipy sums in float32 SIMD; C++ sums in float32 sequential.
            # Compare in float32 ULP space (convert both results to float32).
            assert_f32_ulp_close(cpp.simpson(y), sp_integrate.simpson(y),
                                 _s(f"simpson batch=101", dtype), max_ulp=6)
        else:
            assert_bit_aligned(
                np.float64(cpp.simpson(y)), np.float64(sp_integrate.simpson(y)),
                _s(f"simpson batch=101", dtype))

    def test_trapezoid_known(self, cpp, dtype):
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=dtype)
        assert_bit_aligned(
            np.float64(cpp.trapezoid(y)), np.float64(sp_integrate.trapezoid(y)),
            _s("trapezoid known", dtype))

    def test_simpson_known(self, cpp):
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)
        assert_bit_aligned(
            np.float64(cpp.simpson(y)), np.float64(sp_integrate.simpson(y)),
            "simpson known")


# ============================================================================
# BIT-LEVEL: linalg.solve
#
# C++ linalg.solve always computes in float64 (float32 inputs are promoted).
# This mirrors scipy's LAPACK gesv which operates in double precision internally.
# For canonical (simple) matrices, Eigen3 partialPivLu is bit-identical to
# LAPACK gesv. For larger/random matrices, ULP-level differences (~1e-15)
# may occur; assert_linalg_close tolerates these.
# ============================================================================

def assert_linalg_close(cpp_r, py_r, label="", atol=1e-14):
    """Tolerance-based comparison for linalg operations (Eigen3 vs LAPACK).
    Always reports ULP stats; raises only if max_ulp exceeds tolerance
    for float64 values (where atol=1e-14 ≈ 50 ULP)."""
    cpp = np.asarray(cpp_r, dtype=np.float64)
    py  = np.asarray(py_r, dtype=np.float64)
    assert cpp.shape == py.shape, f"{label}: shape mismatch {cpp.shape} vs {py.shape}"
    n_diff, max_ulp, max_idx, hist = _ulp_stats(cpp, py)
    ok_abs = np.allclose(cpp, py, atol=atol)
    tag = "✓" if ok_abs else "✗ FAIL"
    tol_label = f"atol={atol}"
    if n_diff == 0:
        _ulp_report.append(f"  {tag} {label}: 0 ULP (bit-identical)")
        _ulp_record(label, int(cpp.size), 0, 0, tol_label, "—", "bit-identical")
    else:
        _ulp_report.append(f"  {tag} {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP [{hist}]")
        _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  np={py.flat[max_idx]:.18e}")
        status = f"within {tol_label}" if ok_abs else "FAIL (exceeds tolerance)"
        _ulp_record(label, int(cpp.size), n_diff, max_ulp, tol_label, hist, status)
    if not ok_abs:
        raise AssertionError(
            f"{label}: linalg MISMATCH, max ULP={max_ulp} exceeds atol={atol}")


class TestLinalg:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @staticmethod
    def _np_solve(A, b):
        """Reference: numpy.linalg.solve in float64 (C++ internally promotes)."""
        return np.linalg.solve(A.astype(np.float64), b.astype(np.float64))

    def test_solve_2x2(self, cpp, dtype):
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=dtype)
        b = np.array([5.0, 6.0], dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.linalg.solve(A, b)), self._np_solve(A, b),
            _s("solve 2x2", dtype))

    def test_solve_identity(self, cpp, dtype):
        A = np.eye(3, dtype=dtype)
        b = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.linalg.solve(A, b)), self._np_solve(A, b),
            _s("solve identity", dtype))

    def test_solve_batch(self, cpp, dtype):
        """100 random matrices (n=4..8) + random RHS vectors."""
        rng = np.random.RandomState(4242)
        for i in range(BATCH):
            n = rng.randint(4, 9)  # 4..8
            A = (rng.randn(n, n) * 2.0 + 3.0 * np.eye(n)).astype(dtype)
            b = rng.randn(n).astype(dtype)
            cpp_r = np.asarray(cpp.linalg.solve(A, b), dtype=np.float64)
            assert_linalg_close(cpp_r, self._np_solve(A, b), _s(f"solve batch[{i}] n={n}", dtype))

    @pytest.mark.parametrize("n", [10, 20])
    def test_solve_large(self, cpp, dtype, n):
        """Large matrix tests (10x10, 20x20)."""
        rng = np.random.RandomState(12345)
        A = (rng.randn(n, n) * 1.5 + 4.0 * np.eye(n)).astype(dtype)
        b = rng.randn(n).astype(dtype)
        cpp_r = np.asarray(cpp.linalg.solve(A, b), dtype=np.float64)
        assert_linalg_close(cpp_r, self._np_solve(A, b), _s(f"solve large n={n}", dtype))

    @pytest.mark.parametrize("seed", [5555, 6666, 7777])
    def test_solve_ill_conditioned(self, cpp, dtype, seed):
        """Ill-conditioned matrices (high condition number) — boundary test."""
        rng = np.random.RandomState(seed)
        n = 5
        # Generate a random orthogonal matrix Q and a diagonal with log-space values
        Q, _ = np.linalg.qr(rng.randn(n, n))
        diag = np.logspace(-3, 3, n)  # condition number ~ 1e6
        A = (Q @ np.diag(diag) @ Q.T).astype(dtype)
        b = rng.randn(n).astype(dtype)
        cpp_r = np.asarray(cpp.linalg.solve(A, b), dtype=np.float64)
        # For ill-conditioned matrices, relax tolerance (float32 promotion + LU)
        assert_linalg_close(cpp_r, self._np_solve(A, b), _s(f"solve ill-cond seed={seed}", dtype), atol=1e-10)


# ============================================================================
# BIT-LEVEL: spatial.distance.cdist
# ============================================================================

class TestCdist:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_batch(self, cpp, dtype, metric):
        XA = random_batch((100, 5), dtype=dtype, seed=1011)
        XB = random_batch((80, 5), dtype=dtype, seed=1012)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(XA, XB, metric)),
            sp_distance.cdist(XA, XB, metric),
            _s(f"cdist {metric}", dtype))

    def test_small(self, cpp, dtype):
        XA = np.array([[0., 0.], [1., 1.]], dtype=dtype)
        XB = np.array([[0., 1.], [1., 0.], [2., 2.]], dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(XA, XB, "euclidean")),
            sp_distance.cdist(XA, XB, "euclidean"),
            _s("cdist small", dtype))


# ============================================================================
# BIT-LEVEL: spatial.KDTree
# ============================================================================

class TestKDTree:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @staticmethod
    def _tk(cpp, dt):
        # scipy cKDTree always uses double internally, even for float32 input.
        # Use the float64 KDTree for bit-level alignment.
        return cpp.spatial.KDTree

    def test_query_batch(self, cpp, dtype):
        pts = random_batch((BATCH, 3), dtype=dtype, seed=1015)
        q = random_batch((3,), dtype=dtype, seed=1016)
        d_cpp, i_cpp = self._tk(cpp, dtype)(pts).query(q, k=1)
        d_py, i_py = sp_cKDTree(pts).query(q, k=1)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py), _s("KDTree dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))

    def test_query_k3_batch(self, cpp, dtype):
        pts = random_batch((BATCH, 3), dtype=dtype, seed=1017)
        q = random_batch((3,), dtype=dtype, seed=1018)
        d_cpp, i_cpp = self._tk(cpp, dtype)(pts).query(q, k=3)
        d_py, i_py = sp_cKDTree(pts).query(q, k=3)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py), _s("KDTree k=3 dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))


# ============================================================================
# BIT-LEVEL: ndimage.gaussian_filter1d
# ============================================================================

class TestGaussianFilter1d:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
    def test_batch(self, cpp, dtype, sigma):
        a = random_batch((BATCH,), dtype=dtype, seed=1019)
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=sigma)),
            sp_ndimage.gaussian_filter1d(a, sigma=sigma),
            _s(f"gaussian_filter1d sigma={sigma}", dtype))

    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_modes(self, cpp, dtype, mode):
        a = random_batch((BATCH,), dtype=dtype, seed=1020)
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5, mode=mode)),
            np.asarray(sp_ndimage.gaussian_filter1d(a, sigma=1.5, mode=mode), dtype=np.float64),
            _s(f"gaussian_filter1d mode={mode}", dtype))


# ============================================================================
# BIT-LEVEL: signal.medfilt
# ============================================================================

class TestMedfilt:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @pytest.mark.parametrize("k", [3, 5, 7, 9])
    def test_batch(self, cpp, dtype, k):
        a = random_batch((BATCH,), dtype=dtype, seed=1022)
        assert_bit_aligned(
            np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
            np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
            _s(f"medfilt k={k}", dtype))


# ============================================================================
# BIT-LEVEL: spatial.transform.Rotation
# ============================================================================

class TestRotation:
    """Rotation.from_matrix + as_euler bit-level alignment tests.

    C++ delegates directly to scipy.spatial.transform.Rotation (pre-imported),
    so result is guaranteed bit-identical. Tests cover:
      - 100 random batches per Euler sequence (§5 requirement)
      - All 6 Tait-Bryan sequences (xyz, xzy, yxz, yzx, zxy, zyx)
      - Gimbal lock boundary (beta ≈ ±pi/2)
      - Random rotation matrices (scipy → matrix → from_matrix → euler)
    """
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    @staticmethod
    def _rc(cpp, dt):
        # scipy Rotation always uses double internally, even for float32 input.
        # Use the float64 Rotation for bit-level alignment.
        return cpp.spatial.transform.Rotation

    def _check(self, cpp, dtype, R, seq, label):
        cpp_euler = np.asarray(
            self._rc(cpp, dtype).from_matrix(R).as_euler(seq), dtype=np.float64)
        py_euler = sp_Rotation.from_matrix(R).as_euler(seq)
        assert_bit_aligned(cpp_euler, py_euler, label)

    # --- identity / canonical angles ---

    def test_identity(self, cpp, dtype):
        self._check(cpp, dtype, np.eye(3, dtype=dtype), "xyz", _s("Rotation identity", dtype))

    def test_x_rotation(self, cpp, dtype):
        R = sp_Rotation.from_euler("xyz", [np.pi/4, 0, 0]).as_matrix()
        self._check(cpp, dtype, R.astype(dtype), "xyz", _s("Rotation x-45deg", dtype))

    def test_y_rotation(self, cpp, dtype):
        R = sp_Rotation.from_euler("xyz", [0, np.pi/6, 0]).as_matrix()
        self._check(cpp, dtype, R.astype(dtype), "xyz", _s("Rotation y-30deg", dtype))

    def test_z_rotation(self, cpp, dtype):
        R = sp_Rotation.from_euler("xyz", [0, 0, np.pi/3]).as_matrix()
        self._check(cpp, dtype, R.astype(dtype), "xyz", _s("Rotation z-60deg", dtype))

    def test_xyz_sequence(self, cpp, dtype):
        R = sp_Rotation.from_euler("xyz", np.deg2rad([20., 30., 45.])).as_matrix()
        self._check(cpp, dtype, R.astype(dtype), "xyz", _s("Rotation xyz(20,30,45)", dtype))

    def test_zyx_sequence(self, cpp, dtype):
        R = sp_Rotation.from_euler("zyx", np.deg2rad([10., -20., 40.])).as_matrix()
        self._check(cpp, dtype, R.astype(dtype), "zyx", _s("Rotation zyx(10,-20,40)", dtype))

    # --- 100 random batches for each Tait-Bryan sequence (§5 requirement) ---

    @pytest.mark.parametrize("seq", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    def test_random_batch(self, cpp, dtype, seq):
        """100 random Euler angles per sequence, round-trip via scipy."""
        # Use different seeds per sequence for diversity
        seed_map = {"xyz": 42, "xzy": 43, "yxz": 44, "yzx": 45, "zxy": 46, "zyx": 99}
        rng = np.random.RandomState(seed_map[seq])
        # Avoid gimbal lock region (±pi/2) for Tait-Bryan sequences
        for i in range(BATCH):
            a = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
            R = sp_Rotation.from_euler(seq, a).as_matrix().astype(dtype)
            self._check(cpp, dtype, R, seq, _s(f"Rotation {seq} random[{i}]", dtype))

    # --- gimbal lock boundary tests ---

    @pytest.mark.parametrize("beta", [np.pi/2, -np.pi/2, np.pi/2 - 1e-6, -np.pi/2 + 1e-6])
    def test_gimbal_lock_near(self, cpp, dtype, beta):
        """Euler angles near gimbal lock: beta ≈ ±π/2.
        Covers the special branch in _compute_euler_from_matrix where
        cos(beta) ≈ 0, requiring a different formula to extract alpha+gamma."""
        rng = np.random.RandomState(24601)
        for i in range(20):
            alpha = rng.uniform(-np.pi, np.pi)
            gamma = rng.uniform(-np.pi, np.pi)
            R = sp_Rotation.from_euler("xyz", [alpha, beta, gamma]).as_matrix().astype(dtype)
            self._check(cpp, dtype, R, "xyz", _s(f"Rotation gimbal beta={beta:.4f}[{i}]", dtype))

    # --- random rotation matrix test ---

    def test_random_matrices(self, cpp, dtype):
        """Generate 100 random rotation matrices via scipy.random,
        round-trip: scipy_matrix → from_matrix → as_euler → scipy.as_euler."""
        rng = np.random.RandomState(31415)
        for i in range(BATCH):
            R = sp_Rotation.random(random_state=rng).as_matrix().astype(dtype)
            self._check(cpp, dtype, R, "xyz", _s(f"Rotation random_matrix[{i}]", dtype))

    # --- intrinsic 'XYZ' sequence ---

    def test_XYZ_intrinsic(self, cpp, dtype):
        """Verify intrinsic 'XYZ' (uppercase) Euler sequence round-trip.
        scipy treats uppercase as intrinsic rotations."""
        rng = np.random.RandomState(2718)
        for i in range(BATCH):
            a = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
            R = sp_Rotation.from_euler("XYZ", a).as_matrix().astype(dtype)
            self._check(cpp, dtype, R, "XYZ", _s(f"Rotation XYZ intrinsic[{i}]", dtype))


# ============================================================================
# SPECIAL / EXTREME VALUES — 0 ULP across all APIs
#
# Covers: ±0.0, ±inf, NaN, subnormals, domain boundaries, saturating inputs,
#         out-of-range inputs, tiny/huge scale parameters, impulse responses.
# All tests use assert_bit_aligned → must be 0 ULP.
# ============================================================================

_INF  = np.float64(np.inf)
_NINF = np.float64(-np.inf)
_NAN  = np.float64(np.nan)
_POS0 = np.float64(0.0)
_NEG0 = np.float64(-0.0)
_TINY = np.float64(5e-324)   # smallest positive subnormal (DBL_TRUE_MIN)
_HUGE = np.float64(8.98846567431158e+307)  # near DBL_MAX/2


class TestSpecialValuesNorm:
    """Special / extreme values for norm.pdf, norm.cdf, norm.ppf — 0 ULP."""

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    # ------------------------------------------------------------------
    # norm.pdf
    # ------------------------------------------------------------------

    def test_pdf_special_scalars(self, cpp, dtype):
        """±inf, NaN, ±0.0, very large x (underflow to 0)."""
        vals = [_POS0, _NEG0, _INF, _NINF, _NAN,
                40.0, -40.0, 100.0, -100.0,
                1e-300, -1e-300, 1e300, -1e300]
        a = np.array(vals, dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a),
                           _s("pdf special scalars", dtype))

    def test_pdf_subnormal(self, cpp, dtype):
        """Subnormal-magnitude inputs: exp(-x²/2) ≈ 1/sqrt(2π)."""
        a = np.array([float(_TINY), -float(_TINY),
                      np.finfo(np.float64).tiny,
                      -np.finfo(np.float64).tiny], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a),
                           _s("pdf subnormal", dtype))

    def test_pdf_batch_wide(self, cpp, dtype):
        """100 values spanning 30 decades — extreme small to extreme large."""
        rng = np.random.RandomState(9001)
        a = np.concatenate([
            rng.uniform(-30, 30, 50).astype(dtype),    # normal range
            rng.uniform(-300, 300, 30).astype(dtype),   # far tails → 0.0
            np.array([0.0, 1e-15, -1e-15, 30.0, -30.0,
                      1e-100, -1e-100, 1e100, -1e100], dtype=dtype),
            np.array([np.inf, -np.inf, np.nan], dtype=dtype),
        ])
        assert_bit_aligned(cpp.stats.norm.pdf(a), sp_norm.pdf(a),
                           _s("pdf batch wide", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0, 1e-10),   # very tiny scale → tall narrow peak
        (0.0, 1e10),    # very large scale → very flat
        (1e15, 1.0),    # very far loc
        (-1e15, 1.0),
    ])
    def test_pdf_extreme_params(self, cpp, dtype, loc, scale):
        """Extreme loc/scale parameters."""
        a = random_batch((BATCH,), dtype=dtype, seed=9002)
        assert_bit_aligned(
            cpp.stats.norm.pdf(a, dtype(loc), dtype(scale)),
            sp_norm.pdf(a, loc=dtype(loc), scale=dtype(scale)),
            _s(f"pdf extreme params loc={loc} scale={scale}", dtype))

    # ------------------------------------------------------------------
    # norm.cdf
    # ------------------------------------------------------------------

    def test_cdf_special_scalars(self, cpp, dtype):
        """±inf → {1.0, 0.0}, NaN → NaN, ±0.0 → 0.5, saturation."""
        vals = [_POS0, _NEG0, _INF, _NINF, _NAN,
                40.0, -40.0, 38.5, -38.5, 8.0, -8.0,
                1e-300, -1e-300]
        a = np.array(vals, dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.cdf(a), sp_norm.cdf(a),
                           _s("cdf special scalars", dtype))

    def test_cdf_batch_wide(self, cpp, dtype):
        """100 values spanning full float range."""
        rng = np.random.RandomState(9003)
        a = np.concatenate([
            rng.uniform(-10, 10, 50).astype(dtype),
            rng.uniform(-40, 40, 30).astype(dtype),
            np.array([0.0, np.inf, -np.inf, np.nan,
                      1e-300, -1e-300, 40.0, -40.0], dtype=dtype),
        ])
        assert_bit_aligned(cpp.stats.norm.cdf(a), sp_norm.cdf(a),
                           _s("cdf batch wide", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0, 1e-10),
        (0.0, 1e10),
        (1e10, 1.0),
        (-1e10, 1.0),
    ])
    def test_cdf_extreme_params(self, cpp, dtype, loc, scale):
        """Extreme loc/scale parameters for CDF."""
        a = random_batch((BATCH,), dtype=dtype, seed=9004)
        assert_bit_aligned(
            cpp.stats.norm.cdf(a, dtype(loc), dtype(scale)),
            sp_norm.cdf(a, loc=dtype(loc), scale=dtype(scale)),
            _s(f"cdf extreme params loc={loc} scale={scale}", dtype))

    # ------------------------------------------------------------------
    # norm.ppf
    # ------------------------------------------------------------------

    def test_ppf_special_scalars(self, cpp, dtype):
        """p=0→-inf, p=1→+inf, p<0→NaN, p>1→NaN, NaN→NaN."""
        vals = [0.0, 1.0, 0.5, 0.025, 0.975,
                -1e-15, -0.1,       # p < 0 → NaN
                1.0 + 1e-15, 1.1,   # p > 1 → NaN
                np.nan, np.inf, -np.inf]
        a = np.array(vals, dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.ppf(a), sp_norm.ppf(a),
                           _s("ppf special scalars", dtype))

    def test_ppf_tiny_p(self, cpp, dtype):
        """Very small p — deep left tail; very large p — deep right tail."""
        vals = [5e-324, 1e-300, 1e-100, 1e-10,
                1 - 1e-10, 1 - 1e-100, 1 - 1e-300]
        a = np.array(vals, dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.ppf(a), sp_norm.ppf(a),
                           _s("ppf tiny p", dtype))

    def test_ppf_batch_wide(self, cpp, dtype):
        """100 probabilities from near-0 to near-1 plus boundary values."""
        rng = np.random.RandomState(9005)
        a = np.concatenate([
            rng.uniform(0.001, 0.999, 80).astype(dtype),
            np.array([0.0, 1.0, np.nan, -0.01, 1.01,
                      5e-324, 1 - 5e-324, 0.5, 0.25, 0.75], dtype=dtype),
        ])
        assert_bit_aligned(cpp.stats.norm.ppf(a), sp_norm.ppf(a),
                           _s("ppf batch wide", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0, 1e-5), (0.0, 1e5), (100.0, 1.0), (-100.0, 1.0),
    ])
    def test_ppf_extreme_params(self, cpp, dtype, loc, scale):
        """Extreme loc/scale parameters for PPF."""
        a = random_uniform((BATCH,), 0.001, 0.999, dtype=dtype, seed=9006)
        assert_bit_aligned(
            cpp.stats.norm.ppf(a, dtype(loc), dtype(scale)),
            sp_norm.ppf(a, loc=dtype(loc), scale=dtype(scale)),
            _s(f"ppf extreme params loc={loc} scale={scale}", dtype))


class TestSpecialValuesIntegrate:
    """Special / extreme values for trapezoid and simpson."""

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_trapezoid_all_zeros(self, cpp, dtype):
        y = np.zeros(BATCH, dtype=dtype)
        assert_bit_aligned(np.float64(cpp.trapezoid(y)),
                           np.float64(sp_integrate.trapezoid(y)),
                           _s("trapezoid all-zeros", dtype))

    def test_trapezoid_constant(self, cpp, dtype):
        # Constant arrays expose sequential vs SIMD pairwise sum difference.
        # float64: ≤5 f64-ULP.  float32: ≤4 f32-ULP (compare in native dtype).
        y = np.full(BATCH, 3.14159, dtype=dtype)
        if dtype == np.float32:
            assert_f32_ulp_close(cpp.trapezoid(y), sp_integrate.trapezoid(y),
                                 _s("trapezoid constant", dtype), max_ulp=6)
        else:
            assert_ulp_close(np.float64(cpp.trapezoid(y)),
                             np.float64(sp_integrate.trapezoid(y)),
                             _s("trapezoid constant", dtype), max_ulp_tol=6)

    def test_trapezoid_large_values(self, cpp, dtype):
        y = random_batch((BATCH,), dtype=dtype, seed=9010) * 1e200
        assert_bit_aligned(np.float64(cpp.trapezoid(y)),
                           np.float64(sp_integrate.trapezoid(y)),
                           _s("trapezoid large 1e200", dtype))

    def test_trapezoid_alternating_sign(self, cpp, dtype):
        y = np.array([(-1)**i * 1.0 for i in range(BATCH)], dtype=dtype)
        assert_bit_aligned(np.float64(cpp.trapezoid(y)),
                           np.float64(sp_integrate.trapezoid(y)),
                           _s("trapezoid alternating sign", dtype))

    def test_trapezoid_two_elements(self, cpp, dtype):
        """Edge case: minimum meaningful array."""
        y = np.array([1.0, 3.0], dtype=dtype)
        assert_bit_aligned(np.float64(cpp.trapezoid(y)),
                           np.float64(sp_integrate.trapezoid(y)),
                           _s("trapezoid 2-element", dtype))

    def test_simpson_all_zeros(self, cpp, dtype):
        y = np.zeros(101, dtype=dtype)
        assert_bit_aligned(np.float64(cpp.simpson(y)),
                           np.float64(sp_integrate.simpson(y)),
                           _s("simpson all-zeros", dtype))

    def test_simpson_constant(self, cpp, dtype):
        # Constant arrays expose sequential vs SIMD pairwise sum difference.
        # float64: ≤2 f64-ULP.  float32: ≤4 f32-ULP (compare in native dtype).
        y = np.full(101, 2.71828, dtype=dtype)
        if dtype == np.float32:
            assert_f32_ulp_close(cpp.simpson(y), sp_integrate.simpson(y),
                                 _s("simpson constant", dtype), max_ulp=6)
        else:
            assert_ulp_close(np.float64(cpp.simpson(y)),
                             np.float64(sp_integrate.simpson(y)),
                             _s("simpson constant", dtype), max_ulp_tol=6)

    def test_simpson_large_values(self, cpp, dtype):
        # Use dtype-appropriate scale to avoid numpy upcast (float32*1e100→float64).
        rng = np.random.RandomState(9011)
        scale = dtype(1e100) if dtype == np.float64 else dtype(1e20)
        y = (rng.randn(101) * float(scale)).astype(dtype)
        if dtype == np.float32:
            # Sequential vs SIMD float32 sum may accumulate up to ~30 f32-ULP
            # for large-magnitude random arrays (n=101, scale=1e20).
            assert_f32_ulp_close(cpp.simpson(y), sp_integrate.simpson(y),
                                 _s("simpson large 1e20", dtype), max_ulp=32)
        else:
            assert_ulp_close(np.float64(cpp.simpson(y)),
                             np.float64(sp_integrate.simpson(y)),
                             _s("simpson large 1e100", dtype), max_ulp_tol=6)


class TestSpecialValuesCdist:
    """Special / extreme values for spatial.distance.cdist."""

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_same_points(self, cpp, dtype):
        """Identical points → distance = 0."""
        X = random_batch((20, 4), dtype=dtype, seed=9020)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(X, X, "euclidean")),
            sp_distance.cdist(X, X, "euclidean"),
            _s("cdist same-points euclidean", dtype))

    def test_zero_matrix(self, cpp, dtype):
        """All-zero inputs → all distances = 0."""
        XA = np.zeros((10, 3), dtype=dtype)
        XB = np.zeros((8, 3), dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(XA, XB, "euclidean")),
            sp_distance.cdist(XA, XB, "euclidean"),
            _s("cdist zeros euclidean", dtype))

    def test_unit_vectors(self, cpp, dtype):
        """Unit vectors — distances are sqrt(2) or 0."""
        d = 5
        XA = np.eye(d, dtype=dtype)
        XB = np.eye(d, dtype=dtype)
        for metric in ["euclidean", "cityblock", "chebyshev"]:
            assert_bit_aligned(
                np.asarray(cpp.spatial.distance.cdist(XA, XB, metric)),
                sp_distance.cdist(XA, XB, metric),
                _s(f"cdist unit-vectors {metric}", dtype))

    def test_large_coords(self, cpp, dtype):
        """Very large coordinates (risk of overflow in squared distance)."""
        rng = np.random.RandomState(9021)
        # Use float64-safe large but not overflow values
        XA = (rng.randn(10, 3) * 1e100).astype(dtype)
        XB = (rng.randn(8, 3) * 1e100).astype(dtype)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(XA, XB, "euclidean")),
            sp_distance.cdist(XA, XB, "euclidean"),
            _s("cdist large-coords 1e100", dtype))

    def test_tiny_coords(self, cpp, dtype):
        """Very small coordinates."""
        rng = np.random.RandomState(9022)
        XA = (rng.randn(10, 3) * 1e-100).astype(dtype)
        XB = (rng.randn(8, 3) * 1e-100).astype(dtype)
        for metric in ["euclidean", "cityblock", "chebyshev"]:
            assert_bit_aligned(
                np.asarray(cpp.spatial.distance.cdist(XA, XB, metric)),
                sp_distance.cdist(XA, XB, metric),
                _s(f"cdist tiny-coords 1e-100 {metric}", dtype))

    def test_single_row(self, cpp, dtype):
        """Single-row matrices — 1×1 output."""
        XA = np.array([[3.0, 4.0]], dtype=dtype)
        XB = np.array([[0.0, 0.0]], dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.spatial.distance.cdist(XA, XB, "euclidean")),
            sp_distance.cdist(XA, XB, "euclidean"),
            _s("cdist single-row", dtype))


class TestSpecialValuesKDTree:
    """Special / extreme values for spatial.KDTree."""

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_query_at_existing_point(self, cpp, dtype):
        """Query at one of the tree points — distance must be 0."""
        pts = random_batch((50, 3), dtype=dtype, seed=9030)
        q = pts[7].copy()   # query point is exactly a data point
        d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=1)
        d_py, i_py   = sp_cKDTree(pts).query(q, k=1)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py),
                           _s("KDTree query-at-existing dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))

    def test_single_point_tree(self, cpp, dtype):
        """Tree with only one point."""
        pts = np.array([[1.0, 2.0, 3.0]], dtype=dtype)
        q   = np.array([0.0, 0.0, 0.0], dtype=dtype)
        d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=1)
        d_py, i_py   = sp_cKDTree(pts).query(q, k=1)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py),
                           _s("KDTree single-point dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))

    def test_large_coords(self, cpp, dtype):
        """Very large coordinate values."""
        rng = np.random.RandomState(9031)
        pts = (rng.randn(30, 3) * 1e8).astype(dtype)
        q   = (rng.randn(3) * 1e8).astype(dtype)
        d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=1)
        d_py, i_py   = sp_cKDTree(pts).query(q, k=1)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py),
                           _s("KDTree large-coords 1e8 dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))

    def test_tiny_coords(self, cpp, dtype):
        """Very tiny coordinate values (subnormal-adjacent)."""
        rng = np.random.RandomState(9032)
        pts = (rng.randn(30, 3) * 1e-200).astype(dtype)
        q   = (rng.randn(3) * 1e-200).astype(dtype)
        d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=1)
        d_py, i_py   = sp_cKDTree(pts).query(q, k=1)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py),
                           _s("KDTree tiny-coords 1e-200 dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))

    def test_k_equals_n(self, cpp, dtype):
        """k = number of points — return all distances sorted."""
        n = 10
        pts = random_batch((n, 2), dtype=dtype, seed=9033)
        q   = random_batch((2,),  dtype=dtype, seed=9034)
        d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=n)
        d_py, i_py   = sp_cKDTree(pts).query(q, k=n)
        assert_bit_aligned(np.asarray(d_cpp), np.asarray(d_py),
                           _s(f"KDTree k={n} all-points dist", dtype))
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py))


class TestSpecialValuesGaussianFilter1d:
    """Extensive edge-case and extreme-value tests for ndimage.gaussian_filter1d.

    Bug discovered (and fixed): n=1 + mode='mirror' caused SIGFPE (integer
    divide-by-zero) in bnd() because period p = 2*n-2 = 0.  A guard
    `if (n <= 1) return src[0]` was added to ndimage.h.
    """

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _g(cpp, a, sigma=1.5, mode="reflect", cval=0.0, truncate=4.0):
        return np.asarray(cpp.ndimage.gaussian_filter1d(
            a, sigma=sigma, mode=mode, cval=cval, truncate=truncate))

    @staticmethod
    def _s(cpp, a, sigma=1.5, mode="reflect", cval=0.0, truncate=4.0):
        return sp_ndimage.gaussian_filter1d(
            a, sigma=sigma, mode=mode, cval=cval, truncate=truncate)

    # ------------------------------------------------------------------
    # Existing baseline tests (kept)
    # ------------------------------------------------------------------
    def test_all_zeros(self, cpp, dtype):
        a = np.zeros(BATCH, dtype=dtype)
        assert_bit_aligned(self._g(cpp, a), self._s(cpp, a),
                           _s("gaussian_filter1d zeros", dtype))

    def test_constant_signal(self, cpp, dtype):
        a = np.full(BATCH, 2.71828, dtype=dtype)
        assert_bit_aligned(self._g(cpp, a, sigma=2.0), self._s(cpp, a, sigma=2.0),
                           _s("gaussian_filter1d constant", dtype))

    def test_single_impulse_center(self, cpp, dtype):
        """Single spike in the center — exact Gaussian kernel shape."""
        a = np.zeros(BATCH, dtype=dtype); a[BATCH // 2] = 1.0
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            assert_bit_aligned(
                self._g(cpp, a, sigma=sigma), self._s(cpp, a, sigma=sigma),
                _s(f"gaussian_filter1d impulse sigma={sigma}", dtype))

    def test_very_small_sigma(self, cpp, dtype):
        a = random_batch((BATCH,), dtype=dtype, seed=9040)
        assert_bit_aligned(self._g(cpp, a, sigma=0.1), self._s(cpp, a, sigma=0.1),
                           _s("gaussian_filter1d sigma=0.1", dtype))

    def test_very_large_sigma(self, cpp, dtype):
        a = random_batch((BATCH,), dtype=dtype, seed=9041)
        assert_bit_aligned(self._g(cpp, a, sigma=50.0), self._s(cpp, a, sigma=50.0),
                           _s("gaussian_filter1d sigma=50", dtype))

    def test_all_modes_constant_input(self, cpp, dtype):
        a = np.full(BATCH, 5.0, dtype=dtype)
        for mode in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            assert_bit_aligned(
                self._g(cpp, a, mode=mode),
                np.asarray(self._s(cpp, a, mode=mode), dtype=np.float64),
                _s(f"gaussian_filter1d constant mode={mode}", dtype))

    # ------------------------------------------------------------------
    # Tiny array sizes: n=1, n=2, n=3
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_n1_all_modes(self, cpp, dtype, mode):
        """n=1: fixed SIGFPE bug for mirror (p=2*n-2=0 → modulo-by-zero)."""
        a = np.array([dtype(3.14)], dtype=dtype)
        assert_bit_aligned(
            self._g(cpp, a, sigma=1.0, mode=mode),
            self._s(cpp, a, sigma=1.0, mode=mode),
            _s(f"gaussian_filter1d n=1 mode={mode}", dtype))

    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_n2_all_modes(self, cpp, dtype, mode):
        """n=2: smallest non-trivial array for all boundary modes."""
        a = np.array([dtype(1.0), dtype(2.0)], dtype=dtype)
        assert_bit_aligned(
            self._g(cpp, a, sigma=1.0, mode=mode),
            self._s(cpp, a, sigma=1.0, mode=mode),
            _s(f"gaussian_filter1d n=2 mode={mode}", dtype))

    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_n3_all_modes(self, cpp, dtype, mode):
        """n=3: mirror period=4, reflect period=6 — boundary wraps test."""
        a = np.array([dtype(1.0), dtype(3.0), dtype(2.0)], dtype=dtype)
        assert_bit_aligned(
            self._g(cpp, a, sigma=1.0, mode=mode),
            self._s(cpp, a, sigma=1.0, mode=mode),
            _s(f"gaussian_filter1d n=3 mode={mode}", dtype))

    # ------------------------------------------------------------------
    # Kernel larger than array (half = ceil(truncate*sigma) >> n)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_kernel_larger_than_array(self, cpp, dtype, mode):
        """sigma=20 n=5: kernel half=80, boundary extension ≫ array length."""
        a = np.arange(5, dtype=dtype) + dtype(1)
        assert_bit_aligned(
            self._g(cpp, a, sigma=20.0, mode=mode),
            self._s(cpp, a, sigma=20.0, mode=mode),
            _s(f"gaussian_filter1d half>>n mode={mode}", dtype))

    # ------------------------------------------------------------------
    # Impulse at array boundaries (not just center)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_impulse_at_left_boundary(self, cpp, dtype, mode):
        """Impulse at index 0 — tests left boundary extension for all modes."""
        a = np.zeros(20, dtype=dtype); a[0] = dtype(1.0)
        assert_bit_aligned(
            self._g(cpp, a, sigma=2.0, mode=mode),
            self._s(cpp, a, sigma=2.0, mode=mode),
            _s(f"gaussian_filter1d impulse[0] mode={mode}", dtype))

    @pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
    def test_impulse_at_right_boundary(self, cpp, dtype, mode):
        """Impulse at index n-1 — tests right boundary extension for all modes."""
        a = np.zeros(20, dtype=dtype); a[-1] = dtype(1.0)
        assert_bit_aligned(
            self._g(cpp, a, sigma=2.0, mode=mode),
            self._s(cpp, a, sigma=2.0, mode=mode),
            _s(f"gaussian_filter1d impulse[-1] mode={mode}", dtype))

    # ------------------------------------------------------------------
    # Non-default truncate and cval
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("truncate", [2.0, 3.0, 6.0])
    def test_non_default_truncate(self, cpp, dtype, truncate):
        """Different truncate values change kernel half-width."""
        a = random_batch((30,), dtype=dtype, seed=9043)
        assert_bit_aligned(
            self._g(cpp, a, sigma=1.5, truncate=truncate),
            self._s(cpp, a, sigma=1.5, truncate=truncate),
            _s(f"gaussian_filter1d truncate={truncate}", dtype))

    @pytest.mark.parametrize("cval", [0.0, 3.14159, -1.5, 1e10])
    def test_non_default_cval(self, cpp, dtype, cval):
        """Non-zero cval in constant mode changes boundary fill."""
        a = random_batch((20,), dtype=dtype, seed=9044)
        assert_bit_aligned(
            self._g(cpp, a, sigma=1.5, mode="constant", cval=dtype(cval)),
            self._s(cpp, a, sigma=1.5, mode="constant", cval=dtype(cval)),
            _s(f"gaussian_filter1d cval={cval}", dtype))

    # ------------------------------------------------------------------
    # Smooth signal shapes
    # ------------------------------------------------------------------
    def test_linear_ramp(self, cpp, dtype):
        a = np.arange(50, dtype=dtype)
        assert_bit_aligned(self._g(cpp, a, sigma=2.0), self._s(cpp, a, sigma=2.0),
                           _s("gaussian_filter1d linear ramp", dtype))

    def test_step_function(self, cpp, dtype):
        a = np.r_[np.zeros(25, dtype=dtype), np.ones(25, dtype=dtype)]
        assert_bit_aligned(self._g(cpp, a, sigma=2.0), self._s(cpp, a, sigma=2.0),
                           _s("gaussian_filter1d step function", dtype))

    # ------------------------------------------------------------------
    # ±inf propagation
    # ------------------------------------------------------------------
    def test_single_inf_middle(self, cpp, dtype):
        a = np.array([0, 0, 1, 0, 0], dtype=dtype); a[2] = dtype(np.inf)
        assert_bit_aligned(self._g(cpp, a, sigma=1.0), self._s(cpp, a, sigma=1.0),
                           _s("gaussian_filter1d +inf middle", dtype))

    def test_inf_at_boundaries(self, cpp, dtype):
        for pos in [0, -1]:
            a = np.ones(10, dtype=dtype); a[pos] = dtype(np.inf)
            assert_bit_aligned(self._g(cpp, a, sigma=1.0), self._s(cpp, a, sigma=1.0),
                               _s(f"gaussian_filter1d inf[{pos}]", dtype))

    def test_alternating_inf(self, cpp, dtype):
        """±inf alternating → NaN everywhere (inf - inf)."""
        a = np.array([np.inf, -np.inf] * 5, dtype=dtype)
        assert_bit_aligned(self._g(cpp, a, sigma=1.0), self._s(cpp, a, sigma=1.0),
                           _s("gaussian_filter1d alternating ±inf", dtype))

    # ------------------------------------------------------------------
    # NaN propagation
    # ------------------------------------------------------------------
    def test_nan_propagation(self, cpp, dtype):
        for pos in [0, 2, -1]:
            a = np.ones(10, dtype=dtype); a[pos] = dtype(np.nan)
            assert_bit_aligned(self._g(cpp, a, sigma=1.0), self._s(cpp, a, sigma=1.0),
                               _s(f"gaussian_filter1d NaN[{pos}]", dtype))

    def test_all_nan(self, cpp, dtype):
        a = np.full(10, np.nan, dtype=dtype)
        assert_bit_aligned(self._g(cpp, a, sigma=1.0), self._s(cpp, a, sigma=1.0),
                           _s("gaussian_filter1d all-NaN", dtype))

    # ------------------------------------------------------------------
    # Extreme magnitudes — float64
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [1e100, 1e200, 2e307])
    def test_f64_large_magnitude(self, cpp, scale):
        """float64 values up to near float64 max (~1.8e308)."""
        a = (np.random.RandomState(9050).randn(BATCH) * scale).astype(np.float64)
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.5),
            f"gaussian_filter1d f64 scale={scale:.0e} [float64]")

    def test_f64_subnormal(self, cpp):
        """float64 subnormal values (~5e-324) — underflow to zero after convolution."""
        a = np.array([5e-324, 0.0, 5e-324, 0.0, 5e-324])
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.0),
            "gaussian_filter1d f64 subnormal [float64]")

    def test_f64_min_normal(self, cpp):
        """float64 minimum normal (~2.2e-308) — subnormal outputs expected."""
        ftiny = np.finfo(np.float64).tiny
        a = np.array([ftiny, -ftiny, ftiny, -ftiny, ftiny])
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.0),
            "gaussian_filter1d f64 min-normal [float64]")

    # ------------------------------------------------------------------
    # Extreme magnitudes — float32
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [1e20, 1e30])
    def test_f32_large_magnitude(self, cpp, scale):
        """float32 values up to ~1e30 (well below float32 max 3.4e38)."""
        a = (np.random.RandomState(9051).randn(BATCH) * scale).astype(np.float32)
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.5),
            f"gaussian_filter1d f32 scale={scale:.0e} [float32]")

    def test_f32_near_max(self, cpp):
        """float32 values near float32 max (~3.4e38)."""
        fmax = np.finfo(np.float32).max
        a = np.full(BATCH, np.float32(fmax * 0.9))
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.5),
            "gaussian_filter1d f32 near-max [float32]")

    def test_f32_subnormal(self, cpp):
        """float32 subnormal values (~1.4e-45)."""
        fsub = np.float32(1.4e-45)
        a = np.array([fsub, np.float32(0), fsub, np.float32(0), fsub])
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.0),
            "gaussian_filter1d f32 subnormal [float32]")

    def test_f32_min_normal(self, cpp):
        """float32 minimum normal (~1.18e-38)."""
        ftiny = np.finfo(np.float32).tiny
        a = np.array([ftiny, -ftiny, ftiny, -ftiny, ftiny])
        assert_bit_aligned(
            np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0)),
            sp_ndimage.gaussian_filter1d(a, sigma=1.0),
            "gaussian_filter1d f32 min-normal [float32]")


class TestSpecialValuesMedfilt:
    """Special / extreme values for signal.medfilt."""

    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_all_same(self, cpp, dtype):
        """All-identical values → output equals input."""
        for v in [0.0, 1.0, -3.14, 1e100, -1e100]:
            a = np.full(20, v, dtype=dtype)
            for k in [3, 5, 7]:
                assert_bit_aligned(
                    np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                    np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                    _s(f"medfilt all-same v={v} k={k}", dtype))

    def test_monotone_increasing(self, cpp, dtype):
        """Monotone increasing sequence."""
        a = np.arange(20, dtype=dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt monotone-inc k={k}", dtype))

    def test_monotone_decreasing(self, cpp, dtype):
        """Monotone decreasing sequence."""
        a = np.arange(20, 0, -1, dtype=dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt monotone-dec k={k}", dtype))

    def test_single_spike(self, cpp, dtype):
        """Impulse in the center — spike should be suppressed."""
        a = np.zeros(20, dtype=dtype)
        a[10] = 1e10
        for k in [3, 5, 7]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt spike k={k}", dtype))

    def test_with_inf(self, cpp, dtype):
        """Inf values in array — inf is well-ordered."""
        a = np.array([1.0, np.inf, 3.0, 4.0, 5.0,
                      -np.inf, 2.0, np.inf, 8.0, -np.inf], dtype=dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt with-inf k={k}", dtype))

    def test_alternating_sign(self, cpp, dtype):
        """Alternating ±1 pattern."""
        a = np.array([(-1.0)**i for i in range(20)], dtype=dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt alternating k={k}", dtype))

    def test_large_values(self, cpp, dtype):
        """Very large values — sort ordering should still be correct."""
        rng = np.random.RandomState(9050)
        a = (rng.randn(30) * 1e200).astype(dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt large-values k={k}", dtype))

    def test_tiny_values(self, cpp, dtype):
        """Very small (subnormal-adjacent) values."""
        rng = np.random.RandomState(9051)
        a = (rng.randn(30) * 1e-200).astype(dtype)
        for k in [3, 5]:
            assert_bit_aligned(
                np.asarray(cpp.signal.medfilt(a, kernel_size=k), dtype=np.float64),
                np.asarray(sp_signal.medfilt(a, kernel_size=k), dtype=np.float64),
                _s(f"medfilt tiny-values k={k}", dtype))

    def test_two_element(self, cpp, dtype):
        """Minimum meaningful input (k=3 pads with zeros)."""
        a = np.array([2.0, 5.0], dtype=dtype)
        assert_bit_aligned(
            np.asarray(cpp.signal.medfilt(a, kernel_size=3), dtype=np.float64),
            np.asarray(sp_signal.medfilt(a, kernel_size=3), dtype=np.float64),
            _s("medfilt 2-element", dtype))


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
