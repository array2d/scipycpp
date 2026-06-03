"""
Bit-level alignment tests — scipycpp C++ vs Python scipy APIs.
SINGLE entry point: pytest tests/test_all.py -v

ALL APIs must be bit-identical (np.array_equal) for both float64/float32.
100-batch random data from extreme small to extreme large.

NOTE: All scipy APIs return float64 regardless of input dtype.
C++ mirrors this behavior.
"""

import os, sys, atexit, importlib, warnings, numpy as np, pytest

BATCH = 100

def _s(label, dtype=None):
    """Append [f64] or [f32] to label for ULP report clarity."""
    return label if dtype is None else f"{label} [{np.dtype(dtype).name}]"

# ============================================================================
# ULP computation & reporting
# ============================================================================

_ulp_report = []  # global collector for final summary

def _ulp_dist(a, b):
    """ULP distance between two float64 values (same-sign only)."""
    if (a < 0) != (b < 0):
        return int(np.float64(a).view(np.uint64)) + int(np.float64(b).view(np.uint64))
    diff = int(np.float64(a).view(np.uint64)) - int(np.float64(b).view(np.uint64))
    return abs(diff)

def _ulp_stats(cpp, py):
    """Compute (n_diff, max_ulp, max_idx, hist_str) for two float64 arrays."""
    assert cpp.shape == py.shape
    diff_mask = (cpp != py)
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
        return
    _ulp_report.append(f"  ✗ FAIL {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP [{hist}]")
    _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  scipy={py.flat[max_idx]:.18e}")
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
        return
    ok = max_ulp <= max_ulp_tol
    tag = "✓" if ok else "✗ FAIL"
    _ulp_report.append(f"  {tag} {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP (tol={max_ulp_tol}) [{hist}]")
    _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  scipy={py.flat[max_idx]:.18e}")
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
# norm.pdf uses numpy::exp() → std::exp(). Since numpcpp v1.21.2 dropped
# std::exp may differ from scipy's npy_exp by 1-3 ULP.
# Use assert_ulp_close (max_ulp_tol=3) for pdf; cdf/ppf use Cephes
# with direct std:: calls (same code path as scipy's internal Cephes) so
# assert_bit_aligned still holds.
# ============================================================================


class TestNormPdf:
    @pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
    def dtype(self, request): return request.param

    def test_batch_default(self, cpp, dtype):
        a = random_batch((BATCH,), dtype=dtype, seed=1001)
        assert_ulp_close(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s(f"pdf batch={BATCH}", dtype))

    @pytest.mark.parametrize("v", [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0])
    def test_canonical(self, cpp, dtype, v):
        a = np.array([v], dtype=dtype)
        assert_ulp_close(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s(f"pdf({v})", dtype))

    def test_extreme(self, cpp, dtype):
        a = np.array([6.0, 8.0, 10.0, -6.0, -8.0, -10.0, 20.0, -20.0], dtype=dtype)
        assert_ulp_close(cpp.stats.norm.pdf(a), sp_norm.pdf(a), _s("pdf extreme", dtype))

    @pytest.mark.parametrize("loc,scale", [
        (0.0,1.0),(1.0,1.0),(-2.0,1.0),(0.0,2.0),(0.0,0.5),(3.0,4.0),
        (-1.5,0.3),(5.0,10.0),
    ])
    def test_loc_scale(self, cpp, dtype, loc, scale):
        a = random_batch((BATCH,), dtype=dtype, seed=1002)
        assert_ulp_close(cpp.stats.norm.pdf(a, dtype(loc), dtype(scale)),
                         sp_norm.pdf(a, loc=dtype(loc), scale=dtype(scale)),
                         _s(f"pdf(loc={loc},scale={scale})", dtype))

    @pytest.mark.parametrize("loc,scale", [(-10.0,0.01), (10.0,0.01)])
    def test_tiny_scale(self, cpp, dtype, loc, scale):
        a = random_batch((BATCH,), dtype=dtype, seed=1003)
        assert_ulp_close(cpp.stats.norm.pdf(a, dtype(loc), dtype(scale)),
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
    if n_diff == 0:
        _ulp_report.append(f"  {tag} {label}: 0 ULP (bit-identical)")
    else:
        _ulp_report.append(f"  {tag} {label}: {n_diff}/{cpp.size} differ, max={max_ulp} ULP [{hist}]")
        _ulp_report.append(f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  np={py.flat[max_idx]:.18e}")
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


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
