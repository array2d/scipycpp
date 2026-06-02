"""
Bit-level alignment tests — scipycpp C++ vs Python scipy APIs.

SINGLE entry point: pytest tests/test_all.py -v

Coverage:
    - scipy.stats.norm.pdf: float64 + float32
    - scipy.integrate: trapezoid, simpson
    - scipy.linalg.solve
    - scipy.spatial.distance.cdist
    - scipy.spatial.KDTree (query with distances)
    - scipy.ndimage.gaussian_filter1d
    - scipy.signal.medfilt
    - scipy.spatial.transform.Rotation (from_matrix, as_euler)
"""

import os, importlib, numpy as np, pytest

def check_bit_aligned(cpp_result, py_result, label=""):
    cpp, py = np.asarray(cpp_result), np.asarray(py_result)
    info = {"label": label, "pass": False, "shape_match": cpp.shape == py.shape, "n_diff": 0, "error": None}
    if not info["shape_match"]:
        info["error"] = f"shape mismatch: C++ {cpp.shape} vs Python {py.shape}"
        return info
    if np.array_equal(cpp, py): info["pass"] = True; return info

    diff_mask = cpp != py
    info["n_diff"] = int(np.sum(diff_mask))
    diff_indices = np.flatnonzero(diff_mask.ravel())[:5]
    lines = [f"BIT-LEVEL MISMATCH: {info['n_diff']}/{cpp.size} elements differ"]
    for idx in diff_indices:
        cv, pv = cpp.flat[idx], py.flat[idx]
        lines.append(f"  [{idx}] C++={cv} vs scipy={pv}")
    info["error"] = "\n".join(lines)
    return info

def assert_bit_aligned(cpp_result, py_result, label=""):
    info = check_bit_aligned(cpp_result, py_result, label)
    if not info["pass"]: raise AssertionError(info.get("error", "bit-level alignment failure"))

def assert_approx(cpp_result, py_result, label="", rtol=1e-10):
    cpp, py = np.asarray(cpp_result), np.asarray(py_result)
    if not np.allclose(cpp, py, rtol=rtol):
        raise AssertionError(f"{label}: C++ {cpp} vs scipy {py}")

def random_array(shape, dtype=np.float64, seed=42):
    rng = np.random.RandomState(seed + hash(shape) % (2**31))
    return rng.randn(*shape).astype(dtype)

# C++ module
_cpp = None
def get_cpp():
    global _cpp
    if _cpp is None: _cpp = importlib.import_module(os.environ.get("SCIPYCPP_MODULE", "scipycpp"))
    return _cpp

@pytest.fixture(scope="session")
def cpp(): return get_cpp()

@pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request): return request.param

from scipy.stats import norm as sp_norm
from scipy import integrate as sp_integrate
from scipy.spatial import distance as sp_distance
from scipy.spatial import cKDTree as sp_cKDTree
from scipy import ndimage as sp_ndimage
from scipy import signal as sp_signal
from scipy.spatial.transform import Rotation as sp_Rotation

def ref_norm_pdf(x, **kw):
    r = sp_norm.pdf(x, **kw)
    return r.astype(np.float32) if x.dtype == np.float32 else r

# ============================================================================
# stats.norm.pdf tests
# ============================================================================

class TestNormPdfDefault:
    @pytest.mark.parametrize("shape", [(10,), (3,4), (2,3,4)])
    def test_shapes(self, cpp, dtype, shape):
        a = random_array(shape, dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), ref_norm_pdf(a), f"norm.pdf{shape}")

    def test_zero(self, cpp, dtype):
        a = np.array([0.0], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), ref_norm_pdf(a), "norm.pdf 0")

    @pytest.mark.parametrize("v", [1.0, 2.0, -3.0, 0.5, -0.5])
    def test_scalar(self, cpp, dtype, v):
        a = np.array([v], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), ref_norm_pdf(a), f"norm.pdf x={v}")

    def test_extreme(self, cpp, dtype):
        a = np.array([6.0, 8.0, 10.0, -6.0, -8.0, -10.0], dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a), ref_norm_pdf(a), "norm.pdf extreme")

class TestNormPdfLocScale:
    @pytest.mark.parametrize("loc,scale", [(0.0,1.0),(1.0,1.0),(-2.0,1.0),(0.0,2.0),(0.0,0.5),(3.0,4.0)])
    def test_loc_scale(self, cpp, dtype, loc, scale):
        _l, _s = dtype(loc), dtype(scale)
        a = random_array((20,), dtype=dtype)
        assert_bit_aligned(cpp.stats.norm.pdf(a, _l, _s),
                           ref_norm_pdf(a, loc=_l, scale=_s),
                           f"norm.pdf(loc={loc},scale={scale})")

# ============================================================================
# integrate tests
# ============================================================================

class TestIntegrate:
    def test_trapezoid(self, cpp):
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        cpp_r = cpp.trapezoid(y)
        py_r = sp_integrate.trapezoid(y)
        assert np.isclose(cpp_r, py_r), f"trapezoid: {cpp_r} vs {py_r}"

    def test_simpson(self, cpp):
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        cpp_r = cpp.simpson(y)
        py_r = sp_integrate.simpson(y)
        assert np.isclose(cpp_r, py_r), f"simpson: {cpp_r} vs {py_r}"

    def test_trapezoid_random(self, cpp):
        y = random_array((100,))
        cpp_r = cpp.trapezoid(y)
        py_r = sp_integrate.trapezoid(y)
        assert np.isclose(cpp_r, py_r, rtol=1e-12)

# ============================================================================
# linalg tests
# ============================================================================

class TestLinalg:
    def test_solve_2x2(self, cpp):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 6.0])
        x = cpp.linalg.solve(A, b)
        expected = np.linalg.solve(A, b)
        assert np.allclose(x, expected), f"solve: {x} vs {expected}"

# ============================================================================
# spatial.distance.cdist tests
# ============================================================================

class TestCdist:
    @pytest.mark.parametrize("mA,mB,dim", [(3, 2, 2), (5, 4, 3), (10, 8, 5)])
    def test_cdist_euclidean(self, cpp, mA, mB, dim):
        XA = random_array((mA, dim), seed=100)
        XB = random_array((mB, dim), seed=200)
        cpp_r = np.asarray(cpp.spatial.distance.cdist(XA, XB, "euclidean"))
        py_r  = sp_distance.cdist(XA, XB, "euclidean")
        assert_approx(cpp_r, py_r, f"cdist({mA}x{mB},{dim})")

    def test_cdist_cityblock(self, cpp):
        XA = np.array([[1.0, 2.0], [3.0, 4.0]])
        XB = np.array([[2.0, 3.0], [4.0, 5.0]])
        cpp_r = np.asarray(cpp.spatial.distance.cdist(XA, XB, "cityblock"))
        py_r  = sp_distance.cdist(XA, XB, "cityblock")
        assert_approx(cpp_r, py_r, "cdist cityblock")

    def test_cdist_chebyshev(self, cpp):
        XA = random_array((4, 3), seed=300)
        XB = random_array((3, 3), seed=400)
        cpp_r = np.asarray(cpp.spatial.distance.cdist(XA, XB, "chebyshev"))
        py_r  = sp_distance.cdist(XA, XB, "chebyshev")
        assert_approx(cpp_r, py_r, "cdist chebyshev")

# ============================================================================
# spatial.KDTree tests
# ============================================================================

class TestKDTree:
    def test_query_basic(self, cpp):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        q   = np.array([0.5, 0.5])
        cpp_tree = cpp.spatial.KDTree(pts)
        py_tree  = sp_cKDTree(pts)
        d_cpp, i_cpp = cpp_tree.query(q, k=1)
        d_py,  i_py  = py_tree.query(q, k=1)
        assert_approx(np.asarray(d_cpp), np.asarray(d_py), "KDTree query dist")
        assert_approx(np.asarray(i_cpp).astype(float), np.asarray(i_py).astype(float), "KDTree query idx")

    def test_query_k(self, cpp):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [10.0, 10.0]])
        q   = np.array([1.5, 1.5])
        cpp_tree = cpp.spatial.KDTree(pts)
        py_tree  = sp_cKDTree(pts)
        d_cpp, i_cpp = cpp_tree.query(q, k=3)
        d_py,  i_py  = py_tree.query(q, k=3)
        assert_approx(np.asarray(d_cpp), np.asarray(d_py), "KDTree k=3 dist")
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py), err_msg="KDTree k=3 idx")

    def test_query_random(self, cpp):
        pts = random_array((100, 3), seed=500)
        q   = random_array((3,), seed=600)
        cpp_tree = cpp.spatial.KDTree(pts)
        py_tree  = sp_cKDTree(pts)
        d_cpp, i_cpp = cpp_tree.query(q, k=1)
        d_py,  i_py  = py_tree.query(q, k=1)
        assert_approx(np.asarray(d_cpp), np.asarray(d_py), "KDTree random dist")
        np.testing.assert_array_equal(np.asarray(i_cpp), np.asarray(i_py), err_msg="KDTree random idx")

# ============================================================================
# ndimage.gaussian_filter1d tests
# ============================================================================

class TestGaussianFilter1d:
    def test_basic(self, cpp):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0))
        py_r  = sp_ndimage.gaussian_filter1d(a, sigma=1.0)
        assert_approx(cpp_r, py_r, "gaussian_filter1d basic")

    def test_sigma2(self, cpp):
        a = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0])
        cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=2.0))
        py_r  = sp_ndimage.gaussian_filter1d(a, sigma=2.0)
        assert_approx(cpp_r, py_r, "gaussian_filter1d sigma=2")

    def test_random(self, cpp):
        a = random_array((50,), seed=700)
        for sigma in [0.5, 1.0, 2.0, 3.0]:
            cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=sigma))
            py_r  = sp_ndimage.gaussian_filter1d(a, sigma=sigma)
            assert_approx(cpp_r, py_r, f"gaussian_filter1d random sigma={sigma}")

    def test_mode_reflect(self, cpp):
        a = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5, mode="reflect"))
        py_r  = sp_ndimage.gaussian_filter1d(a, sigma=1.5, mode="reflect")
        assert_approx(cpp_r, py_r, "gaussian_filter1d mode=reflect")

    def test_mode_constant(self, cpp):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.0, mode="constant", cval=0.0))
        py_r  = sp_ndimage.gaussian_filter1d(a, sigma=1.0, mode="constant", cval=0.0)
        assert_approx(cpp_r, py_r, "gaussian_filter1d mode=constant")

    def test_mode_nearest(self, cpp):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cpp_r = np.asarray(cpp.ndimage.gaussian_filter1d(a, sigma=1.5, mode="nearest"))
        py_r  = sp_ndimage.gaussian_filter1d(a, sigma=1.5, mode="nearest")
        assert_approx(cpp_r, py_r, "gaussian_filter1d mode=nearest")

# ============================================================================
# signal.medfilt tests
# ============================================================================

class TestMedfilt:
    def test_basic_k3(self, cpp):
        a = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0])
        cpp_r = np.asarray(cpp.signal.medfilt(a, kernel_size=3))
        py_r  = sp_signal.medfilt(a, kernel_size=3)
        assert_approx(cpp_r, py_r, "medfilt k=3")

    def test_k5(self, cpp):
        a = np.array([3.0, 1.0, 7.0, 2.0, 8.0, 4.0, 6.0, 5.0, 9.0])
        cpp_r = np.asarray(cpp.signal.medfilt(a, kernel_size=5))
        py_r  = sp_signal.medfilt(a, kernel_size=5)
        assert_approx(cpp_r, py_r, "medfilt k=5")

    def test_random(self, cpp):
        a = random_array((100,), seed=800)
        for k in [3, 5, 7]:
            cpp_r = np.asarray(cpp.signal.medfilt(a, kernel_size=k))
            py_r  = sp_signal.medfilt(a, kernel_size=k)
            assert_approx(cpp_r, py_r, f"medfilt random k={k}")

# ============================================================================
# spatial.transform.Rotation tests
# ============================================================================

class TestRotation:
    def test_identity(self, cpp):
        R = np.eye(3)
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
        py_euler  = py_rot.as_euler("xyz")
        assert_approx(cpp_euler, py_euler, "Rotation identity")

    def test_x_rotation_xyz(self, cpp):
        """Pure x-axis rotation, decomposed with 'xyz'"""
        theta = np.pi / 4
        R = sp_Rotation.from_euler("xyz", [theta, 0.0, 0.0]).as_matrix()
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
        py_euler  = py_rot.as_euler("xyz")
        assert_approx(cpp_euler, py_euler, "Rotation x-45deg xyz")

    def test_y_rotation_xyz(self, cpp):
        """Pure y-axis rotation, decomposed with 'xyz'"""
        theta = np.pi / 6
        R = sp_Rotation.from_euler("xyz", [0.0, theta, 0.0]).as_matrix()
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
        py_euler  = py_rot.as_euler("xyz")
        assert_approx(cpp_euler, py_euler, "Rotation y-30deg xyz")

    def test_z_rotation_xyz(self, cpp):
        """Pure z-axis rotation, decomposed with 'xyz'"""
        theta = np.pi / 3
        R = sp_Rotation.from_euler("xyz", [0.0, 0.0, theta]).as_matrix()
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
        py_euler  = py_rot.as_euler("xyz")
        assert_approx(cpp_euler, py_euler, "Rotation z-60deg xyz")

    def test_xyz_sequence(self, cpp):
        """Intrinsic xyz rotations: rx=20°, ry=30°, rz=45°"""
        angles = np.deg2rad([20.0, 30.0, 45.0])
        R = sp_Rotation.from_euler("xyz", angles).as_matrix()
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
        py_euler  = py_rot.as_euler("xyz")
        assert_approx(cpp_euler, py_euler, "Rotation xyz(20,30,45)")

    def test_zyx_sequence(self, cpp):
        """Intrinsic zyx rotations: rz=10°, ry=-20°, rx=40°"""
        angles = np.deg2rad([10.0, -20.0, 40.0])  # z,y,x
        R = sp_Rotation.from_euler("zyx", angles).as_matrix()
        cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
        py_rot  = sp_Rotation.from_matrix(R)
        cpp_euler = np.asarray(cpp_rot.as_euler("zyx"))
        py_euler  = py_rot.as_euler("zyx")
        assert_approx(cpp_euler, py_euler, "Rotation zyx(10,-20,40)")

    def test_random_xyz(self, cpp):
        rng = np.random.RandomState(42)
        for i in range(10):
            angles = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
            R = sp_Rotation.from_euler("xyz", angles).as_matrix()
            cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
            py_rot  = sp_Rotation.from_matrix(R)
            cpp_euler = np.asarray(cpp_rot.as_euler("xyz"))
            py_euler  = py_rot.as_euler("xyz")
            assert_approx(cpp_euler, py_euler, f"Rotation random xyz[{i}]", rtol=1e-6)

    def test_random_zyx(self, cpp):
        rng = np.random.RandomState(99)
        for i in range(10):
            angles = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
            R = sp_Rotation.from_euler("zyx", angles).as_matrix()
            cpp_rot = cpp.spatial.transform.Rotation.from_matrix(R)
            py_rot  = sp_Rotation.from_matrix(R)
            cpp_euler = np.asarray(cpp_rot.as_euler("zyx"))
            py_euler  = py_rot.as_euler("zyx")
            assert_approx(cpp_euler, py_euler, f"Rotation random zyx[{i}]", rtol=1e-6)

if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
