"""
Bit-level alignment tests — scipycpp C++ vs Python scipy APIs.

SINGLE entry point: pytest tests/test_all.py -v

Coverage:
    - scipy.stats.norm.pdf: float64 + float32
    - scipy.integrate: trapezoid, simpson
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
    _V = {2: np.uint16, 4: np.uint32, 8: np.uint64}
    _F = {2: "04x", 4: "08x", 8: "016x"}
    lines = [f"BIT-LEVEL MISMATCH: {info['n_diff']}/{cpp.size} elements differ"]
    for idx in diff_indices:
        cv, pv = cpp.flat[idx], py.flat[idx]
        lines.append(f"  [{idx}] C++={cv} vs scipy={pv}")
    info["error"] = "\n".join(lines)
    return info

def assert_bit_aligned(cpp_result, py_result, label=""):
    info = check_bit_aligned(cpp_result, py_result, label)
    if not info["pass"]: raise AssertionError(info.get("error", "bit-level alignment failure"))

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

if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))
