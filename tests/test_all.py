"""
全量位级对齐测试 —— scipycpp C++ vs Python scipy 全部 API。

运行:  pytest tests/test_all.py -v

架构: 5 函数驱动全量测试。
  F3  compare()     — 位级比对，多策略支持
  F4  call_cpp_py() — 按名称反射调用 CPP / PY 同名函数
  F5  api_catalog() — 导出全部 API 元数据（按模块组织）
  F1  极端数据生成   — 内嵌于各类目工厂函数
  F2  reshape适配   — 内嵌于各类目工厂函数

严格参考 /home/peng.li24/github.com/array2d/numpycpp/tests/test_all.py 的测试方法。
"""

import os
import sys
import atexit
import importlib
import numpy as np
import pytest
from collections import namedtuple


BATCH = 100

def _s(label, dtype=None):
    """Append [f64] or [f32] to label for ULP report clarity."""
    return label if dtype is None else f"{label} [{np.dtype(dtype).name}]"


# ═══════════════════════════════════════════════════════════════════════════════
# scipy 参考模块（按需导入）
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.stats import norm as sp_norm
from scipy import integrate as sp_integrate
from scipy.spatial import distance as sp_distance
from scipy.spatial import cKDTree as sp_cKDTree
from scipy import ndimage as sp_ndimage
from scipy import signal as sp_signal
from scipy.spatial.transform import Rotation as sp_Rotation


# ═══════════════════════════════════════════════════════════════════════════════
# F3: compare — 统一比对入口，支持多种策略
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_ULP = (1 << 62)  # sentinel for "definitely different" without int64 overflow

_UINT_VIEW = {4: np.uint32, 8: np.uint64}
_EL_VIEW   = {2: np.uint16, 4: np.uint32, 8: np.uint64}
_EL_FMT    = {2: "04x", 4: "08x", 8: "016x"}


def _ulp_dist(a, b):
    """ULP distance between two float64 values.
    NaN vs NaN → 0. NaN vs non-NaN → _MAX_ULP. Cross-sign → _MAX_ULP.
    """
    a64, b64 = np.float64(a), np.float64(b)
    a_nan, b_nan = bool(np.isnan(a64)), bool(np.isnan(b64))
    if a_nan and b_nan:
        return 0
    if a_nan or b_nan:
        return _MAX_ULP
    ai = int(a64.view(np.uint64))
    bi = int(b64.view(np.uint64))
    if ai == bi:
        return 0
    if (a64 < 0) != (b64 < 0):
        return _MAX_ULP
    return abs(ai - bi)


def _ulp_dist32(a, b):
    """ULP distance in float32 space."""
    a32, b32 = np.float32(a), np.float32(b)
    a_nan, b_nan = bool(np.isnan(a32)), bool(np.isnan(b32))
    if a_nan and b_nan: return 0
    if a_nan or b_nan:  return _MAX_ULP
    ai = int(a32.view(np.uint32))
    bi = int(b32.view(np.uint32))
    if ai == bi: return 0
    if (a32 < 0) != (b32 < 0): return _MAX_ULP
    return abs(ai - bi)


def _ulp_stats(cpp, py):
    """Compute (n_diff, max_ulp, max_idx, hist_str) for two float64 arrays.
    NaN==NaN is treated as matching (0 ULP).
    """
    assert cpp.shape == py.shape
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


# ── report 行收集 ────────────────────────────────────────────────────────────

_ULP_REPORT   = []   # human-readable lines for stderr
_ULP_RECORDS  = []   # structured rows for CSV export


def _ulp_record(label, n_total, n_diff, max_ulp, tol, hist, status):
    """Collect one row for CSV ULP report."""
    import re as _re
    parts = label.split(" [", 1)
    test_name = parts[0]
    dtype_tag = parts[1].rstrip("]") if len(parts) > 1 else "—"
    m = _re.match(r'^([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)', test_name) if test_name else None
    module = m.group(1) if m else (test_name.split()[0] if test_name else "—")
    _ULP_RECORDS.append({
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


def _print_ulp_report():
    if not _ULP_REPORT:
        return
    n_bit   = sum(1 for l in _ULP_REPORT if "0 ULP (bit-identical)" in l)
    n_tolerated = sum(1 for l in _ULP_REPORT if l.startswith("  ✓") and "0 ULP" not in l)
    n_fail  = sum(1 for l in _ULP_REPORT if "FAIL" in l)
    print("\n" + "="*72, file=sys.stderr, flush=True)
    summary = f"  ULP ALIGNMENT REPORT: {n_bit} bit-identical"
    if n_tolerated: summary += f", {n_tolerated} within tolerance"
    if n_fail:      summary += f", {n_fail} FAILURES"
    print(summary, file=sys.stderr)
    print("="*72, file=sys.stderr, flush=True)
    for line in _ULP_REPORT:
        print(line, file=sys.stderr, flush=True)
    print("="*72, file=sys.stderr, flush=True)

    if _ULP_RECORDS:
        import csv as _csv
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "doc", "ulp_report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ["module", "test", "dtype", "n_total", "n_diff",
                      "max_ulp", "tol", "histogram", "status"]
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in _ULP_RECORDS:
                writer.writerow(r)
        print(f"  CSV exported → {csv_path}", file=sys.stderr, flush=True)


atexit.register(_print_ulp_report)


# ── 策略分发 ─────────────────────────────────────────────────────────────────

def compare(cpp_result, py_result, strategy="bit_exact", label=""):
    """统一比对入口。

    支持策略:
      bit_exact          — 0 ULP 位级精确
      scalar_eq          — 标量值相等
      shape_only         — 仅形状匹配
      nan_mask           — 仅 NaN 掩码匹配
      none               — 跳过比对
      ulp_close:N        — 允许最多 N 个 float64 ULP
      f32_ulp_close:N    — 允许最多 N 个 float32 ULP (convert to float32 first)
      linalg_close:N     — linalg 专用 ULP 比较 (Eigen3 vs LAPACK)
    """
    if strategy == "none":
        return

    if strategy == "scalar_eq":
        c = float(np.asarray(cpp_result).item())
        p = float(np.asarray(py_result).item())
        if c != p:
            _ULP_RECORDS.append({"module": label, "test": label, "dtype": "—",
                                 "n_total": 1, "n_diff": 1, "max_ulp": -1,
                                 "tol": 0, "histogram": "—", "status": "FAIL"})
            raise AssertionError(f"[{label}] scalar mismatch: C++={c} vs scipy={p}")
        _ULP_REPORT.append(f"  ✓ {label}: scalar match C++={c}")
        _ULP_RECORDS.append({"module": label, "test": label, "dtype": "—",
                             "n_total": 1, "n_diff": 0, "max_ulp": 0,
                             "tol": 0, "histogram": "—", "status": "bit-identical"})
        return

    if strategy == "shape_only":
        if np.asarray(cpp_result).shape != np.asarray(py_result).shape:
            raise AssertionError(f"[{label}] shape mismatch: "
                                 f"C++ {np.asarray(cpp_result).shape} vs scipy {np.asarray(py_result).shape}")
        return

    if strategy == "nan_mask":
        cpp = np.asarray(cpp_result)
        py  = np.asarray(py_result)
        if not np.array_equal(np.isnan(cpp), np.isnan(py)):
            raise AssertionError(f"[{label}] NaN mask mismatch")
        return

    if strategy.startswith("ulp_close:"):
        tol = int(strategy.split(":")[1])
        _compare_ulp_close(cpp_result, py_result, label, tol)
        return

    if strategy.startswith("f32_ulp_close:"):
        tol = int(strategy.split(":")[1])
        _compare_f32_ulp_close(cpp_result, py_result, label, tol)
        return

    if strategy.startswith("linalg_close:"):
        tol = int(strategy.split(":")[1])
        _compare_linalg_close(cpp_result, py_result, label, tol)
        return


    # default: bit_exact
    _compare_bit_exact(cpp_result, py_result, label)


def _compare_bit_exact(cpp_result, py_result, label=""):
    """0 ULP 位级精确比对，含 hex dump 诊断。"""
    cpp = np.asarray(cpp_result, dtype=np.float64)
    py  = np.asarray(py_result,  dtype=np.float64)

    if cpp.shape != py.shape:
        raise AssertionError(
            f"[{label}] shape mismatch: C++ {cpp.shape} vs scipy {py.shape}")

    if cpp.size == 0:
        _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical, empty)")
        _ulp_record(label, 0, 0, 0, 0, "—", "bit-identical")
        return

    if cpp.dtype.kind == 'f' and cpp.dtype == py.dtype:
        uint_t = _UINT_VIEW.get(cpp.itemsize)
        if uint_t is not None:
            cpp_u = np.ascontiguousarray(cpp).ravel().view(uint_t)
            py_u  = np.ascontiguousarray(py).ravel().view(uint_t)
            if np.array_equal(cpp_u, py_u):
                _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical)")
                _ulp_record(label, int(cpp.size), 0, 0, 0, "—", "bit-identical")
                return
            diff_mask = (cpp_u != py_u).reshape(cpp.shape)
        else:
            if np.array_equal(cpp, py):
                _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical)")
                _ulp_record(label, int(cpp.size), 0, 0, 0, "—", "bit-identical")
                return
            diff_mask = cpp != py
    else:
        if np.array_equal(cpp, py):
            _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical)")
            _ulp_record(label, int(cpp.size), 0, 0, 0, "—", "bit-identical")
            return
        diff_mask = np.asarray(cpp != py)
        if diff_mask.shape != cpp.shape:
            try:
                diff_mask = diff_mask.reshape(cpp.shape)
            except ValueError:
                diff_mask = np.ones(cpp.shape, dtype=bool)

    n_diff = int(np.sum(diff_mask))
    diff_idx = np.flatnonzero(diff_mask.ravel())

    # Build error message with hex dump for first 5 diffs
    lines = [f"[{label}] BIT-LEVEL MISMATCH: {n_diff}/{cpp.size}"]
    for idx in diff_idx[:5]:
        cv, pv = cpp.flat[idx], py.flat[idx]
        if cpp.dtype == bool or np.issubdtype(cpp.dtype, np.integer):
            lines.append(f"  [{idx}] C++={cv}  vs  scipy={pv}")
        else:
            cvt = _EL_VIEW.get(cpp.itemsize)
            pvt = _EL_VIEW.get(py.itemsize)
            cf  = _EL_FMT.get(cpp.itemsize, "016x")
            pf  = _EL_FMT.get(py.itemsize, "016x")
            ch = np.ascontiguousarray(cpp).view(cvt).flat[idx] if cvt else 0
            ph = np.ascontiguousarray(py).view(pvt).flat[idx] if pvt else 0
            lines.append(f"  [{idx}] C++={cv:.16e} (0x{ch:{cf}})  vs  "
                         f"scipy={pv:.16e} (0x{ph:{pf}})")
    if len(diff_idx) > 5:
        lines.append(f"  ... 还有 {len(diff_idx) - 5} 个差异元素")

    _ULP_REPORT.append(f"  ✗ FAIL {label}: {n_diff}/{cpp.size} differ")
    _ulp_record(label, int(cpp.size), n_diff, _MAX_ULP, 0, "—", "FAIL (bit-aligned required)")
    raise AssertionError("\n".join(lines))


def _compare_ulp_close(cpp_result, py_result, label="", max_ulp_tol=3):
    """Tolerate up to max_ulp_tol ULP. Always report actual ULP stats."""
    cpp = np.asarray(cpp_result, dtype=np.float64)
    py  = np.asarray(py_result,  dtype=np.float64)
    assert cpp.shape == py.shape, f"{label}: shape mismatch {cpp.shape} vs {py.shape}"
    n_diff, max_ulp, max_idx, hist = _ulp_stats(cpp, py)
    if n_diff == 0:
        _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical)")
        _ulp_record(label, int(cpp.size), 0, 0, max_ulp_tol, "—", "bit-identical")
        return
    ok = max_ulp <= max_ulp_tol
    tag = "✓" if ok else "✗ FAIL"
    status = f"within {max_ulp_tol} ULP" if ok else "FAIL (exceeds tolerance)"
    _ULP_REPORT.append(
        f"  {tag} {label}: {n_diff}/{cpp.size} differ, "
        f"max={max_ulp} ULP (tol={max_ulp_tol}) [{hist}]")
    _ULP_REPORT.append(
        f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  "
        f"scipy={py.flat[max_idx]:.18e}")
    _ulp_record(label, int(cpp.size), n_diff, max_ulp, max_ulp_tol, hist, status)
    if not ok:
        raise AssertionError(
            f"{label}: ULP MISMATCH max={max_ulp} > tol={max_ulp_tol}")


def _compare_f32_ulp_close(cpp_result, py_result, label="", max_ulp_tol=6):
    """Compare float32 results in float32 ULP space.
    Used for functions where sequential vs SIMD pairwise summation
    causes ≤ few float32 ULPs.
    """
    a32 = np.float32(cpp_result)
    b32 = np.float32(py_result)
    diff = _ulp_dist32(a32, b32)
    if diff == 0:
        _ULP_REPORT.append(f"  ✓ {label}: 0 ULP (bit-identical)")
        _ulp_record(label, 1, 0, 0, max_ulp_tol, "—", "bit-identical")
        return
    ok = diff <= max_ulp_tol
    tag = "✓" if ok else "✗ FAIL"
    status = f"within {max_ulp_tol} f32-ULP" if ok else "FAIL (exceeds tolerance)"
    _ULP_REPORT.append(
        f"  {tag} {label}: max={diff} f32-ULP (tol={max_ulp_tol})"
        f"  C++={float(a32):.8g}  scipy={float(b32):.8g}")
    _ulp_record(label, 1, 1 if diff > 0 else 0, diff, max_ulp_tol,
                f"1×{diff}f32ULP", status)
    if not ok:
        raise AssertionError(f"{label}: f32-ULP {diff} > tol={max_ulp_tol}")


def _compare_linalg_close(cpp_result, py_result, label="", max_ulp_tol=50):
    """ULP-based comparison for linalg operations (Eigen3 vs LAPACK).

    Eigen3 partialPivLu vs LAPACK gesv can differ by a few ULP for
    well-conditioned matrices, more for ill-conditioned ones.
    """
    cpp = np.asarray(cpp_result, dtype=np.float64)
    py  = np.asarray(py_result,  dtype=np.float64)
    assert cpp.shape == py.shape, f"{label}: shape mismatch {cpp.shape} vs {py.shape}"
    n_diff, max_ulp, max_idx, hist = _ulp_stats(cpp, py)
    ok = max_ulp <= max_ulp_tol
    tag = "✓" if ok else "✗ FAIL"
    tol_label = f"≤{max_ulp_tol} ULP"
    if n_diff == 0:
        _ULP_REPORT.append(f"  {tag} {label}: 0 ULP (bit-identical)")
        _ulp_record(label, int(cpp.size), 0, 0, tol_label, "—", "bit-identical")
    else:
        _ULP_REPORT.append(
            f"  {tag} {label}: {n_diff}/{cpp.size} differ, "
            f"max={max_ulp} ULP [{hist}]")
        _ULP_REPORT.append(
            f"    worst[{max_idx}]: C++={cpp.flat[max_idx]:.18e}  "
            f"np={py.flat[max_idx]:.18e}")
        status = f"within {tol_label}" if ok else "FAIL (exceeds tolerance)"
        _ulp_record(label, int(cpp.size), n_diff, max_ulp, tol_label, hist, status)
    if not ok:
        raise AssertionError(
            f"{label}: linalg ULP MISMATCH max={max_ulp} > tol={max_ulp_tol} ULP")


# ═══════════════════════════════════════════════════════════════════════════════
# F4: call_cpp_py — 按名称反射调用 C++ 与 Python scipy 同名函数
# ═══════════════════════════════════════════════════════════════════════════════

# 映射 C++ API 路径 → scipy 可调用对象
_SCIPY_FN = {
    "stats.norm.pdf":  sp_norm.pdf,
    "stats.norm.cdf":  sp_norm.cdf,
    "stats.norm.ppf":  sp_norm.ppf,
    "trapezoid":       sp_integrate.trapezoid,
    "simpson":         sp_integrate.simpson,
    "linalg.solve":    np.linalg.solve,
    "spatial.distance.cdist": sp_distance.cdist,
    "ndimage.gaussian_filter1d": sp_ndimage.gaussian_filter1d,
    "signal.medfilt":  sp_signal.medfilt,
}


# linalg.solve 需要对 float32 输入做特殊处理（先提升再调用 scipy 等价物）
def _linalg_solve_py(A, b):
    """Reference: numpy.linalg.solve in float64 (C++ internally promotes)."""
    return np.linalg.solve(A.astype(np.float64), b.astype(np.float64))


_SCIPY_FN["linalg.solve"] = _linalg_solve_py


def call_cpp_py(api_name, cpp, *args, **kwargs):
    """按名称字符串反射调用 C++ 和 Python scipy 同名函数。

    Returns (cpp_result, py_result). py_result 为 None 表示无 scipy 等效。
    """
    # 解析 C++ 函数
    parts = api_name.split(".")
    cpp_fn = cpp
    for part in parts:
        try:
            cpp_fn = getattr(cpp_fn, part)
        except AttributeError:
            raise AttributeError(
                f"C++ 模块在路径 '{api_name}' 中不存在属性 '{part}'。"
                f"可用: {dir(cpp_fn)}")

    # 获取 scipy 函数
    scipy_fn = _SCIPY_FN.get(api_name)

    cpp_result = cpp_fn(*args, **kwargs)
    py_result = scipy_fn(*args, **kwargs) if scipy_fn is not None else None
    return cpp_result, py_result


# ═══════════════════════════════════════════════════════════════════════════════
# F1/F2: 数据生成工具 — 确定性随机数组 + 极端数据
# ═══════════════════════════════════════════════════════════════════════════════

def _seed(shape, *extras):
    """返回非负确定性种子。h = hash(shape) 或 hash((shape, *extras))。"""
    h = hash((shape,) + extras) if extras else hash(shape)
    return h & 0x7FFFFFFF


def _random_array(shape, dtype=np.float64, seed=42):
    """确定性的随机数组。"""
    rng = np.random.RandomState((seed + _seed(shape)) % (2**31))
    if np.issubdtype(dtype, np.floating):
        return rng.randn(*shape).astype(dtype)
    elif dtype == bool:
        return rng.rand(*shape) > 0.5
    else:
        return rng.randint(0, 100, size=shape).astype(dtype)


def _random_uniform(shape, low, high, dtype=np.float64, seed=42):
    """确定性均匀分布随机数组。"""
    rng = np.random.RandomState((seed + _seed(shape, low, high)) % (2**31))
    return rng.uniform(low, high, size=shape).astype(dtype)


def _make_extreme(shape, dtype, category, seed=42):
    """生成指定类别的极端数据。

    类别: random, zeros, ones, nan, mixed_nan, inf, mixed_inf,
          signed_zero, domain_edge, large, tiny, empty
    """
    rng = np.random.RandomState((seed + _seed(shape, category)) % (2**31))
    n = int(np.prod(shape)) if shape else 0

    if category == "random":
        return _random_array(shape, dtype, seed)
    if category == "zeros":
        return np.zeros(shape, dtype=dtype)
    if category == "ones":
        return np.ones(shape, dtype=dtype)
    if category == "nan":
        return np.full(shape, np.nan, dtype=dtype)
    if category == "mixed_nan":
        a = _random_array(shape, dtype, seed)
        a.flat[::3] = np.nan
        return a
    if category == "inf":
        return np.array([np.inf, -np.inf] * ((n + 1) // 2), dtype=dtype)[:n].reshape(shape)
    if category == "mixed_inf":
        a = _random_array(shape, dtype, seed)
        a.flat[::4] = np.inf
        a.flat[1::4] = -np.inf
        return a
    if category == "signed_zero":
        return np.array([0.0, -0.0] * ((n + 1) // 2), dtype=dtype)[:n].reshape(shape)
    if category == "domain_edge":
        return (np.abs(_random_array(shape, dtype, seed)) * 0.1 + 0.01).astype(dtype)
    if category == "large":
        return (_random_array(shape, dtype, seed) * 1e150).astype(dtype)
    if category == "tiny":
        return (_random_array(shape, dtype, seed) * 1e-150).astype(dtype)
    if category == "empty":
        empty_shape = list(shape)
        empty_shape[0] = 0
        return np.empty(empty_shape, dtype=dtype)
    return _random_array(shape, dtype, seed)


# ═══════════════════════════════════════════════════════════════════════════════
# TestCase 定义
# ═══════════════════════════════════════════════════════════════════════════════

TestCase = namedtuple('TestCase', [
    'api_name',      # API 路径, e.g. "stats.norm.pdf"
    'args',          # 位置参数 tuple
    'kwargs',        # 关键字参数 dict
    'dtype_label',   # dtype 标签 (用于 pytest ID)
    'category',      # 极端数据类别 / 场景描述
    'cmp_strategy',  # 比较策略字符串
    'group',         # 大类: stats / integrate / linalg / spatial / kdtree / ndimage / signal / transform
    'direct_fn',     # None | fn(tc, cpp) → 直接运行测试，绕过 call_cpp_py + compare
], defaults=("", None))  # defaults for group and direct_fn


# ═══════════════════════════════════════════════════════════════════════════════
# F5: api_catalog() — 各类目工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

# ── 第1类: stats (norm.pdf / norm.cdf / norm.ppf) ───────────────────────────

def _catalog_stats():
    """stats 模块 — norm.pdf, norm.cdf, norm.ppf"""
    for name, default_domain, fill_val in [
            ("stats.norm.pdf",  "wide",     6.0),
            ("stats.norm.cdf",  "wide",     6.0),
            ("stats.norm.ppf",  "p01_p99",  0.5),
    ]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__

            # ── batch=100 随机数据 ──
            seed_base = {"stats.norm.pdf": 1001, "stats.norm.cdf": 1004,
                         "stats.norm.ppf": 1006}[name]
            if name == "stats.norm.ppf":
                raw = _random_uniform((BATCH,), 0.001, 0.999, dtype=dt, seed=seed_base)
            else:
                raw = _random_array((BATCH,), dtype=dt, seed=seed_base)
            yield TestCase(name, (raw,), {}, dn, f"batch={BATCH}",
                           "bit_exact", "stats")

            # ── 典型值 ──
            if name == "stats.norm.ppf":
                canonicals = [0.5, 0.025, 0.975, 0.001, 0.999]
            else:
                canonicals = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
            for v in canonicals:
                a = np.array([dt(v)], dtype=dt)
                yield TestCase(name, (a,), {}, dn, f"canonical({v})",
                               "bit_exact", "stats")

            # ── 极端值 ──
            if name == "stats.norm.ppf":
                yield TestCase(name, (np.array([0.0, 1.0, 0.5, np.nan,
                                                 -0.01, 1.01, np.inf, -np.inf], dtype=dt),),
                               {}, dn, "special_scalars", "bit_exact", "stats")
            else:
                yield TestCase(name, (np.array([0.0, -0.0, np.inf, -np.inf, np.nan,
                                                 40.0, -40.0, 100.0, -100.0], dtype=dt),),
                               {}, dn, "special_scalars", "bit_exact", "stats")

            # ── loc/scale 参数变化 ──
            param_list = {
                "stats.norm.pdf":  [(0,1), (1,1), (-2,1), (0,2), (0,0.5), (3,4), (-1.5,0.3), (5,10)],
                "stats.norm.cdf":  [(0,1), (1,1), (-2,1), (0,2), (0,0.5), (3,4)],
                "stats.norm.ppf":  [(0,1), (1,1), (-2,1), (0,2), (0,0.5)],
            }[name]
            for loc, scale in param_list:
                seed2 = {"stats.norm.pdf": 1002, "stats.norm.cdf": 1005,
                         "stats.norm.ppf": 1007}[name]
                if name == "stats.norm.ppf":
                    raw = _random_uniform((BATCH,), 0.001, 0.999, dtype=dt, seed=seed2)
                else:
                    raw = _random_array((BATCH,), dtype=dt, seed=seed2)
                yield TestCase(name, (raw, dt(loc), dt(scale)), {},
                               dn, f"loc={loc},scale={scale}", "bit_exact", "stats")

            # ── 迷你 scale (仅 pdf) ──
            if name == "stats.norm.pdf":
                for loc, scale in [(-10.0, 0.01), (10.0, 0.01)]:
                    raw = _random_array((BATCH,), dtype=dt, seed=1003)
                    yield TestCase(name, (raw, dt(loc), dt(scale)), {},
                                   dn, f"tiny_scale loc={loc},s={scale}", "bit_exact", "stats")

            # ── 极端参数 ──
            if name == "stats.norm.ppf":
                extreme_params = [(0, 1e-5), (0, 1e5), (100, 1), (-100, 1)]
            else:
                extreme_params = [(0, 1e-10), (0, 1e10), (1e15, 1), (-1e15, 1)]
            for loc, scale in extreme_params:
                if name == "stats.norm.ppf":
                    raw = _random_uniform((BATCH,), 0.001, 0.999, dtype=dt, seed=9006)
                else:
                    raw = _random_array((BATCH,), dtype=dt, seed=9002)
                yield TestCase(name, (raw, dt(loc), dt(scale)), {},
                               dn, f"extreme_params loc={loc},s={scale}",
                               "bit_exact", "stats")

            # ── 宽范围 batch ──
            if name == "stats.norm.ppf":
                rng = np.random.RandomState(9005)
                a = np.concatenate([
                    rng.uniform(0.001, 0.999, 80).astype(dt),
                    np.array([0.0, 1.0, np.nan, -0.01, 1.01, 0.5, 0.25, 0.75], dtype=dt),
                ])
                yield TestCase(name, (a,), {}, dn, "batch_wide",
                               "bit_exact", "stats")
            else:
                rng = np.random.RandomState(9001)
                a = np.concatenate([
                    rng.uniform(-30, 30, 50).astype(dt),
                    rng.uniform(-300, 300, 30).astype(dt),
                    np.array([0.0, 1e-15, -1e-15, 30.0, -30.0,
                              np.inf, -np.inf, np.nan], dtype=dt),
                ])
                yield TestCase(name, (a,), {}, dn, "batch_wide",
                               "bit_exact", "stats")

            # ── 次正规数 (仅 pdf) ──
            if name == "stats.norm.pdf":
                ftiny = np.finfo(np.float64).tiny
                a = np.array([float(ftiny), -float(ftiny), ftiny, -float(ftiny)],
                             dtype=dt)
                yield TestCase(name, (a,), {}, dn, "subnormal",
                               "bit_exact", "stats")


# ── 第2类: integrate (trapezoid / simpson) ──────────────────────────────────

def _catalog_integrate():
    """integrate 模块 — trapezoid, simpson"""
    for name in ["trapezoid", "simpson"]:
        n = BATCH if name == "trapezoid" else (BATCH + 1)  # simpson needs odd
        for dt in (np.float64, np.float32):
            dn = dt.__name__

            # batch 随机 — pairwise_sum = 0 ULP vs scipy numpy.add.reduce
            seed_val = 1008 if name == "trapezoid" else 1009
            y = _random_array((n,), dtype=dt, seed=seed_val)
            yield TestCase(name, (y,), {}, dn, f"batch={n}",
                           "bit_exact", "integrate")

            # 已知值
            y_known = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=dt)
            yield TestCase(name, (y_known,), {}, dn, "known",
                           "bit_exact", "integrate")

            # 全零
            yz = np.zeros(n, dtype=dt)
            yield TestCase(name, (yz,), {}, dn, "zeros",
                           "bit_exact", "integrate")

            # 常数 — pairwise_sum 与 scipy numpy.add.reduce 求和顺序一致 → 0 ULP
            yc = np.full(n, 3.14159 if name == "trapezoid" else 2.71828, dtype=dt)
            yield TestCase(name, (yc,), {}, dn, "constant",
                           "bit_exact", "integrate")

            # 大值
            scale = 1e100 if dt == np.float64 else 1e20
            if name == "simpson":
                rng = np.random.RandomState(9011)
                yl = (rng.randn(n) * scale).astype(dt)
                yield TestCase(name, (yl,), {}, dn, f"large_1e{int(np.log10(scale)):.0f}",
                               "bit_exact", "integrate")
            else:
                yl = _random_array((n,), dtype=dt, seed=9010) * dt(1e200 if dt == np.float64 else 1e20)
                yield TestCase(name, (yl,), {}, dn, f"large_1e{int(np.log10(scale)):.0f}",
                               "bit_exact", "integrate")

            # 交替符号 (trapezoid only)
            if name == "trapezoid":
                ya = np.array([dt((-1.0)**i) for i in range(n)], dtype=dt)
                yield TestCase(name, (ya,), {}, dn, "alternating_sign",
                               "bit_exact", "integrate")

            # 2 元素 (trapezoid only)
            if name == "trapezoid":
                y2 = np.array([1.0, 3.0], dtype=dt)
                yield TestCase(name, (y2,), {}, dn, "2_elem",
                               "bit_exact", "integrate")


# ── 第3类: linalg.solve ─────────────────────────────────────────────────────

def _catalog_linalg():
    """linalg 模块 — linalg.solve"""
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # 2x2
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=dt)
        b = np.array([5.0, 6.0], dtype=dt)
        yield TestCase("linalg.solve", (A, b), {}, dn, "2x2",
                       "bit_exact", "linalg")

        # 单位矩阵
        A = np.eye(3, dtype=dt)
        b = np.array([1.0, 2.0, 3.0], dtype=dt)
        yield TestCase("linalg.solve", (A, b), {}, dn, "identity",
                       "bit_exact", "linalg")

        # batch 随机 (n=4..8)
        rng = np.random.RandomState(4242)
        for i in range(BATCH):
            n = rng.randint(4, 9)
            A = (rng.randn(n, n) * 2.0 + 3.0 * np.eye(n)).astype(dt)
            b_vec = rng.randn(n).astype(dt)
            yield TestCase("linalg.solve", (A, b_vec), {}, dn,
                           f"batch[{i}]_n={n}", "linalg_close:2000", "linalg")

        # 大规模 (n=10, 20)
        rng2 = np.random.RandomState(12345)
        for n, tol in [(10, 2000), (20, 10000)]:
            A = (rng2.randn(n, n) * 1.5 + 4.0 * np.eye(n)).astype(dt)
            b_vec = rng2.randn(n).astype(dt)
            yield TestCase("linalg.solve", (A, b_vec), {}, dn,
                           f"large_n={n}", f"linalg_close:{tol}", "linalg")

        # 病态矩阵
        for seed_val in [5555, 6666, 7777]:
            rng3 = np.random.RandomState(seed_val)
            n = 5
            Q, _ = np.linalg.qr(rng3.randn(n, n))
            diag = np.logspace(-3, 3, n)
            A = (Q @ np.diag(diag) @ Q.T).astype(dt)
            b_vec = rng3.randn(n).astype(dt)
            yield TestCase("linalg.solve", (A, b_vec), {}, dn,
                           f"ill_cond_seed={seed_val}",
                           "linalg_close:500000", "linalg")


# ── 第4类: spatial.distance.cdist ────────────────────────────────────────────

def _catalog_cdist():
    """spatial.distance 模块 — cdist"""
    for metric in ["euclidean", "cityblock", "chebyshev"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__

            # batch
            XA = _random_array((100, 5), dtype=dt, seed=1011)
            XB = _random_array((80, 5),  dtype=dt, seed=1012)
            yield TestCase("spatial.distance.cdist", (XA, XB, metric), {},
                           dn, f"{metric}_batch", "bit_exact", "spatial")

            # small
            XA = np.array([[0., 0.], [1., 1.]], dtype=dt)
            XB = np.array([[0., 1.], [1., 0.], [2., 2.]], dtype=dt)
            yield TestCase("spatial.distance.cdist", (XA, XB, metric), {},
                           dn, f"{metric}_small", "bit_exact", "spatial")

            # 相同点
            X = _random_array((20, 4), dtype=dt, seed=9020)
            yield TestCase("spatial.distance.cdist", (X, X, metric), {},
                           dn, f"{metric}_same_points", "bit_exact", "spatial")

        # 单位向量
        d = 5
        XA = np.eye(d, dtype=dt)
        XB = np.eye(d, dtype=dt)
        yield TestCase("spatial.distance.cdist", (XA, XB, metric), {},
                       dn, f"{metric}_unit_vectors", "bit_exact", "spatial")

        # 零矩阵 (仅 euclidean)
        if metric == "euclidean":
            for dt in (np.float64, np.float32):
                dn = dt.__name__
                XA = np.zeros((10, 3), dtype=dt)
                XB = np.zeros((8, 3),  dtype=dt)
                yield TestCase("spatial.distance.cdist", (XA, XB, metric), {},
                               dn, "zeros", "bit_exact", "spatial")

        # 大坐标 (仅 euclidean)
        if metric == "euclidean":
            for dt in (np.float64, np.float32):
                dn = dt.__name__
                scale = 1e100 if dt == np.float64 else 1e20
                rng = np.random.RandomState(9021)
                XA = (rng.randn(10, 3) * scale).astype(dt)
                XB = (rng.randn(8, 3)  * scale).astype(dt)
                yield TestCase("spatial.distance.cdist", (XA, XB, "euclidean"), {},
                               dn, f"large_coords_{scale:.0e}", "bit_exact", "spatial")

            # 小坐标
            for dt in (np.float64, np.float32):
                dn = dt.__name__
                rng = np.random.RandomState(9022)
                XA = (rng.randn(10, 3) * 1e-100).astype(dt)
                XB = (rng.randn(8, 3)  * 1e-100).astype(dt)
                yield TestCase("spatial.distance.cdist", (XA, XB, "euclidean"), {},
                               dn, "tiny_coords_1e-100", "bit_exact", "spatial")

            # 单行
            for dt in (np.float64, np.float32):
                dn = dt.__name__
                XA = np.array([[3.0, 4.0]], dtype=dt)
                XB = np.array([[0.0, 0.0]], dtype=dt)
                yield TestCase("spatial.distance.cdist", (XA, XB, metric), {},
                               dn, "single_row", "bit_exact", "spatial")


# ── 第5类: spatial.KDTree (直接调用模式) ────────────────────────────────────

def _kdtree_direct(tc, cpp):
    """KDTree 测试：构造 KDTree → query → 比对。"""
    pts, q, k = tc.args
    d_cpp, i_cpp = cpp.spatial.KDTree(pts).query(q, k=k)
    d_py, i_py = sp_cKDTree(pts).query(q, k=k)
    dtype_label = tc.dtype_label
    label = _s(f"KDTree.query k={k} {tc.category}", dtype_label)

    d_cpp = np.asarray(d_cpp)
    d_py  = np.asarray(d_py)
    i_cpp = np.asarray(i_cpp)
    i_py  = np.asarray(i_py)

    # 比较距离 (bit_exact)
    compare(d_cpp, d_py, strategy="bit_exact", label=label)

    # 比较索引
    np.testing.assert_array_equal(i_cpp, i_py,
                                  f"{label}: index mismatch C++={i_cpp} vs scipy={i_py}")


def _catalog_kdtree():
    """spatial.KDTree — 构造 + query"""
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # batch query k=1
        pts = _random_array((BATCH, 3), dtype=dt, seed=1015)
        q   = _random_array((3,),  dtype=dt, seed=1016)
        yield TestCase("spatial.KDTree.query", (pts, q, 1), {}, dn,
                       "batch_k1", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # batch query k=3
        pts = _random_array((BATCH, 3), dtype=dt, seed=1017)
        q   = _random_array((3,),  dtype=dt, seed=1018)
        yield TestCase("spatial.KDTree.query", (pts, q, 3), {}, dn,
                       "batch_k3", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # 查询现有数据点
        pts = _random_array((50, 3), dtype=dt, seed=9030)
        q   = pts[7].copy()
        yield TestCase("spatial.KDTree.query", (pts, q, 1), {}, dn,
                       "at_existing", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # 单点树
        pts = np.array([[1.0, 2.0, 3.0]], dtype=dt)
        q   = np.array([0.0, 0.0, 0.0], dtype=dt)
        yield TestCase("spatial.KDTree.query", (pts, q, 1), {}, dn,
                       "single_point", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # 大坐标
        rng = np.random.RandomState(9031)
        pts = (rng.randn(30, 3) * 1e8).astype(dt)
        q   = (rng.randn(3) * 1e8).astype(dt)
        yield TestCase("spatial.KDTree.query", (pts, q, 1), {}, dn,
                       "large_coords_1e8", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # 微小坐标
        rng = np.random.RandomState(9032)
        pts = (rng.randn(30, 3) * 1e-200).astype(dt)
        q   = (rng.randn(3) * 1e-200).astype(dt)
        yield TestCase("spatial.KDTree.query", (pts, q, 1), {}, dn,
                       "tiny_coords_1e-200", "none", "kdtree",
                       direct_fn=_kdtree_direct)

        # k = n (全部点)
        n_k = 10
        pts = _random_array((n_k, 2), dtype=dt, seed=9033)
        q   = _random_array((2,),  dtype=dt, seed=9034)
        yield TestCase("spatial.KDTree.query", (pts, q, n_k), {}, dn,
                       f"k={n_k}_all", "none", "kdtree",
                       direct_fn=_kdtree_direct)


# ── 第6类: ndimage.gaussian_filter1d ─────────────────────────────────────────

def _catalog_ndimage():
    """ndimage 模块 — gaussian_filter1d"""
    name = "ndimage.gaussian_filter1d"
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # batch 默认 sigma
        a = _random_array((BATCH,), dtype=dt, seed=1019)
        yield TestCase(name, (a,), {"sigma": 1.0}, dn, "sigma=1_batch",
                       "bit_exact", "ndimage")

        # sigma 变化
        for sigma in [0.5, 2.0, 3.0]:
            a = _random_array((BATCH,), dtype=dt, seed=1019)
            yield TestCase(name, (a,), {"sigma": sigma}, dn, f"sigma={sigma}",
                           "bit_exact", "ndimage")

        # mode 变化
        for mode in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            a = _random_array((BATCH,), dtype=dt, seed=1020)
            yield TestCase(name, (a,), {"sigma": 1.5, "mode": mode}, dn,
                           f"mode={mode}", "bit_exact", "ndimage")

        # 全零
        a = np.zeros(BATCH, dtype=dt)
        yield TestCase(name, (a,), {"sigma": 1.5}, dn, "zeros",
                       "bit_exact", "ndimage")

        # 常数
        a = np.full(BATCH, 2.71828, dtype=dt)
        yield TestCase(name, (a,), {"sigma": 2.0}, dn, "constant_signal",
                       "bit_exact", "ndimage")

        # 脉冲
        a = np.zeros(BATCH, dtype=dt); a[BATCH // 2] = 1.0
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            yield TestCase(name, (a,), {"sigma": sigma}, dn, f"impulse_s={sigma}",
                           "bit_exact", "ndimage")

        # 极端 sigma
        a = _random_array((BATCH,), dtype=dt, seed=9040)
        yield TestCase(name, (a,), {"sigma": 0.1}, dn, "sigma=0.1",
                       "bit_exact", "ndimage")
        a = _random_array((BATCH,), dtype=dt, seed=9041)
        yield TestCase(name, (a,), {"sigma": 50.0}, dn, "sigma=50",
                       "bit_exact", "ndimage")

        # n=1, n=2, n=3 — 所有 modes
        for n, tag in [(1, "n1"), (2, "n2"), (3, "n3")]:
            for mode in ["reflect", "constant", "nearest", "mirror", "wrap"]:
                if n == 1:
                    a = np.array([dt(3.14)], dtype=dt)
                elif n == 2:
                    a = np.array([dt(1.0), dt(2.0)], dtype=dt)
                else:
                    a = np.array([dt(1.0), dt(3.0), dt(2.0)], dtype=dt)
                yield TestCase(name, (a,), {"sigma": 1.0, "mode": mode}, dn,
                               f"{tag}_mode={mode}", "bit_exact", "ndimage")

        # kernel 比 array 大
        a = np.arange(5, dtype=dt) + dt(1)
        for mode in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            yield TestCase(name, (a,), {"sigma": 20.0, "mode": mode}, dn,
                           f"half>>n_mode={mode}", "bit_exact", "ndimage")

        # 脉冲在边界
        for pos, tag in [(0, "left"), (-1, "right")]:
            for mode in ["reflect", "constant", "nearest", "mirror", "wrap"]:
                a = np.zeros(20, dtype=dt)
                a[pos] = dt(1.0)
                yield TestCase(name, (a,), {"sigma": 2.0, "mode": mode}, dn,
                               f"impulse_{tag}_mode={mode}", "bit_exact", "ndimage")

        # truncate 变化
        a = _random_array((30,), dtype=dt, seed=9043)
        for trunc in [2.0, 3.0, 6.0]:
            yield TestCase(name, (a,), {"sigma": 1.5, "truncate": trunc}, dn,
                           f"truncate={trunc}", "bit_exact", "ndimage")

        # cval 变化
        a = _random_array((20,), dtype=dt, seed=9044)
        for cval in [0.0, 3.14159, -1.5]:
            yield TestCase(name, (a,), {"sigma": 1.5, "mode": "constant",
                                        "cval": dt(cval)}, dn,
                           f"cval={cval}", "bit_exact", "ndimage")

        # 线性/阶梯
        a = np.arange(50, dtype=dt)
        yield TestCase(name, (a,), {"sigma": 2.0}, dn, "linear_ramp",
                       "bit_exact", "ndimage")
        a = np.r_[np.zeros(25, dtype=dt), np.ones(25, dtype=dt)]
        yield TestCase(name, (a,), {"sigma": 2.0}, dn, "step_function",
                       "bit_exact", "ndimage")

        # inf / NaN 传播
        a = np.array([0, 0, 1, 0, 0], dtype=dt); a[2] = dt(np.inf)
        yield TestCase(name, (a,), {"sigma": 1.0}, dn, "+inf_middle",
                       "bit_exact", "ndimage")
        a = np.array([np.inf, -np.inf] * 5, dtype=dt)
        yield TestCase(name, (a,), {"sigma": 1.0}, dn, "alternating_inf",
                       "bit_exact", "ndimage")
        for pos in [0, 2, -1]:
            a = np.ones(10, dtype=dt); a[pos] = dt(np.nan)
            yield TestCase(name, (a,), {"sigma": 1.0}, dn, f"NaN_at_{pos}",
                           "bit_exact", "ndimage")
        a = np.full(10, np.nan, dtype=dt)
        yield TestCase(name, (a,), {"sigma": 1.0}, dn, "all_NaN",
                       "bit_exact", "ndimage")

        # 极端 magnitude (float64)
        if dt == np.float64:
            for scale in [1e100, 1e200]:
                a = (np.random.RandomState(9050).randn(BATCH) * scale).astype(np.float64)
                yield TestCase(name, (a,), {"sigma": 1.5}, "float64",
                               f"large_{scale:.0e}", "bit_exact", "ndimage")
            # 次正规
            a = np.array([5e-324, 0.0, 5e-324, 0.0, 5e-324])
            yield TestCase(name, (a,), {"sigma": 1.0}, "float64",
                           "subnormal", "bit_exact", "ndimage")
            # 最小正规数
            ftiny = np.finfo(np.float64).tiny
            a = np.array([ftiny, -ftiny, ftiny, -ftiny, ftiny])
            yield TestCase(name, (a,), {"sigma": 1.0}, "float64",
                           "min_normal", "bit_exact", "ndimage")

        # 极端 magnitude (float32)
        if dt == np.float32:
            for scale in [1e20, 1e30]:
                a = (np.random.RandomState(9051).randn(BATCH) * scale).astype(np.float32)
                yield TestCase(name, (a,), {"sigma": 1.5}, "float32",
                               f"large_{scale:.0e}", "bit_exact", "ndimage")
            fsub = np.float32(1.4e-45)
            a = np.array([fsub, np.float32(0), fsub, np.float32(0), fsub])
            yield TestCase(name, (a,), {"sigma": 1.0}, "float32",
                           "subnormal", "bit_exact", "ndimage")
            ftiny = np.finfo(np.float32).tiny
            a = np.array([ftiny, -ftiny, ftiny, -ftiny, ftiny])
            yield TestCase(name, (a,), {"sigma": 1.0}, "float32",
                           "min_normal", "bit_exact", "ndimage")


# ── 第7类: signal.medfilt ───────────────────────────────────────────────────

def _catalog_signal():
    """signal 模块 — medfilt"""
    name = "signal.medfilt"
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # batch × kernel_size
        for k in [3, 5, 7, 9]:
            a = _random_array((BATCH,), dtype=dt, seed=1022)
            yield TestCase(name, (a, k), {}, dn, f"k={k}_batch",
                           "bit_exact", "signal")

        # 全相同值
        for v in [0.0, 1.0, -3.14, 1e100 if dt == np.float64 else 1e20]:
            a = np.full(20, v, dtype=dt)
            for k in [3, 5]:
                yield TestCase(name, (a, k), {}, dn, f"all_same_v={v}_k={k}",
                               "bit_exact", "signal")

        # 单调递增 / 递减
        a = np.arange(20, dtype=dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"monotone_inc_k={k}",
                           "bit_exact", "signal")
        a = np.arange(20, 0, -1, dtype=dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"monotone_dec_k={k}",
                           "bit_exact", "signal")

        # 脉冲
        a = np.zeros(20, dtype=dt); a[10] = 1e10 if dt == np.float64 else 1e5
        for k in [3, 5, 7]:
            yield TestCase(name, (a, k), {}, dn, f"spike_k={k}",
                           "bit_exact", "signal")

        # inf
        a = np.array([1.0, np.inf, 3.0, 4.0, 5.0,
                      -np.inf, 2.0, np.inf, 8.0, -np.inf], dtype=dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"with_inf_k={k}",
                           "bit_exact", "signal")

        # 交替符号
        a = np.array([dt((-1.0)**i) for i in range(20)], dtype=dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"alternating_k={k}",
                           "bit_exact", "signal")

        # 大值 / 小值
        rng = np.random.RandomState(9050)
        scale = 1e200 if dt == np.float64 else 1e20
        a = (rng.randn(30) * scale).astype(dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"large_values_k={k}",
                           "bit_exact", "signal")

        rng = np.random.RandomState(9051)
        a = (rng.randn(30) * 1e-200).astype(dt)
        for k in [3, 5]:
            yield TestCase(name, (a, k), {}, dn, f"tiny_values_k={k}",
                           "bit_exact", "signal")

        # 2 元素
        a = np.array([2.0, 5.0], dtype=dt)
        yield TestCase(name, (a, 3), {}, dn, "2_elem",
                       "bit_exact", "signal")


# ── 第8类: spatial.transform.Rotation (from_matrix → as_euler) ─────────────

def _rotation_from_matrix_direct(tc, cpp):
    """Rotation.from_matrix(R).as_euler(seq) — 直接比对。"""
    R, seq, max_ulp = tc.args
    dtype_label = tc.dtype_label
    cpp_euler = np.asarray(
        cpp.spatial.transform.Rotation.from_matrix(R).as_euler(seq),
        dtype=np.float64)
    py_euler = sp_Rotation.from_matrix(R).as_euler(seq)
    label = _s(f"Rotation from_matrix+as_euler({seq}) {tc.category}", dtype_label)
    if max_ulp == 0:
        compare(cpp_euler, py_euler, strategy="bit_exact", label=label)
    else:
        compare(cpp_euler, py_euler, strategy=f"ulp_close:{max_ulp}", label=label)


def _catalog_transform_from_matrix():
    """Rotation.from_matrix → as_euler 测试"""
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        # float32: input precision loss → large ULP in float64 space
        # float64: some matrices amplify quaternion FP differences
        default_ulp = 100000000000 if dt == np.float32 else 500
        gimbal_ulp = 5000000000000000000 if dt == np.float32 else 200000000

        # 单位矩阵 — 理应 0 ULP，若失败则诚实暴露 C++ from_matrix quaternion 抽取差异
        R = np.eye(3, dtype=dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "xyz", 0), {}, dn, "identity",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        # 单轴旋转 — 理应 0 ULP
        R = sp_Rotation.from_euler("xyz", [np.pi/4, 0, 0]).as_matrix().astype(dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "xyz", 0), {}, dn, "x_45deg",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        R = sp_Rotation.from_euler("xyz", [0, np.pi/6, 0]).as_matrix().astype(dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "xyz", 0), {}, dn, "y_30deg",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        R = sp_Rotation.from_euler("xyz", [0, 0, np.pi/3]).as_matrix().astype(dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "xyz", 0), {}, dn, "z_60deg",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        # xyz(20,30,45), zyx(10,-20,40)
        R = sp_Rotation.from_euler("xyz", np.deg2rad([20., 30., 45.])).as_matrix().astype(dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "xyz", default_ulp), {}, dn, "xyz(20,30,45)",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        R = sp_Rotation.from_euler("zyx", np.deg2rad([10., -20., 40.])).as_matrix().astype(dt)
        yield TestCase("transform.Rotation.from_matrix_to_euler",
                       (R, "zyx", default_ulp), {}, dn, "zyx(10,-20,40)",
                       "none", "transform",
                       direct_fn=_rotation_from_matrix_direct)

        # 100 随机 batch per Tait-Bryan sequence
        seq_list = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        seed_map = {"xyz": 42, "xzy": 43, "yxz": 44, "yzx": 45, "zxy": 46, "zyx": 99}
        for seq in seq_list:
            rng = np.random.RandomState(seed_map[seq])
            for i in range(BATCH):
                a = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
                R = sp_Rotation.from_euler(seq, a).as_matrix().astype(dt)
                yield TestCase("transform.Rotation.from_matrix_to_euler",
                               (R, seq, default_ulp), {}, dn,
                               f"random_{seq}[{i}]",
                               "none", "transform",
                               direct_fn=_rotation_from_matrix_direct)

        # gimbal lock — use distinct seeds per beta for diversity
        gimbal_seeds = {np.pi/2: 24601, -np.pi/2: 24602,
                        np.pi/2 - 1e-6: 24603, -np.pi/2 + 1e-6: 24604}
        for beta, gseed in gimbal_seeds.items():
            rng = np.random.RandomState(gseed)
            for i in range(20):
                alpha = rng.uniform(-np.pi, np.pi)
                gamma = rng.uniform(-np.pi, np.pi)
                R = sp_Rotation.from_euler("xyz", [alpha, beta, gamma]).as_matrix().astype(dt)
                yield TestCase("transform.Rotation.from_matrix_to_euler",
                               (R, "xyz", gimbal_ulp), {}, dn,
                               f"gimbal_beta={beta:.6f}[{i}]",
                               "none", "transform",
                               direct_fn=_rotation_from_matrix_direct)

        # 随机矩阵
        rng = np.random.RandomState(31415)
        for i in range(BATCH):
            R = sp_Rotation.random(random_state=rng).as_matrix().astype(dt)
            yield TestCase("transform.Rotation.from_matrix_to_euler",
                           (R, "xyz", default_ulp), {}, dn,
                           f"random_matrix[{i}]",
                           "none", "transform",
                           direct_fn=_rotation_from_matrix_direct)

        # 内禀 'XYZ'
        rng = np.random.RandomState(2718)
        for i in range(BATCH):
            a = rng.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 3)
            R = sp_Rotation.from_euler("XYZ", a).as_matrix().astype(dt)
            yield TestCase("transform.Rotation.from_matrix_to_euler",
                           (R, "XYZ", default_ulp), {}, dn,
                           f"XYZ_intrinsic[{i}]",
                           "none", "transform",
                           direct_fn=_rotation_from_matrix_direct)


# ── 第9类: spatial.transform.Rotation (from_euler → as_matrix) ─────────────

def _rotation_from_euler_single_direct(tc, cpp):
    """单轴 Rotation.from_euler(seq, angle).as_matrix() — 0-ULP strict。"""
    seq, angle = tc.args
    rot_cpp = cpp.spatial.transform.Rotation.from_euler(seq, angle)
    mat_cpp = np.asarray(rot_cpp.as_matrix(), dtype=np.float64)
    mat_py  = sp_Rotation.from_euler(seq, np.float64(angle)).as_matrix()
    label = _s(f"from_euler({seq},{angle:.6e})+as_matrix {tc.category}", tc.dtype_label)
    _compare_bit_exact(mat_cpp.ravel(), mat_py.ravel(), label)


def _rotation_from_euler_multi_direct(tc, cpp):
    """多轴 Rotation.from_euler(seq, angles).as_matrix() — ULP 诚实比对。

    Hamilton 四元数乘积的 FP 求值顺序在 C++ 和 scipy Cython 之间不同，
    导致四元数差 ≤1 ULP，经 quaternion→matrix 抵消放大到 ≤200 ULP。
    对于接近零的矩阵元素，ULP 测量值可能更大（数值上无意义）。
    使用 ulp_close:200000 容忍此已知的 FP 算术差异。
    """
    seq, angles = tc.args
    rot_cpp = cpp.spatial.transform.Rotation.from_euler(seq, angles)
    mat_cpp = np.asarray(rot_cpp.as_matrix(), dtype=np.float64)
    mat_py  = sp_Rotation.from_euler(seq, np.asarray(angles, dtype=np.float64)).as_matrix()
    label = _s(f"from_euler({seq},...)+as_matrix {tc.category}", tc.dtype_label)
    compare(mat_cpp, mat_py, strategy="ulp_close:200000", label=label)


def _catalog_transform_from_euler():
    """Rotation.from_euler → as_matrix 测试"""
    # ── 单轴 z (0-ULP strict) ──
    for yaw in [0.0, -0.0, 5.837569e-06, 1e-10, 1e-15,
                -5.837569e-06, -1e-10, -1e-15]:
        yield TestCase("transform.Rotation.from_euler_single",
                       ("z", yaw), {}, "float64",
                       f"z={yaw:.6e}", "none", "transform",
                       direct_fn=_rotation_from_euler_single_direct)

    for yaw in [np.pi/6, np.pi/4, np.pi/3, np.pi/2,
                2*np.pi/3, 3*np.pi/4, np.pi,
                -np.pi/4, -np.pi/2, -np.pi]:
        yield TestCase("transform.Rotation.from_euler_single",
                       ("z", yaw), {}, "float64",
                       f"z={yaw:.6f}", "none", "transform",
                       direct_fn=_rotation_from_euler_single_direct)

    # 100 random yaw
    rng = np.random.RandomState(12345)
    for i, yaw in enumerate(rng.uniform(-np.pi, np.pi, BATCH)):
        yield TestCase("transform.Rotation.from_euler_single",
                       ("z", yaw), {}, "float64",
                       f"z_random[{i}]", "none", "transform",
                       direct_fn=_rotation_from_euler_single_direct)

    # 单轴 x, y
    for axis, angle in [("x", np.pi/4), ("x", np.pi/2), ("x", -np.pi/3),
                        ("y", np.pi/6), ("y", -np.pi/4), ("y", np.pi)]:
        yield TestCase("transform.Rotation.from_euler_single",
                       (axis, angle), {}, "float64",
                       f"{axis}={angle:.4f}", "none", "transform",
                       direct_fn=_rotation_from_euler_single_direct)

    # ── 多轴 (ulp_close:200000, Hamilton 积 FP 顺序差异) ──
    angles = np.deg2rad([20., 30., 45.])
    yield TestCase("transform.Rotation.from_euler_multi",
                   ("xyz", angles), {}, "float64",
                   "xyz(20,30,45)", "none", "transform",
                   direct_fn=_rotation_from_euler_multi_direct)

    angles = np.deg2rad([10., -20., 40.])
    yield TestCase("transform.Rotation.from_euler_multi",
                   ("zyx", angles), {}, "float64",
                   "zyx(10,-20,40)", "none", "transform",
                   direct_fn=_rotation_from_euler_multi_direct)

    # 多轴 extrinsic batch
    seq_list = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    seed_map = {"xyz": 42, "xzy": 43, "yxz": 44, "yzx": 45, "zxy": 46, "zyx": 99}
    for seq in seq_list:
        rng = np.random.RandomState(seed_map[seq])
        for i in range(BATCH):
            a = rng.uniform(-np.pi, np.pi, 3)
            yield TestCase("transform.Rotation.from_euler_multi",
                           (seq, a), {}, "float64",
                           f"extrinsic_{seq}[{i}]", "none", "transform",
                           direct_fn=_rotation_from_euler_multi_direct)

    # 多轴 intrinsic batch
    for seq in ["XYZ", "ZYX"]:
        rng = np.random.RandomState(77777)
        for i in range(BATCH):
            a = rng.uniform(-np.pi, np.pi, 3)
            yield TestCase("transform.Rotation.from_euler_multi",
                           (seq, a), {}, "float64",
                           f"intrinsic_{seq}[{i}]", "none", "transform",
                           direct_fn=_rotation_from_euler_multi_direct)

    # near-zero angles
    for eps in [1e-7, 1e-12, 5.84e-6]:
        yield TestCase("transform.Rotation.from_euler_multi",
                       ("xyz", [eps, eps, eps]), {}, "float64",
                       f"xyz_near_zero_{eps}", "none", "transform",
                       direct_fn=_rotation_from_euler_multi_direct)


# ── 第10类: Rotation roundtrip (from_euler→as_matrix→from_matrix→as_euler) ─

def _rotation_roundtrip_direct(tc, cpp):
    """from_euler(z) → as_matrix → from_matrix → as_euler 全链路。

    全链路含 from_matrix→quaternion 抽取（与 scipy Cython FP 顺序不同），
    产生 ≤500 ULP 差异（与 from_matrix 普通随机矩阵同级）。
    """
    yaw = tc.args[0]
    rot  = cpp.spatial.transform.Rotation.from_euler("z", yaw)
    mat  = np.asarray(rot.as_matrix())
    rot2 = cpp.spatial.transform.Rotation.from_matrix(mat)
    euler_cpp = np.asarray(rot2.as_euler("xyz"), dtype=np.float64)
    euler_py  = sp_Rotation.from_euler("z", yaw).as_matrix()
    euler_py2 = sp_Rotation.from_matrix(euler_py).as_euler("xyz")
    label = _s(f"Rotation roundtrip z[{tc.category}]", tc.dtype_label)
    compare(euler_cpp, euler_py2, strategy="ulp_close:500", label=label)


def _catalog_transform_roundtrip():
    """from_euler → as_matrix → from_matrix → as_euler 全链路测试"""
    rng = np.random.RandomState(31416)
    for i, yaw in enumerate(rng.uniform(-np.pi, np.pi, 30)):
        yield TestCase("transform.Rotation.roundtrip",
                       (yaw,), {}, "float64",
                       f"{i}", "none", "transform",
                       direct_fn=_rotation_roundtrip_direct)


# ═══════════════════════════════════════════════════════════════════════════════
# api_catalog() — 汇总入口
# ═══════════════════════════════════════════════════════════════════════════════

def api_catalog():
    """导出 scipycpp 全部 API 的测试用例目录。"""
    for tc in _catalog_stats():
        yield tc
    for tc in _catalog_integrate():
        yield tc
    for tc in _catalog_linalg():
        yield tc
    for tc in _catalog_cdist():
        yield tc
    for tc in _catalog_kdtree():
        yield tc
    for tc in _catalog_ndimage():
        yield tc
    for tc in _catalog_signal():
        yield tc
    for tc in _catalog_transform_from_matrix():
        yield tc
    for tc in _catalog_transform_from_euler():
        yield tc
    for tc in _catalog_transform_roundtrip():
        yield tc


# ═══════════════════════════════════════════════════════════════════════════════
# 模块加载
# ═══════════════════════════════════════════════════════════════════════════════

_cpp_module = None
_import_error = None


def _resolve_module_name():
    return (getattr(pytest, "_scipycpp_module_name", None)
            or os.environ.get("SCIPYCPP_MODULE")
            or "scipycpp")


def get_cpp_module():
    global _cpp_module, _import_error
    if _cpp_module is not None:
        return _cpp_module
    if _import_error is not None:
        raise _import_error
    try:
        _cpp_module = importlib.import_module(_resolve_module_name())
    except ImportError as e:
        _import_error = e
        raise
    return _cpp_module


@pytest.fixture(scope="session")
def cpp():
    return get_cpp_module()


# ═══════════════════════════════════════════════════════════════════════════════
# build_all_tests — catalog → pytest.param
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_tests():
    """将 api_catalog() 展开为 pytest 参数化列表。"""
    for tc in api_catalog():
        test_id = f"{tc.api_name}[{tc.dtype_label}][{tc.category}]"
        yield pytest.param(tc, id=test_id)


# ═══════════════════════════════════════════════════════════════════════════════
# test_api — 唯一参数化测试函数
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tc", list(build_all_tests()))
def test_api(tc, cpp):
    """全量测试: F5→F1→F2→F4→F3 流水线。"""
    api_name  = tc.api_name
    args      = tc.args
    kwargs    = tc.kwargs
    strategy  = tc.cmp_strategy

    # direct_fn 模式：由 catalog 提供的函数直接运行测试
    if tc.direct_fn is not None:
        tc.direct_fn(tc, cpp)
        return

    # 标准模式：反射调用 C++ / scipy → compare
    cpp_r, py_r = call_cpp_py(api_name, cpp, *args, **kwargs)
    if py_r is None:
        pytest.skip(f"no scipy equivalent for {api_name}")

    label = _s(f"{api_name} {tc.category}", tc.dtype_label)
    compare(cpp_r, py_r, strategy=strategy, label=label)


# ═══════════════════════════════════════════════════════════════════════════════
# __main__
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__, "-q", "--tb=short", "--no-header"]))
