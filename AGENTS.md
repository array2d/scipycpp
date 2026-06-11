# Agent Instructions for scipycpp

## 1. 零容错 — 暴露错误，禁止吞错

- 前提条件不满足（BLAS符号缺失、ABI不匹配等）→ **立刻 `throw`，进程退出**。绝不 fallback、不返回假值。
- 数学失败（奇异矩阵等）→ **throw**。调用方如需处理，显式 catch。
- **禁止 silent failure。** 一个静默错误排查成本远超一次清晰 crash。
- ULP 偏差 → 追查根因并消除，**禁止放宽 tolerance 绕过**。

## 2. 比较函数 — 只用 ULP，严禁 allclose

- **只允许三种比较函数：**
  - `assert_bit_aligned` — 0 ULP（bit-identical 才通过）
  - `assert_ulp_close(cpp, py, label, max_ulp_tol=N)` — 允许 ≤N ULP
  - `_ulp_stats` — 返回原始统计，调用方自行判断
- **严禁** `np.allclose`、`np.isclose`、`np.testing.assert_allclose` 及任何 `atol`/`rtol` 比较。
- 所有比较必须报告 ULP 统计（max ULP、分布直方图、worst index），写入 `doc/ulp_report.csv`。
