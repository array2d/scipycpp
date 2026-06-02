// scipy.linalg — based on Eigen3 (header-only, MPL2).
//
// Eigen3 provides: solve, inv, det, eig, svd, cholesky, QR, LU, etc.
// We wrap Eigen3 to provide a scipy-compatible pointer-based API.
//
// Usage: #include "scipy/linalg.h"

#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <vector>
#include <cmath>
#include <cstddef>

namespace scipy {
namespace linalg {

// Helper: map C array → Eigen (row-major by default)
template<typename T>
inline auto as_matrix(const T* A, int rows, int cols) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
           Eigen::RowMajor>>(A, rows, cols);
}

template<typename T>
inline auto as_mutable_matrix(T* A, int rows, int cols) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
           Eigen::RowMajor>>(A, rows, cols);
}

// ============================================================================
// scipy.linalg.solve — solve Ax = b
// ============================================================================

/// scipy.linalg.solve(a, b)
/// A is n×n row-major, b is length n, x is length n.
/// Returns true on success.
template<typename T>
inline bool solve(const T* A, T* x, const T* b, int n) {
    auto mat = as_matrix(A, n, n);
    Eigen::Matrix<T, Eigen::Dynamic, 1> vec(n);
    for (int i = 0; i < n; ++i) vec(i) = b[i];

    Eigen::Matrix<T, Eigen::Dynamic, 1> result;
    // Use LU decomposition with partial pivoting
    auto lu = mat.partialPivLu();
    if (std::abs(mat.determinant()) < T(1e-15)) return false;
    result = lu.solve(vec);
    for (int i = 0; i < n; ++i) x[i] = result(i);
    return true;
}

// ============================================================================
// scipy.linalg.inv — matrix inverse
// ============================================================================

/// scipy.linalg.inv(a)
/// A is n×n row-major, Ainv is n×n row-major.
/// Returns true on success.
template<typename T>
inline bool inv(const T* A, T* Ainv, int n) {
    auto mat = as_matrix(A, n, n);
    auto lu = mat.partialPivLu();
    if (std::abs(mat.determinant()) < T(1e-15)) return false;
    auto inv_mat = lu.inverse();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Ainv[i * n + j] = inv_mat(i, j);
    return true;
}

// ============================================================================
// scipy.linalg.det — matrix determinant
// ============================================================================

/// scipy.linalg.det(a)
/// A is n×n row-major.
template<typename T>
inline T det(const T* A, int n) {
    auto mat = as_matrix(A, n, n);
    return mat.determinant();
}

// ============================================================================
// scipy.linalg.eig — eigenvalues (real symmetric)
// ============================================================================

/// scipy.linalg.eigvalsh(a) — eigenvalues of symmetric/hermitian matrix
template<typename T>
inline void eigvalsh(const T* A, T* eigenvalues, int n) {
    auto mat = as_matrix(A, n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor>> solver(mat);
    for (int i = 0; i < n; ++i)
        eigenvalues[i] = solver.eigenvalues()(i);
}

// ============================================================================
// scipy.linalg.cholesky — Cholesky decomposition
// ============================================================================

/// scipy.linalg.cholesky(a, lower=True)
/// A is n×n row-major positive-definite, L is n×n row-major lower-triangular.
/// Returns true on success.
template<typename T>
inline bool cholesky(const T* A, T* L, int n) {
    auto mat = as_matrix(A, n, n);
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor>> llt(mat);
    if (llt.info() != Eigen::Success) return false;
    auto result = llt.matrixL();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            L[i * n + j] = result(i, j);
    return true;
}

// ============================================================================
// scipy.linalg.svd — singular value decomposition
// ============================================================================

/// scipy.linalg.svd(a, full_matrices=False)
/// A is m×n row-major. Returns singular values.
template<typename T>
inline void svd_vals(const T* A, T* s, int m, int n) {
    auto mat = as_matrix(A, m, n);
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor>> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    for (int i = 0; i < std::min(m, n); ++i)
        s[i] = svd.singularValues()(i);
}

// ============================================================================
// scipy.linalg.lstsq — least squares
// ============================================================================

/// scipy.linalg.lstsq(a, b)
/// A is m×n row-major, b is length m, x is length n.
template<typename T>
inline bool lstsq(const T* A, int m, int n, const T* b, T* x) {
    auto mat = as_matrix(A, m, n);
    Eigen::Matrix<T, Eigen::Dynamic, 1> vec(m);
    for (int i = 0; i < m; ++i) vec(i) = b[i];

    // Use SVD-based least squares
    auto result = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vec);
    for (int i = 0; i < n; ++i) x[i] = result(i);
    return true;
}

// ============================================================================
// scipy.linalg.norm — matrix norms
// ============================================================================

/// scipy.linalg.norm(a) — Frobenius norm (default)
template<typename T>
inline T norm_fro(const T* A, int rows, int cols) {
    T sum = T(0);
    for (int i = 0; i < rows * cols; ++i) sum += A[i] * A[i];
    return std::sqrt(sum);
}

template<typename T>
inline T norm_l2(const T* v, int n) {
    T sum = T(0);
    for (int i = 0; i < n; ++i) sum += v[i] * v[i];
    return std::sqrt(sum);
}

}  // namespace linalg
}  // namespace scipy
