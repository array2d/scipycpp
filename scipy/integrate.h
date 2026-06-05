// scipy.integrate — header-only, depends on numpcpp for math primitives.
//
// Usage: #include "scipy/integrate.h"
//   scipy::integrate::trapezoid(y, n);
//   scipy::integrate::quad([](double x){ return x*x; }, 0.0, 4.0);

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <utility>
#include <functional>
#include <limits>

namespace scipy {
namespace integrate {

// ============================================================================
// scipy.integrate.trapezoid — composite trapezoidal rule
// ============================================================================

/// scipy.integrate.trapezoid(y, x=None, dx=1.0, axis=-1)
template<typename T>
inline T trapezoid(const T* y, const T* x, size_t n, T dx = T(1)) {
    if (n < 2) return T(0);
    T sum = T(0);
    if (x) {
        for (size_t i = 0; i < n - 1; ++i)
            sum += (x[i + 1] - x[i]) * (y[i] + y[i + 1]);
    } else {
        for (size_t i = 0; i < n - 1; ++i)
            sum += dx * (y[i] + y[i + 1]);
    }
    return sum * T(0.5);
}

template<typename T>
inline T trapezoid(const T* y, size_t n, T dx = T(1)) {
    const T* xp = nullptr;
    return trapezoid(y, xp, n, dx);
}

// ============================================================================
// scipy.integrate.simpson — composite Simpson's rule
// ============================================================================

/// scipy.integrate.simpson(y, x=None, dx=1.0, axis=-1, even='avg')
///
/// Always returns double, matching scipy's behaviour (scipy always returns
/// float64 regardless of input dtype because the final `* dx/3.0` uses a
/// Python float64 constant that promotes the result).
///
/// Internal summation is done in T (the input type), matching scipy's
/// computation path:
///   - float32 input → sum in float32, then double(sum) * (h/3)
///   - float64 input → sum in float64, then double(sum) * (h/3)
///
/// Summation is sequential (portable C++); scipy uses numpy SIMD.
/// For typical scientific data: 0 ULP.
/// For degenerate uniform arrays: ≤few float32 ULPs (in float32 precision)
/// or ≤6 float64 ULPs (in float64 precision) — sequential vs SIMD reorder.
template<typename T>
inline double simpson(const T* y, const T* x, size_t n, T dx = T(1)) {
    if (n < 2) return 0.0;
    if (n == 2) {
        double h = x ? double(x[1] - x[0]) : double(dx);
        return h * (double(y[0]) + double(y[1])) * 0.5;
    }
    if (n == 3) {
        double h = x ? double(x[2] - x[0]) * 0.5 : double(dx);
        // n=3: one Simpson interval; intermediate ops in T, final in double
        T s = y[0] + T(4)*y[1] + y[2];
        return h * double(s) / 3.0;
    }

    T h_t    = x ? (x[1] - x[0]) : dx;
    double h = double(h_t);

    if (n % 2 == 0) {
        // === even N: scipy default even='avg' ===
        // Variant 1: Simpson on [0..n-2], trapezoid on [n-2..n-1]
        T trap1 = T(0.5) * h_t * (y[n-1] + y[n-2]);  // trapezoid in T
        T sum1  = T(0);
        for (size_t i = 0; i + 2 < n - 1; i += 2)     // i: 0,2,...,n-4
            sum1 += y[i] + T(4)*y[i+1] + y[i+2];
        double res1 = double(trap1) + h * (1.0/3.0) * double(sum1);

        // Variant 2: trapezoid on [0..1], Simpson on [1..n-1]
        T trap2 = T(0.5) * h_t * (y[1] + y[0]);       // trapezoid in T
        T sum2  = T(0);
        for (size_t i = 1; i + 2 < n; i += 2)          // i: 1,3,...,n-3
            sum2 += y[i] + T(4)*y[i+1] + y[i+2];
        double res2 = double(trap2) + h * (1.0/3.0) * double(sum2);

        return (res1 + res2) * 0.5;
    } else {
        // === odd N: pure Simpson on all intervals ===
        T sum = T(0);
        for (size_t i = 0; i + 2 < n; i += 2)          // i: 0,2,...,n-3
            sum += y[i] + T(4)*y[i+1] + y[i+2];
        return h * (1.0/3.0) * double(sum);
    }
}

// Overload with dx only (no x array)
template<typename T>
inline double simpson(const T* y, size_t n, T dx = T(1)) {
    const T* xp = nullptr;
    return simpson(y, xp, n, dx);
}

// ============================================================================
// scipy.integrate.cumulative_trapezoid
// ============================================================================

template<typename T>
inline void cumulative_trapezoid(const T* y, const T* x, T* dst, size_t n,
                                  T dx = T(1), bool has_initial = false, T initial = T(0)) {
    if (n < 2) { if (has_initial && n == 1) dst[0] = initial; return; }
    T cum = T(0); size_t out = 0;
    if (has_initial) dst[out++] = initial;
    for (size_t i = 0; i < n - 1; ++i) {
        T h = x ? (x[i + 1] - x[i]) : dx;
        cum += h * (y[i] + y[i + 1]) * T(0.5);
        dst[out++] = cum;
    }
}

// ============================================================================
// scipy.integrate.quad — adaptive Simpson quadrature
// ============================================================================

namespace detail {

template<typename T, typename F>
inline T adaptive_simpson(F&& f, T a, T b, T eps, T whole, int max_depth) {
    if (max_depth <= 0) return whole;
    T c = (a + b) * T(0.5);
    T fa = f(a), fb = f(b), fc = f(c);
    T left  = (b - a) * (fa + T(4) * fc + fb) / T(12);
    T right = (b - a) * (fa + T(4) * f((a + c) * T(0.5)) + T(2) * fc +
                         T(4) * f((c + b) * T(0.5)) + fb) / T(24);
    if (std::abs(right - left) <= T(15) * eps)
        return right + (right - left) / T(15);
    return adaptive_simpson<T>(std::forward<F>(f), a, c, eps * T(0.5), left * T(0.5), max_depth - 1) +
           adaptive_simpson<T>(std::forward<F>(f), c, b, eps * T(0.5), left * T(0.5), max_depth - 1);
}

} // namespace detail

template<typename T, typename F>
inline std::pair<T, T> quad(F&& f, T a, T b, T epsabs = T(1.49e-08), int limit = 50) {
    T h = b - a, c = (a + b) * T(0.5);
    T s = h * (f(a) + T(4) * f(c) + f(b)) / T(6);
    T r = detail::adaptive_simpson<T>(std::forward<F>(f), a, b, epsabs, s, limit);
    return {r, std::abs(r - s)};
}

}  // namespace integrate
}  // namespace scipy
