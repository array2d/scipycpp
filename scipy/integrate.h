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
    return trapezoid(y, nullptr, n, dx);
}

// ============================================================================
// scipy.integrate.simpson — composite Simpson's rule
// ============================================================================

/// scipy.integrate.simpson(y, x=None, dx=1.0, axis=-1, even='avg')
template<typename T>
inline T simpson(const T* y, const T* x, size_t n, T dx = T(1)) {
    if (n < 2) return T(0);
    if (n == 2) { T h = x ? (x[1] - x[0]) : dx; return h * (y[0] + y[1]) * T(0.5); }
    if (n == 3) { T h = x ? (x[2] - x[0]) * T(0.5) : dx; return h * (y[0] + T(4)*y[1] + y[2]) / T(3); }

    T e = T(0), o = T(0);
    if (n % 2 == 0) {
        for (size_t i = 2; i + 3 < n; i += 2) e += y[i];
        for (size_t i = 1; i + 3 < n; i += 2) o += y[i];
        T h = x ? (x[1] - x[0]) : dx;
        T r = h / T(3) * (y[0] + y[n - 4] + T(4)*o + T(2)*e);
        return r + h * T(3) / T(8) * (y[n-4] + T(3)*y[n-3] + T(3)*y[n-2] + y[n-1]);
    } else {
        for (size_t i = 2; i < n - 1; i += 2) e += y[i];
        for (size_t i = 1; i < n - 1; i += 2) o += y[i];
        T h = x ? (x[1] - x[0]) : dx;
        return h / T(3) * (y[0] + y[n - 1] + T(4)*o + T(2)*e);
    }
}


// Overload with dx only (no x array)
template<typename T>
inline T simpson(const T* y, size_t n, T dx = T(1)) {
    return simpson(y, (const T*)nullptr, n, dx);
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
