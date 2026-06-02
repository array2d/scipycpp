// scipy.interpolate — linear, cubic spline, interp1d.
//
// Usage: #include "scipy/interpolate.h"

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <limits>

namespace scipy {
namespace interpolate {

// ============================================================================
// scipy.interpolate.interp1d — piecewise linear
// ============================================================================

template<typename T>
inline void interp1d_linear(const T* xp, const T* fp, size_t np,
                             const T* xi, T* yi, size_t ni,
                             T fill_left  = std::numeric_limits<T>::quiet_NaN(),
                             T fill_right = std::numeric_limits<T>::quiet_NaN()) {
    for (size_t k = 0; k < ni; ++k) {
        T x = xi[k];
        if (x <= xp[0])      { yi[k] = std::isnan(fill_left)  ? fp[0]   : fill_left;  continue; }
        if (x >= xp[np - 1]) { yi[k] = std::isnan(fill_right) ? fp[np - 1] : fill_right; continue; }
        size_t lo = 0, hi = np - 1;
        while (hi - lo > 1) { size_t mid = lo + (hi - lo) / 2; if (xp[mid] <= x) lo = mid; else hi = mid; }
        T t = (x - xp[lo]) / (xp[lo + 1] - xp[lo]);
        yi[k] = fp[lo] + t * (fp[lo + 1] - fp[lo]);
    }
}

// ============================================================================
// scipy.interpolate.CubicSpline — natural cubic spline
// ============================================================================

namespace detail {

template<typename T>
inline void spline_second_deriv(const T* x, const T* y, size_t n, T* y2) {
    if (n < 2) return;
    std::vector<T> u(n - 1);
    y2[0] = T(0); u[0] = T(0);

    for (size_t i = 1; i < n - 1; ++i) {
        T sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        T p = sig * y2[i - 1] + T(2);
        y2[i] = (sig - T(1)) / p;
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
               (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
        u[i] = (T(6) * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    y2[n - 1] = T(0);
    for (int i = static_cast<int>(n) - 2; i >= 0; --i)
        y2[i] = y2[i] * y2[i + 1] + u[i];
}

} // namespace detail

template<typename T>
inline void cubic_spline(const T* xp, const T* fp, size_t np,
                          const T* xi, T* yi, size_t ni) {
    if (np < 2) return;
    std::vector<T> y2(np);
    detail::spline_second_deriv(xp, fp, np, y2.data());

    for (size_t k = 0; k < ni; ++k) {
        T x = xi[k];
        if (x <= xp[0])      { yi[k] = fp[0];   continue; }
        if (x >= xp[np - 1]) { yi[k] = fp[np - 1]; continue; }
        size_t lo = 0, hi = np - 1;
        while (hi - lo > 1) { size_t mid = lo + (hi - lo) / 2; if (xp[mid] <= x) lo = mid; else hi = mid; }
        T h = xp[lo + 1] - xp[lo];
        T a = (xp[lo + 1] - x) / h, b = (x - xp[lo]) / h;
        yi[k] = a * fp[lo] + b * fp[lo + 1] +
                ((a * a * a - a) * y2[lo] + (b * b * b - b) * y2[lo + 1]) * (h * h) / T(6);
    }
}

}  // namespace interpolate
}  // namespace scipy
