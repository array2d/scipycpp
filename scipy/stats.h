// Native C++ stats — scipy.stats.* equivalents.
//
// Uses numpcpp's SVML bridge for bit-exact transcendental ops matching
// Python numpy/scipy.  Falls back to std::exp when SVML is unavailable.
//
//   scipy.stats.norm.pdf(x, loc=0, scale=1)
//
// Formula: exp(-0.5 * z^2) / (scale * sqrt(2*pi)),  z = (x-loc)/scale
// Computation order mirrors scipy.stats._continuous_distns._norm_pdf exactly.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

// numpcpp SVML bridge for bit-exact exp/log
#include "numpy/svml_bridge.h"

namespace scipy::stats {

// ============================================================================
// Loop unrolling
// ============================================================================

#define SCIPY_UNROLL4(dst_i, body)       \
    do { size_t _i = 0;                  \
        for (; _i + 3 < n; _i += 4) {    \
            size_t dst_i = _i + 0; body; \
            dst_i = _i + 1; body;        \
            dst_i = _i + 2; body;        \
            dst_i = _i + 3; body;        \
        }                                \
        for (; _i < n; ++_i) {           \
            size_t dst_i = _i; body;     \
        }                                \
    } while(0)

// ============================================================================
// norm.pdf
// ============================================================================

/// scipy.stats.norm.pdf(x, loc, scale)
template<typename T>
inline void norm_pdf(const T* src, T* dst, size_t n, T loc, T scale) {
    static const T sqrt_2pi = std::sqrt(T(2.0 * M_PI));
    SCIPY_UNROLL4(i, {
        T z = (src[i] - loc) / scale;
        T arg = -(z * z) / T(2.0);
        dst[i] = numpy::svml::exp(arg) / sqrt_2pi / scale;
    });
}

/// scipy.stats.norm.pdf(x) — default loc=0, scale=1
template<typename T>
inline void norm_pdf(const T* src, T* dst, size_t n) {
    static const T sqrt_2pi = std::sqrt(T(2.0 * M_PI));
    SCIPY_UNROLL4(i, {
        T arg = -(src[i] * src[i]) / T(2.0);
        dst[i] = numpy::svml::exp(arg) / sqrt_2pi;
    });
}

// ============================================================================
// norm.cdf — cumulative distribution function
// ============================================================================

template<typename T>
inline void norm_cdf(const T* src, T* dst, size_t n, T loc, T scale) {
    static const T sqrt2 = std::sqrt(T(2.0));
    SCIPY_UNROLL4(i, {
        T z = (src[i] - loc) / scale;
        dst[i] = T(0.5) * std::erfc(-z / sqrt2);
    });
}

template<typename T>
inline void norm_cdf(const T* src, T* dst, size_t n) {
    norm_cdf(src, dst, n, T(0), T(1));
}

// ============================================================================
// norm.ppf — percent point function (Acklam's approximation)
// ============================================================================

template<typename T>
inline void norm_ppf(const T* src, T* dst, size_t n, T loc, T scale) {
    const T a1 = T(-3.969683028665376e+01);
    const T a2 = T( 2.209460984245205e+02);
    const T a3 = T(-2.759285104469687e+02);
    const T a4 = T( 1.383577518672690e+02);
    const T a5 = T(-3.066479806614716e+01);
    const T a6 = T( 2.506628277459239e+00);
    const T b1 = T(-5.447609879822406e+01);
    const T b2 = T( 1.615858368580409e+02);
    const T b3 = T(-1.556989798598866e+02);
    const T b4 = T( 6.680131188771972e+01);
    const T b5 = T(-1.328068155288572e+01);
    const T c1 = T(-7.784894002430293e-03);
    const T c2 = T(-3.223964580411365e-01);
    const T c3 = T(-2.400758277161838e+00);
    const T c4 = T(-2.549732539343734e+00);
    const T c5 = T( 4.374664141464968e+00);
    const T c6 = T( 2.938163982698783e+00);
    const T d1 = T( 7.784695709041462e-03);
    const T d2 = T( 3.224671290700398e-01);
    const T d3 = T( 2.445134137142996e+00);
    const T d4 = T( 3.754408661907416e+00);
    const T plow  = T(0.02425);
    const T phigh = T(1.0) - plow;

    SCIPY_UNROLL4(i, {
        T p = src[i];
        if (p <= T(0)) { dst[i] = -std::numeric_limits<T>::infinity() * scale + loc; continue; }
        if (p >= T(1)) { dst[i] =  std::numeric_limits<T>::infinity() * scale + loc; continue; }

        T x;
        if (p < plow) {
            T q = std::sqrt(T(-2) * std::log(p));
            x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + T(1));
        } else if (p <= phigh) {
            T q = p - T(0.5);
            T r = q * q;
            x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + T(1));
        } else {
            T q = std::sqrt(T(-2) * std::log(T(1) - p));
            x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + T(1));
        }
        dst[i] = x * scale + loc;
    });
}

template<typename T>
inline void norm_ppf(const T* src, T* dst, size_t n) {
    norm_ppf(src, dst, n, T(0), T(1));
}

// ============================================================================
// uniform.pdf / uniform.cdf
// ============================================================================

template<typename T>
inline void uniform_pdf(const T* src, T* dst, size_t n, T loc, T scale) {
    T inv_scale = T(1) / scale;
    T upper = loc + scale;
    SCIPY_UNROLL4(i, {
        dst[i] = (src[i] >= loc && src[i] <= upper) ? inv_scale : T(0);
    });
}

template<typename T>
inline void uniform_pdf(const T* src, T* dst, size_t n) {
    uniform_pdf(src, dst, n, T(0), T(1));
}

template<typename T>
inline void uniform_cdf(const T* src, T* dst, size_t n, T loc, T scale) {
    T upper = loc + scale;
    SCIPY_UNROLL4(i, {
        if (src[i] < loc)       dst[i] = T(0);
        else if (src[i] > upper) dst[i] = T(1);
        else                     dst[i] = (src[i] - loc) / scale;
    });
}

}  // namespace scipy::stats
