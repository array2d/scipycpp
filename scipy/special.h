// scipy.special — based on C++17 std math + numpcpp.
//
// C++17 provides: std::erf, std::erfc, std::tgamma, std::lgamma.
// For advanced functions (bessel, etc.), optional Boost.Math can be used.
//
// Usage: #include "scipy/special.h"

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace scipy {
namespace special {

// ============================================================================
// Error functions — std::erf / std::erfc (C++11)
// ============================================================================

template<typename T> inline T erf(T x)  { return std::erf(x); }
template<typename T> inline T erfc(T x) { return std::erfc(x); }

// ============================================================================
// Gamma functions — std::tgamma / std::lgamma (C++11)
// ============================================================================

template<typename T> inline T gamma(T x)   { return std::tgamma(x); }
template<typename T> inline T gammaln(T x) { return std::lgamma(x); }

// ============================================================================
// Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
// ============================================================================

template<typename T>
inline T beta(T a, T b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// ============================================================================
// Factorial, comb, perm
// ============================================================================

inline double factorial(int n) {
    if (n < 0) return std::numeric_limits<double>::quiet_NaN();
    if (n <= 1) return 1.0;
    return std::tgamma(static_cast<double>(n + 1));
}

inline double comb(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    return std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0));
}

inline double perm(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0) return 1.0;
    return std::exp(std::lgamma(n + 1.0) - std::lgamma(n - k + 1.0));
}

// ============================================================================
// logsumexp — numerically stable log(sum(exp(x)))
// ============================================================================

template<typename T>
inline T logsumexp(const T* a, size_t n) {
    if (n == 0) return -std::numeric_limits<T>::infinity();
    T amax = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] > amax) amax = a[i];
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += std::exp(a[i] - amax);
    return amax + std::log(sum);
}

// ============================================================================
// softmax
// ============================================================================

template<typename T>
inline void softmax(const T* src, T* dst, size_t n) {
    T amax = src[0];
    for (size_t i = 1; i < n; ++i) if (src[i] > amax) amax = src[i];
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) { dst[i] = std::exp(src[i] - amax); sum += dst[i]; }
    for (size_t i = 0; i < n; ++i) dst[i] /= sum;
}

// ============================================================================
// digamma — ψ(x) = d/dx ln(Γ(x)), asymptotic expansion
// ============================================================================

template<typename T>
inline T digamma(T x) {
    T result = T(0);
    while (x < T(6)) { result -= T(1) / x; x += T(1); }
    T inv_x = T(1) / x, inv_x2 = inv_x * inv_x;
    result += std::log(x) - T(0.5) * inv_x
              - inv_x2 * (T(1.0/12.0) - inv_x2 * (T(1.0/120.0) - inv_x2 * T(1.0/252.0)));
    return result;
}

}  // namespace special
}  // namespace scipy
