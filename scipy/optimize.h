// scipy.optimize — Brent's method, golden-section, root finding.
//
// Usage: #include "scipy/optimize.h"

#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <limits>

namespace scipy {
namespace optimize {

// ============================================================================
// scipy.optimize.minimize_scalar — Brent's method for 1D
// ============================================================================

template<typename T, typename F>
inline std::pair<T, T> brent_min(F&& f, T a, T b, T tol = T(1e-5), int maxiter = 500) {
    const T golden = (T(3) - std::sqrt(T(5))) / T(2);
    T x = a + golden * (b - a), w = x, v = x;
    T fx = f(x), fw = fx, fv = fx;
    T d = T(0), e = T(0);

    for (int iter = 0; iter < maxiter; ++iter) {
        T m = (a + b) * T(0.5);
        T tol1 = tol * std::abs(x) + T(1e-10);
        T tol2 = T(2) * tol1;
        if (std::abs(x - m) <= tol2 - (b - a) * T(0.5))
            return {x, fx};

        T p, q, u, fu;
        if (std::abs(e) > tol1) {
            T r = (x - w) * (fx - fv), q_ = (x - v) * (fx - fw);
            p = (x - v) * q_ - (x - w) * r;
            q = T(2) * (q_ - r);
            if (q > T(0)) p = -p; else q = -q;
            T old_e = e; e = d;
            if (std::abs(p) >= std::abs(q * T(0.5) * old_e) || p <= q * (a - x) || p >= q * (b - x))
                { e = (x < m) ? b - x : a - x; d = golden * e; }
            else {
                d = p / q; u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (m > x) ? tol1 : -tol1;
            }
        } else {
            e = (x < m) ? b - x : a - x; d = golden * e;
        }
        u = (std::abs(d) >= tol1) ? x + d : x + ((d > T(0)) ? tol1 : -tol1);
        fu = f(u);

        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
            else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
        }
    }
    return {x, fx};
}

template<typename T, typename F>
inline std::pair<T, T> minimize_scalar_brent(F&& f, T a, T b, T tol = T(1e-5), int maxiter = 500) {
    if (a > b) std::swap(a, b);
    return brent_min<T>(std::forward<F>(f), a, b, tol, maxiter);
}

// ============================================================================
// scipy.optimize.root_scalar — Brent's method for root finding
// ============================================================================

template<typename T, typename F>
inline T brentq(F&& f, T a, T b, T tol = T(1e-8), int maxiter = 100) {
    T fa = f(a), fb = f(b);
    if (fa * fb >= T(0))
        throw std::invalid_argument("brentq: f(a) and f(b) must have opposite signs");

    if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
    T c = a, fc = fa, d_old = b - a;
    bool mflag = true;

    for (int iter = 0; iter < maxiter; ++iter) {
        if (std::abs(b - a) < tol || std::abs(fb) < tol) return b;

        T s;
        if (fa != fc && fb != fc) {
            T r = fa / fc, t = fb / fc, u = fb / fa;
            s = b - (t * (c - b) * r - (b - a) * (u - T(1))) /
                    ((r - T(1)) * (t - T(1)) * (u - T(1)));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }

        T mid = (T(3) * a + b) * T(0.25);
        if (!(((s > mid) && (s < b)) || ((s < mid) && (s > b))) ||
            (mflag && std::abs(s - b) >= std::abs(b - c) * T(0.5)) ||
            (!mflag && std::abs(s - b) >= std::abs(c - d_old) * T(0.5)) ||
            (mflag && std::abs(b - c) < tol) ||
            (!mflag && std::abs(d_old) < tol)) {
            s = (a + b) * T(0.5); mflag = true;
        } else { mflag = false; }
        d_old = c - b;

        T fs = f(s); c = b; fc = fb;
        if (fa * fs < T(0)) { b = s; fb = fs; }
        else                 { a = s; fa = fs; }
        if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
    }
    return b;
}

}  // namespace optimize
}  // namespace scipy
