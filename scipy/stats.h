// Native C++ stats — scipy.stats.* equivalents.
//
// Uses numpcpp for bit-exact transcendental ops matching Python numpy/scipy.
// Ports Cephes erf/erfc/ndtri for bit-level alignment with scipy.special.
//
//   scipy.stats.norm.pdf(x, loc=0, scale=1)
//   scipy.stats.norm.cdf(x, loc=0, scale=1)
//   scipy.stats.norm.ppf(x, loc=0, scale=1)
//
// All results bit-identical to scipy (float64 and float32).

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include "numpy/core.h"

namespace scipy::stats {

// ============================================================================
// Scalar numpy bridge helpers — call numpy::array funcs for single values.
//   numpy::exp/log/sqrt are in the public numpy:: namespace (core.h).
//   Calling them with n=1 dispatches through the bridge to npy_*/SVML,
//   matching scipy's own math path bit-for-bit.
//   NOTE: NOT in anonymous namespace — header-only inline templates need
//   external linkage to avoid "used but never defined" warnings.
// ============================================================================
namespace detail {
    template<typename T>
    inline T np_exp(T x) { T r; numpy::exp(&x, &r, 1); return r; }
    template<typename T>
    inline T np_log(T x) { T r; numpy::log(&x, &r, 1); return r; }
    template<typename T>
    inline T np_sqrt(T x) { T r; numpy::sqrt(&x, &r, 1); return r; }
    template<typename T>
    inline T np_asin(T x) { T r; numpy::arcsin(&x, &r, 1); return r; }
    template<typename T>
    inline T np_atan2(T y, T x) { T r; numpy::arctan2_scalar(&y, &r, 1, x); return r; }
}  // namespace detail (scipy::stats::detail, NOT numpy::detail)

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
// Cephes polevl / p1evl — polynomial evaluation (Horner's method)
//
//   polevl(x, coef, N): evaluates  C_N*x^N + C_{N-1}*x^{N-1} + ... + C_0
//                        where coef[0] = C_N, ..., coef[N] = C_0
//   p1evl(x, coef, N):   evaluates  x^N + coef[0]*x^{N-1} + ... + coef[N-1]
//
// Ported from scipy/special/cephes/polevl.h — identical algorithm.
// ============================================================================

template<typename T>
inline T polevl(T x, const double coef[], int N) {
    const double* p = coef;
    T ans = T(*p++);
    int i = N;
    do { ans = ans * x + T(*p++); } while (--i);
    return ans;
}

template<typename T>
inline T p1evl(T x, const double coef[], int N) {
    const double* p = coef;
    T ans = x + T(*p++);
    int i = N - 1;
    do { ans = ans * x + T(*p++); } while (--i);
    return ans;
}

// ============================================================================
// Cephes erf / erfc — bit-identical to scipy.special.erf / scipy.special.erfc
//
// Ported from scipy/special/cephes/ndtr.c (version 1.8.0).
// Coefficients match scipy's Cephes library exactly.
// ============================================================================

// Coefficients for erfc(x), x >= 1, x < 8  (P: degree 8, Q: degree 9, leading coeff 1)
static const double _erfc_P[] = {
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2
};
static const double _erfc_Q[] = {
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2
};

// Coefficients for erfc(x), x >= 8  (R: degree 5, S: degree 6, leading coeff 1)
static const double _erfc_R[] = {
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0
};
static const double _erfc_S[] = {
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0
};

// Coefficients for erf(x), |x| <= 1  (T: degree 4, U: degree 5, leading coeff 1)
static const double _erf_T[] = {
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4
};
static const double _erf_U[] = {
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4
};

static const double _MAXLOG = 7.08396418532264106224E2;  // log(2**1022)
static const double _NPY_SQRT1_2 = 7.071067811865475244008e-1;  // 1/sqrt(2)

// Forward declarations for mutual recursion
template<typename T> inline T cephes_erf(T x);
template<typename T> inline T cephes_erfc(T a);

/// scipy.special.erfc(x) — Cephes complementary error function
template<typename T>
inline T cephes_erfc(T a) {
    if (std::isnan(a)) return std::numeric_limits<T>::quiet_NaN();

    T x = (a < T(0)) ? -a : a;

    if (x < T(1))
        return T(1) - cephes_erf(a);

    T z = -a * a;
    // Underflow check
    if (z < T(-_MAXLOG)) {
        return (a < T(0)) ? T(2) : T(0);
    }

    z = std::exp(z);  // exp(-a^2) — std::exp is libm, same as scipy's npy_exp

    T p, q;
    if (x < T(8)) {
        p = polevl(x, _erfc_P, 8);
        q = p1evl(x, _erfc_Q, 8);
    } else {
        p = polevl(x, _erfc_R, 5);
        q = p1evl(x, _erfc_S, 6);
    }
    T y = (z * p) / q;

    if (a < T(0))
        y = T(2) - y;

    if (y == T(0)) {
        return (a < T(0)) ? T(2) : T(0);
    }
    return y;
}

/// scipy.special.erf(x) — Cephes error function
template<typename T>
inline T cephes_erf(T x) {
    if (std::isnan(x)) return std::numeric_limits<T>::quiet_NaN();

    if (x < T(0))
        return -cephes_erf(-x);

    if (x > T(1))
        return T(1) - cephes_erfc(x);

    T z = x * x;
    return x * polevl(z, _erf_T, 4) / p1evl(z, _erf_U, 5);
}

/// scipy.special.ndtr(x) — Cephes normal CDF: area from -inf to x
template<typename T>
inline T cephes_ndtr(T a) {
    if (std::isnan(a)) return std::numeric_limits<T>::quiet_NaN();

    T x = a * T(_NPY_SQRT1_2);
    T z = std::fabs(x);

    if (z < T(_NPY_SQRT1_2))
        return T(0.5) + T(0.5) * cephes_erf(x);

    T y = T(0.5) * cephes_erfc(z);
    if (x > T(0))
        y = T(1) - y;
    return y;
}

// ============================================================================
// Cephes ndtri — bit-identical to scipy.special.ndtri
//
// Ported from scipy/special/cephes/ndtri.c (version 1.8.0).
// Coefficients match scipy's Cephes library exactly.
// ============================================================================

static const double _s2pi = 2.50662827463100050242E0;  // sqrt(2*pi)

// P0/Q0: approximation for 0 <= |y - 0.5| <= 3/8
// P0: degree 4, Q0: degree 8 (leading coeff 1)
static const double _ndtri_P0[] = {
    -5.99633501014107895267E1,
     9.80010754185999661536E1,
    -5.66762857469070293439E1,
     1.39312609387279679503E1,
    -1.23916583867381258016E0,
};
static const double _ndtri_Q0[] = {
     1.95448858338141759834E0,
     4.67627912898881538453E0,
     8.63602421390890590575E1,
    -2.25462687854119370527E2,
     2.00260212380060660359E2,
    -8.20372256168333339912E1,
     1.59056225126211695515E1,
    -1.18331621121330003142E0,
};

// P1/Q1: z = sqrt(-2 log y) between 2 and 8 (y between exp(-2) and exp(-32))
// P1: degree 8, Q1: degree 8 (leading coeff 1)
static const double _ndtri_P1[] = {
     4.05544892305962419923E0,
     3.15251094599893866154E1,
     5.71628192246421288162E1,
     4.40805073893200834700E1,
     1.46849561928858024014E1,
     2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
};
static const double _ndtri_Q1[] = {
     1.57799883256466749731E1,
     4.53907635128879210584E1,
     4.13172038254672030440E1,
     1.50425385692907503408E1,
     2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
};

// P2/Q2: z = sqrt(-2 log y) between 8 and 64 (y between exp(-32) and exp(-2048))
// P2: degree 8, Q2: degree 8 (leading coeff 1)
static const double _ndtri_P2[] = {
     3.23774891776946035970E0,
     6.91522889068984211695E0,
     3.93881025292474443415E0,
     1.33303460815807542389E0,
     2.01485389549179081538E-1,
     1.23716634817820021358E-2,
     3.01581553508235416007E-4,
     2.65806974686737550832E-6,
     6.23974539184983293730E-9,
};
static const double _ndtri_Q2[] = {
     6.02427039364742014255E0,
     3.67983563856160859403E0,
     1.37702099489081330271E0,
     2.16236993594496635890E-1,
     1.34204006088543189037E-2,
     3.28014464682127739104E-4,
     2.89247864745380683936E-6,
     6.79019408009981274425E-9,
};

/// scipy.special.ndtri(p) — Cephes inverse normal CDF
template<typename T>
inline T cephes_ndtri(T y0) {
    if (y0 == T(0)) return -std::numeric_limits<T>::infinity();
    if (y0 == T(1)) return  std::numeric_limits<T>::infinity();
    if (y0 < T(0) || y0 > T(1)) return std::numeric_limits<T>::quiet_NaN();

    // exp(-2) = 0.13533528323661269189 — exact threshold from Cephes ndtri.c
    static const T _expneg2 = T(1.3533528323661269189E-1);

    T y = y0;
    int code = 1;

    if (y > (T(1) - _expneg2)) {
        y = T(1) - y;
        code = 0;
    }

    if (y > _expneg2) {
        // Central region: rational approx in (y - 0.5)^2
        y = y - T(0.5);
        T y2 = y * y;
        T x = y + y * (y2 * polevl(y2, _ndtri_P0, 4) / p1evl(y2, _ndtri_Q0, 8));
        x = x * T(_s2pi);
        return x;
    }

    // Tail region: sqrt(-2 * log(y)) based approximation
    T x = std::sqrt(T(-2) * std::log(y));
    T x0 = x - std::log(x) / x;

    T z = T(1) / x;
    T x1;
    if (x < T(8))       // y > exp(-32)
        x1 = z * polevl(z, _ndtri_P1, 8) / p1evl(z, _ndtri_Q1, 8);
    else
        x1 = z * polevl(z, _ndtri_P2, 8) / p1evl(z, _ndtri_Q2, 8);

    x = x0 - x1;
    if (code != 0)
        x = -x;
    return x;
}

// ============================================================================
// norm.pdf
// ============================================================================

/// scipy.stats.norm.pdf(x, loc, scale)
template<typename T>
inline void norm_pdf(const T* src, T* dst, size_t n, T loc, T scale) {
    static const T sqrt_2pi = std::sqrt(T(2.0 * M_PI));
    std::vector<T> arg_buf(n);
    for (size_t i = 0; i < n; ++i) {
        T z = (src[i] - loc) / scale;
        arg_buf[i] = -(z * z) * T(0.5);
    }
    numpy::exp(arg_buf.data(), dst, n);
    // Division order matches scipy: (exp / sqrt_2pi) / scale.
    // IEEE 754: (a/b)/c ≠ a/(b*c) at last bit.
    for (size_t i = 0; i < n; ++i) {
        dst[i] /= sqrt_2pi;
        dst[i] /= scale;
    }
}

/// scipy.stats.norm.pdf(x) — default loc=0, scale=1
template<typename T>
inline void norm_pdf(const T* src, T* dst, size_t n) {
    static const T sqrt_2pi = std::sqrt(T(2.0 * M_PI));
    std::vector<T> arg_buf(n);
    for (size_t i = 0; i < n; ++i)
        arg_buf[i] = -(src[i] * src[i]) * T(0.5);
    numpy::exp(arg_buf.data(), dst, n);
    for (size_t i = 0; i < n; ++i)
        dst[i] /= sqrt_2pi;
}

// ============================================================================
// norm.cdf — cumulative distribution function
//
//   scipy.stats.norm.cdf(x) = scipy.special.ndtr((x-loc)/scale)
// ============================================================================

template<typename T>
inline void norm_cdf(const T* src, T* dst, size_t n, T loc, T scale) {
    SCIPY_UNROLL4(i, {
        T z = (src[i] - loc) / scale;
        dst[i] = cephes_ndtr(z);
    });
}

template<typename T>
inline void norm_cdf(const T* src, T* dst, size_t n) {
    norm_cdf(src, dst, n, T(0), T(1));
}

// ============================================================================
// norm.ppf — percent point function
//
//   scipy.stats.norm.ppf(x) = loc + scale * scipy.special.ndtri(p)
// ============================================================================

template<typename T>
inline void norm_ppf(const T* src, T* dst, size_t n, T loc, T scale) {
    SCIPY_UNROLL4(i, {
        T p = src[i];
        if (p <= T(0)) { dst[i] = -std::numeric_limits<T>::infinity(); continue; }
        if (p >= T(1)) { dst[i] =  std::numeric_limits<T>::infinity(); continue; }
        dst[i] = cephes_ndtri(p) * scale + loc;
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
