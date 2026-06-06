// scipy.ndimage — gaussian_filter1d and other image filters.
//
// Usage: #include "scipy/ndimage.h"

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include "numpy/core.h"

namespace scipy {
namespace ndimage {

// ============================================================================
// scipy.ndimage.gaussian_filter1d
// ============================================================================
// Aligns with Python: scipy.ndimage.gaussian_filter1d(input, sigma, axis=-1,
//     order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
//
// Parameters:
//   src       — input array, size n
//   dst       — output array, size n (may alias src)
//   n         — array length
//   sigma     — standard deviation of Gaussian kernel
//   truncate  — kernel half-width in units of sigma (default 4.0)
//   mode      — boundary mode: 0=reflect, 1=constant, 2=nearest, 3=mirror, 4=wrap

template<typename T>
inline void gaussian_filter1d(const T* src, T* dst, size_t n,
                               T sigma = T(1),
                               T truncate = T(4.0),
                               int mode = 0,    // 0=reflect
                               T cval = T(0)) {
    if (n == 0) return;
    if (sigma <= T(0)) { std::copy(src, src + n, dst); return; }

    // Compute kernel half-width and full kernel
    int half = static_cast<int>(std::ceil(truncate * sigma));
    if (half < 1) half = 1;
    int ks = 2 * half + 1;

    // Build Gaussian kernel — must match scipy's exact algorithm:
    //   x = arange(-radius, radius+1)
    //   phi_x = numpy.exp(-0.5 / sigma2 * x**2)
    //   phi_x = phi_x / phi_x.sum()
    // Use numpy::exp (core.h → SVML) to match scipy's numpy.exp output.
    T s2_inv_half = T(-0.5) / (sigma * sigma);
    std::vector<T> kernel(ks);
    std::vector<T> args(ks);
    for (int i = 0; i < ks; ++i) {
        T x = T(i - half);
        args[i] = s2_inv_half * (x * x);
    }
    numpy::exp(args.data(), kernel.data(), ks);
    // Sum in index order (matching numpy's phi_x.sum() sequential order)
    T ksum = T(0);
    for (int i = 0; i < ks; ++i) ksum += kernel[i];
    for (int i = 0; i < ks; ++i) kernel[i] /= ksum;

    // ====================================================================
    // Pre-filled line buffer (matching scipy's NI_InitLineBuffer).
    // Buffer size = n + 2*half; center = original data; edges = extended.
    // The inner loop reads from the buffer without boundary checks,
    // matching scipy's NI_Correlate1D exactly.
    // ====================================================================
    size_t buf_size = n + 2 * static_cast<size_t>(half);
    std::vector<T> buf(buf_size);
    T* buf_ptr = buf.data();

    // Fill center with original data
    std::copy(src, src + n, buf_ptr + half);

    // Boundary value function for filling edges.
    // Uses ptrdiff_t throughout (no unsigned underflow).
    // Formulas match scipy's NI_InitLineBuffer.
    auto bnd = [&](ptrdiff_t idx) -> T {
        if (idx >= 0 && static_cast<size_t>(idx) < n)
            return src[static_cast<size_t>(idx)];
        switch (mode) {
            case 0:  // reflect (d c b a | a b c d | d c b a)
            default: {
                // period = 2*n, boundary elements appear twice
                ptrdiff_t p = static_cast<ptrdiff_t>(2 * n);
                ptrdiff_t r = idx % p;
                if (r < 0) r += p;
                if (static_cast<size_t>(r) >= n)
                    r = p - 1 - r;
                return src[static_cast<size_t>(r)];
            }
            case 1:  // constant
                return cval;
            case 2:  // nearest
                return (idx < 0) ? src[0] : src[n - 1];
            case 3: {  // mirror (d c b | a b c d | c b a)
                // period = 2*n - 2, first/last elements appear once.
                // Guard: n==1 → period=0, modulo undefined; only element is src[0].
                if (n <= 1) return src[0];
                ptrdiff_t p = static_cast<ptrdiff_t>(2 * n - 2);
                ptrdiff_t r = idx % p;
                if (r < 0) r += p;
                if (static_cast<size_t>(r) >= n)
                    r = p - r;
                return src[static_cast<size_t>(r)];
            }
            case 4: {  // wrap
                ptrdiff_t r = idx % static_cast<ptrdiff_t>(n);
                if (r < 0) r += static_cast<ptrdiff_t>(n);
                return src[static_cast<size_t>(r)];
            }
        }
    };

    // Fill left edge: buf[half-1] = bnd(-1), buf[half-2] = bnd(-2), ...
    for (int j = 0; j < half; ++j)
        buf_ptr[half - 1 - j] = bnd(-j - 1);

    // Fill right edge: buf[half+n] = bnd(n), buf[half+n+1] = bnd(n+1), ...
    for (int j = 0; j < half; ++j)
        buf_ptr[half + n + j] = bnd(static_cast<ptrdiff_t>(n) + j);

    // ====================================================================
    // Convolution with symmetric correlation (matching scipy's fast path).
    // output[i] = buf[half+i] * kernel[half]
    //           + sum_{j=1}^{half} (buf[half+i-j] + buf[half+i+j]) * kernel[half-j]
    // ====================================================================
    for (size_t i = 0; i < n; ++i) {
        T sum = buf_ptr[half + i] * kernel[half];
        for (int j = half; j >= 1; --j) {
            sum += (buf_ptr[half + i - j] + buf_ptr[half + i + j]) * kernel[half - j];
        }
        dst[i] = sum;
    }
}

// ============================================================================
// gaussian_filter_correlate — convolution with pre-computed kernel
// ============================================================================
// Same convolution as gaussian_filter1d, but kernel is pre-computed
// externally (e.g., by Python numpy to guarantee bit-level alignment).
//
// Parameters:
//   src    — input array, size n
//   dst    — output array, size n
//   n      — array length
//   kernel — pre-computed normalized Gaussian kernel, size 2*half+1
//   half   — kernel half-width (number of elements on each side of center)
//   mode   — boundary mode: 0=reflect, 1=constant, 2=nearest, 3=mirror, 4=wrap
//   cval   — constant fill value for constant mode
template<typename T>
inline void gaussian_filter_correlate(const T* src, T* dst, size_t n,
                                       const T* kernel, int half,
                                       int mode = 0,
                                       T cval = T(0)) {
    if (n == 0) return;
    if (half < 0) half = 0;

    size_t buf_size = n + 2 * static_cast<size_t>(half);
    std::vector<T> buf(buf_size);
    T* buf_ptr = buf.data();

    // Fill center with original data
    std::copy(src, src + n, buf_ptr + half);

    // Boundary value function (same as gaussian_filter1d)
    auto bnd = [&](ptrdiff_t idx) -> T {
        if (idx >= 0 && static_cast<size_t>(idx) < n)
            return src[static_cast<size_t>(idx)];
        switch (mode) {
            case 0:  // reflect
            default: {
                ptrdiff_t p = static_cast<ptrdiff_t>(2 * n);
                ptrdiff_t r = idx % p;
                if (r < 0) r += p;
                if (static_cast<size_t>(r) >= n)
                    r = p - 1 - r;
                return src[static_cast<size_t>(r)];
            }
            case 1:  // constant
                return cval;
            case 2:  // nearest
                return (idx < 0) ? src[0] : src[n - 1];
            case 3: {  // mirror
                if (n <= 1) return src[0];  // guard: period = 2*n-2 = 0 when n=1
                ptrdiff_t p = static_cast<ptrdiff_t>(2 * n - 2);
                ptrdiff_t r = idx % p;
                if (r < 0) r += p;
                if (static_cast<size_t>(r) >= n)
                    r = p - r;
                return src[static_cast<size_t>(r)];
            }
            case 4:  // wrap
                ptrdiff_t r = idx % static_cast<ptrdiff_t>(n);
                if (r < 0) r += static_cast<ptrdiff_t>(n);
                return src[static_cast<size_t>(r)];
        }
    };

    // Fill edges
    for (int j = 0; j < half; ++j)
        buf_ptr[half - 1 - j] = bnd(-j - 1);
    for (int j = 0; j < half; ++j)
        buf_ptr[half + n + j] = bnd(static_cast<ptrdiff_t>(n) + j);

    // Convolution (matching scipy NI_Correlate1D symmetric correlation)
    for (size_t i = 0; i < n; ++i) {
        T sum = buf_ptr[half + i] * kernel[half];
        for (int j = half; j >= 1; --j) {
            sum += (buf_ptr[half + i - j] + buf_ptr[half + i + j]) * kernel[half - j];
        }
        dst[i] = sum;
    }
}

}  // namespace ndimage
}  // namespace scipy
