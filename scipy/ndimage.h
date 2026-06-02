// scipy.ndimage — gaussian_filter1d and other image filters.
//
// Usage: #include "scipy/ndimage.h"

#pragma once

#include <cmath>
#include <vector>
#include <cstddef>

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

    // Build Gaussian kernel
    T two_s2 = T(2) * sigma * sigma;
    std::vector<T> kernel(ks);
    T ksum = T(0);
    for (int i = 0; i < ks; ++i) {
        T x = T(i - half);
        kernel[i] = std::exp(-(x * x) / two_s2);
        ksum += kernel[i];
    }
    for (int i = 0; i < ks; ++i) kernel[i] /= ksum;

    // Convolve with boundary mode
    for (size_t i = 0; i < n; ++i) {
        T sum = T(0);
        for (int j = -half; j <= half; ++j) {
            ptrdiff_t idx = static_cast<ptrdiff_t>(i) + j;
            T val;

            if (idx >= 0 && static_cast<size_t>(idx) < n) {
                val = src[static_cast<size_t>(idx)];
            } else {
                // Apply boundary mode
                switch (mode) {
                    case 0:  // reflect (d c b a | a b c d | d c b a)
                    default: {
                        if (idx < 0) {
                            ptrdiff_t r = -idx - 1;
                            if (static_cast<size_t>(r) >= n) r = n - 1;
                            val = src[static_cast<size_t>(r)];
                        } else {
                            size_t r = n - 1 - (static_cast<size_t>(idx) - n);
                            if (r >= n) r = 0;
                            val = src[r];
                        }
                        break;
                    }
                    case 1:  // constant
                        val = cval;
                        break;
                    case 2: {  // nearest
                        if (idx < 0) val = src[0];
                        else         val = src[n - 1];
                        break;
                    }
                    case 3: {  // mirror (d c b a | a b c d | c b a)
                        if (idx < 0) {
                            ptrdiff_t r = -idx;
                            if (static_cast<size_t>(r) >= n) r = static_cast<ptrdiff_t>(n) - 1;
                            val = src[static_cast<size_t>(r)];
                        } else {
                            ptrdiff_t r = static_cast<ptrdiff_t>(n) - 2 - (static_cast<size_t>(idx) - n);
                            if (r < 0) r = 0;
                            val = src[static_cast<size_t>(r)];
                        }
                        break;
                    }
                    case 4: {  // wrap
                        if (idx < 0) {
                            ptrdiff_t r = static_cast<ptrdiff_t>(n) + idx;
                            while (r < 0) r += static_cast<ptrdiff_t>(n);
                            val = src[static_cast<size_t>(r)];
                        } else {
                            val = src[(static_cast<size_t>(idx)) % n];
                        }
                        break;
                    }
                }
            }
            sum += val * kernel[j + half];
        }
        dst[i] = sum;
    }
}

}  // namespace ndimage
}  // namespace scipy
