// scipy.signal — convolution, correlation, medfilt.
//
// Usage: #include "scipy/signal.h"

#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace scipy {
namespace signal {

template<typename T>
inline void convolve_full(const T* a, size_t na, const T* b, size_t nb, T* dst) {
    size_t nc = na + nb - 1;
    for (size_t k = 0; k < nc; ++k) {
        T sum = T(0);
        size_t j_start = (k >= nb - 1) ? k - (nb - 1) : 0;
        size_t j_end   = std::min(k + 1, na);
        for (size_t j = j_start; j < j_end; ++j)
            sum += a[j] * b[k - j];
        dst[k] = sum;
    }
}

template<typename T>
inline void convolve_same(const T* a, size_t na, const T* b, size_t nb, T* dst) {
    size_t nc = std::max(na, nb);
    for (size_t k = 0; k < nc; ++k) {
        T sum = T(0);
        size_t shift = (nb > 1) ? (nb / 2) : 0;
        size_t kf = k + shift;
        size_t js = (kf >= nb) ? kf - (nb - 1) : 0;
        for (size_t j = js; j < std::min(kf + 1, na); ++j)
            sum += a[j] * b[kf - j];
        dst[k] = sum;
    }
}

template<typename T>
inline void correlate_full(const T* a, size_t na, const T* b, size_t nb, T* dst) {
    size_t nc = na + nb - 1;
    for (size_t k = 0; k < nc; ++k) {
        T sum = T(0);
        ptrdiff_t lag = static_cast<ptrdiff_t>(k) - static_cast<ptrdiff_t>(nb) + 1;
        for (size_t j = 0; j < na; ++j) {
            ptrdiff_t bi = static_cast<ptrdiff_t>(j) - lag;
            if (bi >= 0 && static_cast<size_t>(bi) < nb) sum += a[j] * b[bi];
        }
        dst[k] = sum;
    }
}

// ============================================================================
// scipy.signal.medfilt — 1D median filter
// ============================================================================
// Aligns with Python: scipy.signal.medfilt(volume, kernel_size=None)
// For 1D: sliding window median with odd kernel_size.
// Boundary: scipy.signal.medfilt uses zero-padding (constant, cval=0),
//           NOT ndimage's default reflect mode.

template<typename T>
inline void medfilt(const T* src, T* dst, size_t n, int kernel_size = 3) {
    if (kernel_size < 2) kernel_size = 3;  // enforce odd
    if (kernel_size % 2 == 0) kernel_size += 1;
    int k2 = kernel_size / 2;
    std::vector<T> window(kernel_size);

    for (size_t i = 0; i < n; ++i) {
        // Collect window elements with zero-padding (matching scipy.signal.medfilt)
        int w = 0;
        for (int j = -k2; j <= k2; ++j) {
            ptrdiff_t idx = static_cast<ptrdiff_t>(i) + j;
            if (idx < 0 || static_cast<size_t>(idx) >= n)
                window[w++] = T(0);
            else
                window[w++] = src[static_cast<size_t>(idx)];
        }
        std::nth_element(window.begin(), window.begin() + k2, window.begin() + kernel_size);
        dst[i] = window[k2];
    }
}

}  // namespace signal
}  // namespace scipy
