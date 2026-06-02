// scipy.signal — convolution, correlation.
//
// Usage: #include "scipy/signal.h"

#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

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

}  // namespace signal
}  // namespace scipy
