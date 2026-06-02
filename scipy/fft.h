// scipy.fft — based on pocketfft (the exact same C++ library numpy/scipy use).
//
// PocketFFT is a header-only C++11 FFT library by Martin Reinecke.
// It is the backend for numpy.fft and scipy.fft since numpy 1.17.
// Using it ensures bit-exact alignment with Python scipy.fft.
//
// Usage: #include "scipy/fft.h"

#pragma once

#include <pocketfft_hdronly.h>
#include <vector>
#include <complex>
#include <cstddef>

namespace scipy {
namespace fft {

// ============================================================================
// scipy.fft.fft — complex-to-complex FFT (uses pocketfft)
// ============================================================================

/// scipy.fft.fft(x, n=None, axis=-1, norm=None)
template<typename T>
inline void fft(const std::complex<T>* src, std::complex<T>* dst, size_t n) {
    std::vector<std::complex<T>> tmp(src, src + n);
    std::vector<size_t> shape = {n};
    std::vector<ptrdiff_t> stride = {1};
    pocketfft::c2c(shape, stride, stride, {0}, {0}, {0}, true,
                   tmp.data(), tmp.data(), T(1));
    for (size_t i = 0; i < n; ++i) dst[i] = tmp[i];
}

/// scipy.fft.ifft(x, n=None, axis=-1, norm=None)
template<typename T>
inline void ifft(const std::complex<T>* src, std::complex<T>* dst, size_t n) {
    std::vector<std::complex<T>> tmp(src, src + n);
    std::vector<size_t> shape = {n};
    std::vector<ptrdiff_t> stride = {1};
    pocketfft::c2c(shape, stride, stride, {0}, {0}, {0}, false,
                   tmp.data(), tmp.data(), T(1) / T(n));
    for (size_t i = 0; i < n; ++i) dst[i] = tmp[i];
}

/// scipy.fft.rfft — real-to-complex FFT
template<typename T>
inline void rfft(const T* src, std::complex<T>* dst, size_t n) {
    std::vector<T> tmp(src, src + n);
    std::vector<size_t> shape = {n};
    std::vector<ptrdiff_t> stride = {1};
    size_t n_out = n / 2 + 1;
    std::vector<std::complex<T>> out(n_out);
    pocketfft::r2c(shape, stride, stride, {0}, {0}, true,
                   tmp.data(), out.data(), T(1));
    for (size_t i = 0; i < n_out; ++i) dst[i] = out[i];
}

/// scipy.fft.irfft — complex-to-real IFFT
template<typename T>
inline void irfft(const std::complex<T>* src, T* dst, size_t n) {
    size_t n_in = n / 2 + 1;
    std::vector<std::complex<T>> tmp(src, src + n_in);
    std::vector<size_t> shape = {n};
    std::vector<ptrdiff_t> stride = {1};
    pocketfft::c2r(shape, stride, stride, {0}, {0}, false,
                   tmp.data(), dst, T(1) / T(n));
}

/// scipy.fft.fftfreq(n, d=1.0)
template<typename T>
inline void fftfreq(T* freqs, size_t n, T d = T(1)) {
    T val = T(1) / (T(n) * d);
    for (size_t i = 0; i < (n + 1) / 2; ++i)
        freqs[i] = T(i) * val;
    for (size_t i = (n + 1) / 2; i < n; ++i)
        freqs[i] = T(static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(n)) * val;
}

/// scipy.fft.rfftfreq(n, d=1.0)
template<typename T>
inline void rfftfreq(T* freqs, size_t n, T d = T(1)) {
    T val = T(1) / (T(n) * d);
    for (size_t i = 0; i <= n / 2; ++i)
        freqs[i] = T(i) * val;
}

}  // namespace fft
}  // namespace scipy
