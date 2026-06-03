// Pybind11 wrappers for scipy/ndimage.h.
//
// Exports gaussian_filter1d.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../scipy/ndimage.h"

namespace py = pybind11;

namespace scipy_py {
namespace ndimage {

/// Python: scipy.ndimage.gaussian_filter1d(input, sigma, ...)
///
/// For float32: scipy computes internally in float64, then rounds to float32.
/// We replicate by promoting float32 → float64, computing in float64,
/// then rounding back to float32.
template <typename T>
inline py::array_t<T> gaussian_filter1d(const py::array_t<T>& input,
                                         T sigma = T(1),
                                         T truncate = T(4.0),
                                         const std::string& mode_str = "reflect",
                                         T cval = T(0)) {
    auto buf = input.request();
    int mode = 0;
    if (mode_str == "reflect")   mode = 0;
    else if (mode_str == "constant") mode = 1;
    else if (mode_str == "nearest")  mode = 2;
    else if (mode_str == "mirror")   mode = 3;
    else if (mode_str == "wrap")     mode = 4;

    if constexpr (std::is_same_v<T, float>) {
        // Float32 → promote to float64, compute, round back
        std::vector<double> src64(buf.size);
        const float* sf = static_cast<const float*>(buf.ptr);
        for (size_t i = 0; i < buf.size; ++i) src64[i] = static_cast<double>(sf[i]);
        std::vector<double> dst64(buf.size);
        scipy::ndimage::gaussian_filter1d(src64.data(), dst64.data(), buf.size,
            static_cast<double>(sigma), static_cast<double>(truncate),
            mode, static_cast<double>(cval));
        py::array_t<float> result(buf.shape);
        float* df = static_cast<float*>(result.request().ptr);
        for (size_t i = 0; i < buf.size; ++i) df[i] = static_cast<float>(dst64[i]);
        return result;
    } else {
        py::array_t<T> result(buf.shape);
        const T* src = static_cast<const T*>(buf.ptr);
        T*       dst = static_cast<T*>(result.request().ptr);
        scipy::ndimage::gaussian_filter1d(src, dst, buf.size, sigma, truncate, mode, cval);
        return result;
    }
}

}  // namespace ndimage
}  // namespace scipy_py
