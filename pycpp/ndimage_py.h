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
template <typename T>
inline py::array_t<T> gaussian_filter1d(const py::array_t<T>& input,
                                         T sigma = T(1),
                                         T truncate = T(4.0),
                                         const std::string& mode_str = "reflect",
                                         T cval = T(0)) {
    auto buf = input.request();
    py::array_t<T> result(buf.shape);
    const T* src = static_cast<const T*>(buf.ptr);
    T*       dst = static_cast<T*>(result.request().ptr);

    // Map Python mode string to integer mode
    int mode = 0;  // default: reflect
    if (mode_str == "reflect")   mode = 0;
    else if (mode_str == "constant") mode = 1;
    else if (mode_str == "nearest")  mode = 2;
    else if (mode_str == "mirror")   mode = 3;
    else if (mode_str == "wrap")     mode = 4;

    scipy::ndimage::gaussian_filter1d(src, dst, buf.size, sigma, truncate, mode, cval);
    return result;
}

}  // namespace ndimage
}  // namespace scipy_py
