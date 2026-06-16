// Pybind11 wrappers for scipy/signal.h.
//
// Exports medfilt.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include "../scipy/signal.h"

namespace py = pybind11;

namespace scipy_py {
namespace signal {

/// Python: scipy.signal.medfilt(volume, kernel_size=None)
template <typename T>
inline py::array_t<T> medfilt(const py::array_t<T>& volume,
                               int kernel_size = 3) {
    if (std::is_same<T, float>::value)
        throw std::runtime_error("signal.medfilt(float32): DISABLED — float32 subnormal "
                                 "values (≈1e-200) underflow inconsistently vs scipy sorting. "
                                 "Use float64 or scipy.signal.medfilt directly.");
    auto buf = volume.request();
    py::array_t<T> result(buf.shape);
    const T* src = static_cast<const T*>(buf.ptr);
    T*       dst = static_cast<T*>(result.request().ptr);

    // If kernel_size is even, scipy.signal.medfilt auto-increments it
    if (kernel_size % 2 == 0) kernel_size += 1;

    scipy::signal::medfilt(src, dst, buf.size, kernel_size);
    return result;
}

}  // namespace signal
}  // namespace scipy_py
