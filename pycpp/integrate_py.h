#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../scipy/integrate.h"
namespace py = pybind11;
namespace scipy::integrate {

template<typename T>
inline T trapezoid_py(const py::array_t<T>& y) {
    auto buf = y.request();
    return trapezoid<T>(static_cast<const T*>(buf.ptr), nullptr, static_cast<size_t>(buf.size));
}

// scipy.integrate.simpson always returns float64 regardless of input dtype.
// simpson<T> in integrate.h matches this: always returns double.
template<typename T>
inline double simpson_py(const py::array_t<T>& y) {
    auto buf = y.request();
    return simpson<T>(static_cast<const T*>(buf.ptr), nullptr, static_cast<size_t>(buf.size));
}

}  // namespace scipy::integrate
