// Pybind11 wrappers for scipy/stats.h.
//
// norm_pdf<double> — native float64 path.
// norm_pdf<float>  — removed; float32 promotion handled in module.cpp.
//   Scipy always promotes float32 to float64 internally and returns float64.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../scipy/stats.h"

namespace py = pybind11;

namespace scipy::stats {

template <typename T>
inline py::array_t<T> norm_pdf(const py::array_t<T>& x, T loc = T(0), T scale = T(1));

template <>
inline py::array_t<double> norm_pdf<double>(const py::array_t<double>& x,
                                             double loc, double scale) {
    auto buf = x.request();
    py::array_t<double> result(buf.shape);
    if (loc == 0.0 && scale == 1.0) {
        norm_pdf(static_cast<const double*>(buf.ptr),
                 static_cast<double*>(result.request().ptr), buf.size);
    } else {
        norm_pdf(static_cast<const double*>(buf.ptr),
                 static_cast<double*>(result.request().ptr), buf.size, loc, scale);
    }
    return result;
}

// Float32 specialization removed — float32→float64 promotion handled in
// module.cpp to match scipy's internal promotion. See module.cpp pdf lambda.

}  // namespace scipy::stats
