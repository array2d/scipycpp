// Pybind11 wrappers for scipy/stats.h.
//
// Exports norm_pdf<T> with specializations for double/float.
// No _f32 suffixes — same name, different types.

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

template <>
inline py::array_t<float> norm_pdf<float>(const py::array_t<float>& x,
                                            float loc, float scale) {
    auto buf = x.request();
    py::array_t<float> result(buf.shape);
    const float* src = static_cast<const float*>(buf.ptr);
    float*       dst = static_cast<float*>(result.request().ptr);

    if (loc == 0.0f && scale == 1.0f) {
        py::array_t<double> x64(buf.shape), r64(buf.shape);
        for (py::ssize_t i = 0; i < buf.size; ++i)
            static_cast<double*>(x64.request().ptr)[i] = static_cast<double>(src[i]);
        norm_pdf(static_cast<const double*>(x64.request().ptr),
                 static_cast<double*>(r64.request().ptr), buf.size);
        for (py::ssize_t i = 0; i < buf.size; ++i)
            dst[i] = static_cast<float>(static_cast<const double*>(r64.request().ptr)[i]);
    } else {
        static const double sqrt_2pi = std::sqrt(2.0 * M_PI);
        double scale64 = static_cast<double>(scale);
        for (py::ssize_t i = 0; i < buf.size; ++i) {
            float z = (src[i] - loc) / scale;
            double z64 = static_cast<double>(z);
            double arg = -(z64 * z64) / 2.0;
            dst[i] = static_cast<float>(numpy::svml::exp(arg) / sqrt_2pi / scale64);
        }
    }
    return result;
}

}  // namespace scipy::stats
