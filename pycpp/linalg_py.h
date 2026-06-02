#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../scipy/linalg.h"
namespace py = pybind11;
namespace scipy::linalg {
template<typename T>
inline py::array_t<T> solve_py(const py::array_t<T>& A, const py::array_t<T>& b) {
    auto ba = A.request(), bb = b.request();
    int n = static_cast<int>(ba.shape[0]);
    py::array_t<T> result({n});
    solve(static_cast<const T*>(ba.ptr), static_cast<T*>(result.request().ptr),
          static_cast<const T*>(bb.ptr), n);
    return result;
}
}  // namespace scipy::linalg
