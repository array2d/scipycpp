// Pybind11 wrappers for scipy/transform.h.
//
// Exports Rotation.from_matrix and .as_euler.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../scipy/transform.h"
#include <vector>
#include <string>

namespace py = pybind11;

namespace scipy_py {
namespace transform {

/// Python: scipy.spatial.transform.Rotation
template <typename T>
struct RotationWrap {
    scipy::spatial::transform::Rotation<T> rot;

    /// Python: Rotation.from_matrix(matrix)
    /// matrix is 3x3 or Nx3x3 numpy array
    static py::object from_matrix(const py::array_t<T>& matrix) {
        auto buf = matrix.request();

        // If Nx3x3 array, return list of Rotations
        if (buf.ndim == 3) {
            size_t N = static_cast<size_t>(buf.shape[0]);
            py::list result;
            const T* ptr = static_cast<const T*>(buf.ptr);
            for (size_t i = 0; i < N; ++i) {
                auto r = scipy::spatial::transform::Rotation<T>::from_matrix(ptr + i * 9);
                result.append(RotationWrap{r});
            }
            return result;
        }

        // Single 3x3 matrix
        auto r = scipy::spatial::transform::Rotation<T>::from_matrix(
            static_cast<const T*>(buf.ptr));
        return py::cast(RotationWrap{r});
    }

    /// Python: rot.as_euler("xyz") → returns [rx, ry, rz]
    py::array_t<T> as_euler(const std::string& seq) const {
        std::vector<T> euler(3);
        rot.as_euler(seq.c_str(), euler.data());
        return py::array_t<T>({3}, euler.data());
    }
};

/// Register Rotation with pybind11 module
template <typename T>
inline void bind_rotation(py::module_& m, const char* name) {
    py::class_<RotationWrap<T>>(m, name)
        .def_static("from_matrix", &RotationWrap<T>::from_matrix, py::arg("matrix"),
                    "Initialize from 3x3 rotation matrix.\n"
                    "Aligns with scipy.spatial.transform.Rotation.from_matrix()")
        .def("as_euler", &RotationWrap<T>::as_euler, py::arg("seq"),
             "Represent as Euler angles.\n"
             "seq: 'xyz', 'zyx', 'XYZ'.\n"
             "Aligns with scipy.spatial.transform.Rotation.as_euler()");
}

}  // namespace transform
}  // namespace scipy_py
