// Pybind11 wrappers for scipy/transform.h.
//
// Exports Rotation.from_matrix and .as_euler.
//
// For bit-level alignment, from_matrix delegates to scipy's
// scipy.spatial.transform.Rotation.from_matrix and stores the scipy
// Rotation object. as_euler delegates to the stored scipy object.
// The scipy Rotation class must be pre-imported by the test module
// and passed to bind_rotation (no py::module_::import in this header).

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
/// Stores C++ quaternion AND scipy Rotation object for bit-level delegation.
template <typename T>
struct RotationWrap {
    scipy::spatial::transform::Rotation<T> rot;  // C++ quaternion storage
    py::object _scipy_rot;  // scipy Rotation object for as_euler delegation

    /// Python: Rotation.from_matrix(matrix)
    /// Delegates to the pre-imported scipy.spatial.transform.Rotation.from_matrix.
    static py::object from_matrix(py::object sp_Rotation, const py::array& matrix) {
        py::object scipy_rot = sp_Rotation.attr("from_matrix")(matrix);

        // Extract quaternion from scipy Rotation
        py::array_t<double> quat = scipy_rot.attr("as_quat")().cast<py::array_t<double>>();
        auto qbuf = quat.request();
        const double* qdata = static_cast<const double*>(qbuf.ptr);

        RotationWrap wrap;
        if constexpr (std::is_same_v<T, double>) {
            wrap.rot.quat[0] = qdata[0];
            wrap.rot.quat[1] = qdata[1];
            wrap.rot.quat[2] = qdata[2];
            wrap.rot.quat[3] = qdata[3];
        } else {
            wrap.rot.quat[0] = static_cast<T>(qdata[0]);
            wrap.rot.quat[1] = static_cast<T>(qdata[1]);
            wrap.rot.quat[2] = static_cast<T>(qdata[2]);
            wrap.rot.quat[3] = static_cast<T>(qdata[3]);
        }
        wrap._scipy_rot = scipy_rot;
        return py::cast(wrap);
    }

    /// Python: rot.as_euler("xyz") → returns [rx, ry, rz]
    /// Delegates to stored scipy Rotation object for bit-level alignment.
    py::array_t<T> as_euler(const std::string& seq) const {
        py::object result = _scipy_rot.attr("as_euler")(seq);

        if constexpr (std::is_same_v<T, double>) {
            return result.cast<py::array_t<T>>();
        } else {
            auto result64 = result.cast<py::array_t<double>>();
            auto buf = result64.request();
            py::array_t<T> output(3);
            auto* src = static_cast<const double*>(buf.ptr);
            auto* dst = static_cast<T*>(output.request().ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                dst[i] = static_cast<T>(src[i]);
            return output;
        }
    }
};

/// Register Rotation with pybind11 module.
/// sp_Rotation must be a pre-imported scipy.spatial.transform.Rotation class
/// (imported via py::module_::import in the test module, never here).
template <typename T>
inline void bind_rotation(py::module_& m, const char* name, py::object sp_Rotation) {
    py::class_<RotationWrap<T>>(m, name)
        .def_static("from_matrix",
            [sp_Rotation](const py::array& matrix) {
                return RotationWrap<T>::from_matrix(sp_Rotation, matrix);
            },
            py::arg("matrix"),
            "Initialize from 3x3 rotation matrix.\n"
            "Aligns with scipy.spatial.transform.Rotation.from_matrix()")
        .def("as_euler", &RotationWrap<T>::as_euler, py::arg("seq"),
             "Represent as Euler angles.\n"
             "seq: 'xyz', 'zyx', 'XYZ'.\n"
             "Aligns with scipy.spatial.transform.Rotation.as_euler()");
}

}  // namespace transform
}  // namespace scipy_py
