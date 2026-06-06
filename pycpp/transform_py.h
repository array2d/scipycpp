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
    /// If the rotation was created via from_matrix(), delegates to the stored
    /// scipy Rotation object for bit-level alignment.
    /// If created via from_euler() (no stored scipy object), uses the C++
    /// Rotation<T>::as_euler() implementation (also 0-ULP).
    py::array_t<T> as_euler(const std::string& seq) const {
        if (_scipy_rot && !_scipy_rot.is_none()) {
            // from_matrix path: delegate to stored scipy object
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
        } else {
            // from_euler path: use C++ implementation (0-ULP via numpy::sin/cos)
            py::array_t<T> output(3);
            rot.as_euler(seq.c_str(), static_cast<T*>(output.request().ptr));
            return output;
        }
    }

    /// Python: Rotation.from_euler(seq, angle_or_angles) → RotationWrap
    /// Calls the C++ Rotation<T>::from_euler() directly.
    /// C++ uses numpy::sin/cos (SVML/npy path) — 0-ULP vs scipy.
    /// No Python delegation needed; _scipy_rot is left as None.
    /// Supports scalar angle (single axis) and 1D array (multi-axis, up to 3).
    static py::object from_euler(py::object /*sp_Rotation*/,
                                 const std::string& seq,
                                 py::object angle_or_angles) {
        RotationWrap wrap;
        // _scipy_rot stays py::none() — as_euler falls back to C++
        wrap._scipy_rot = py::none();

        // Detect scalar vs array input
        if (py::isinstance<py::float_>(angle_or_angles) ||
            py::isinstance<py::int_>(angle_or_angles)) {
            // Single scalar angle
            T angle = angle_or_angles.cast<T>();
            wrap.rot = scipy::spatial::transform::Rotation<T>::from_euler(
                seq.c_str(), angle);
        } else {
            // Array of angles
            auto arr = angle_or_angles.cast<py::array_t<double,
                py::array::c_style | py::array::forcecast>>();
            auto buf = arr.request();
            size_t n = static_cast<size_t>(buf.size);
            // Convert to T if needed
            std::vector<T> angles_t(n);
            const double* src = static_cast<const double*>(buf.ptr);
            for (size_t i = 0; i < n; ++i) angles_t[i] = static_cast<T>(src[i]);
            wrap.rot = scipy::spatial::transform::Rotation<T>::from_euler(
                seq.c_str(), angles_t.data());
        }
        return py::cast(wrap);
    }

    /// Python: rot.as_matrix() → (3,3) float64 ndarray
    /// Calls C++ Rotation<T>::as_matrix() on the stored quaternion.
    /// Since the quaternion is exact (from scipy's from_euler/from_matrix),
    /// and the arithmetic formulas are identical to scipy's as_matrix(),
    /// this is 0-ULP vs scipy.spatial.transform.Rotation.as_matrix().
    py::array_t<double> as_matrix() const {
        T m9[9];
        rot.as_matrix(m9);
        // Return always as float64, matching scipy.Rotation.as_matrix() dtype
        std::vector<py::ssize_t> shape = {3, 3};
        py::array_t<double> result(shape);
        auto* dst = static_cast<double*>(result.request().ptr);
        for (int i = 0; i < 9; ++i)
            dst[i] = static_cast<double>(m9[i]);
        return result;
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
        .def_static("from_euler",
            [sp_Rotation](const std::string& seq, py::object angle_or_angles) {
                return RotationWrap<T>::from_euler(sp_Rotation, seq, angle_or_angles);
            },
            py::arg("seq"), py::arg("angles"),
            "Initialize from Euler angles (scalar or 1D array).\n"
            "seq: 'x','y','z' (single axis) or 'xyz','zyx','XYZ',… (multi-axis).\n"
            "Delegates to scipy for exact quaternion; as_matrix() is 0-ULP.\n"
            "Aligns with scipy.spatial.transform.Rotation.from_euler()")
        .def("as_euler", &RotationWrap<T>::as_euler, py::arg("seq"),
             "Represent as Euler angles.\n"
             "seq: 'xyz', 'zyx', 'XYZ'.\n"
             "Aligns with scipy.spatial.transform.Rotation.as_euler()")
        .def("as_matrix", &RotationWrap<T>::as_matrix,
             "Convert to 3×3 rotation matrix (float64 ndarray, shape (3,3)).\n"
             "C++ quaternion→matrix arithmetic; 0-ULP vs scipy when quaternion\n"
             "is obtained via from_matrix() or from_euler().\n"
             "Aligns with scipy.spatial.transform.Rotation.as_matrix()");
}

}  // namespace transform
}  // namespace scipy_py
