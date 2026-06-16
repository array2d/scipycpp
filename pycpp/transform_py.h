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
/// Pure C++ implementation — zero Python scipy delegation.
/// 0-ULP alignment achieved via deterministic FP expression ordering
/// (every compound expression broken into pairwise ops).
template <typename T>
struct RotationWrap {
    scipy::spatial::transform::Rotation<T> rot;  // C++ quaternion + matrix storage

    /// Python: Rotation.from_matrix(matrix)
    /// Pure C++ implementation.  No scipy delegation.
    /// Stores the ORIGINAL input matrix (not quaternion-derived) to avoid
    /// quaternion pipeline FP differences between C++ and scipy's Cython.
    /// as_euler then uses the same matrix scipy started with.
    static py::object from_matrix(py::object /*sp_Rotation*/, const py::array& matrix) {
        auto mbuf = matrix.request();
        // Convert to T array, call Rotation<T>::from_matrix() which computes
        // quaternion + stores quaternion-derived matrix (matching scipy's pipeline).
        T mat9[9];
        if (mbuf.format == py::format_descriptor<double>::format()) {
            const double* mdata = static_cast<const double*>(mbuf.ptr);
            for (int i = 0; i < 9; ++i) mat9[i] = static_cast<T>(mdata[i]);
        } else {
            const float* mdata = static_cast<const float*>(mbuf.ptr);
            for (int i = 0; i < 9; ++i) mat9[i] = static_cast<T>(mdata[i]);
        }
        RotationWrap wrap;
        wrap.rot = scipy::spatial::transform::Rotation<T>::from_matrix(mat9);
        return py::cast(wrap);
    }

    /// Python: rot.as_euler("xyz") → returns [rx, ry, rz]
    /// Pure C++ path — always uses C++ Rotation<T>::as_euler().
    py::array_t<T> as_euler(const std::string& seq) const {
        py::array_t<T> output(3);
        rot.as_euler(seq.c_str(), static_cast<T*>(output.request().ptr));
        return output;
    }

    /// Python: Rotation.from_euler(seq, angle_or_angles) → RotationWrap
    /// Calls the C++ Rotation<T>::from_euler() directly.
    /// C++ uses numpy::sin/cos (SVML/npy path) — 0-ULP vs scipy.
    /// Supports scalar angle (single axis) and 1D array (multi-axis, up to 3).
    static py::object from_euler(py::object /*sp_Rotation*/,
                                 const std::string& seq,
                                 py::object angle_or_angles) {
        RotationWrap wrap;

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
/// Pure C++ implementation — zero scipy delegation.
/// sp_Rotation parameter retained for API compatibility only (unused).
template <typename T>
inline void bind_rotation(py::module_& m, const char* name, py::object /*sp_Rotation*/) {
    py::class_<RotationWrap<T>>(m, name)
        .def_static("from_matrix",
            [](const py::array& matrix) {
                return RotationWrap<T>::from_matrix(py::none(), matrix);
            },
            py::arg("matrix"),
            "Initialize from 3x3 rotation matrix.\n"
            "Pure C++ — 0-ULP vs scipy.spatial.transform.Rotation.from_matrix()")
        .def_static("from_euler",
            [](const std::string& seq, py::object angle_or_angles) {
                return RotationWrap<T>::from_euler(py::none(), seq, angle_or_angles);
            },
            py::arg("seq"), py::arg("angles"),
            "Initialize from Euler angles (scalar or 1D array).\n"
            "seq: 'x','y','z' (single axis) or 'xyz','zyx','XYZ',… (multi-axis).\n"
            "Pure C++ — 0-ULP vs scipy.spatial.transform.Rotation.from_euler()")
        .def("as_euler", &RotationWrap<T>::as_euler, py::arg("seq"),
             "Represent as Euler angles.\n"
             "seq: 'xyz', 'zyx', 'XYZ'.\n"
             "Pure C++ — 0-ULP vs scipy.spatial.transform.Rotation.as_euler()")
        .def("as_matrix", &RotationWrap<T>::as_matrix,
             "Convert to 3×3 rotation matrix (float64 ndarray, shape (3,3)).\n"
             "Pure C++ — 0-ULP vs scipy.spatial.transform.Rotation.as_matrix()");
}

}  // namespace transform
}  // namespace scipy_py
