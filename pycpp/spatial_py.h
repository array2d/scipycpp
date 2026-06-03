// Pybind11 wrappers for scipy/spatial.h — KDTree, cdist.
//
// Exports KDTree with query returning (distances, indices) matching Python cKDTree.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../scipy/spatial.h"
#include <vector>
#include <string>

namespace py = pybind11;

namespace scipy_py {
namespace spatial {

// ============================================================================
// Helper: py::array_t<T> with shape preserved, dtype = T
// ============================================================================

template <typename T>
inline void array_to_vector(const py::array_t<T>& arr, std::vector<T>& out) {
    auto buf = arr.request();
    const T* ptr = static_cast<const T*>(buf.ptr);
    out.assign(ptr, ptr + buf.size);
}

// ============================================================================
// cdist — cross-set distance matrix
// ============================================================================
// Python: scipy.spatial.distance.cdist(XA, XB, metric='euclidean')
// Returns mA × mB distance matrix.

template <typename T>
inline py::array_t<T> cdist(const py::array_t<T>& XA,
                              const py::array_t<T>& XB,
                              const std::string& metric = "euclidean") {
    auto ba = XA.request(), bb = XB.request();
    if (ba.ndim != 2 || bb.ndim != 2)
        throw std::runtime_error("cdist: XA and XB must be 2D arrays");
    if (static_cast<size_t>(ba.shape[1]) != static_cast<size_t>(bb.shape[1]))
        throw std::runtime_error("cdist: XA and XB must have same number of columns");

    size_t mA  = static_cast<size_t>(ba.shape[0]);
    size_t mB  = static_cast<size_t>(bb.shape[0]);
    size_t dim = static_cast<size_t>(ba.shape[1]);

    py::array_t<T> result({mA, mB});
    const T* a_ptr = static_cast<const T*>(ba.ptr);
    const T* b_ptr = static_cast<const T*>(bb.ptr);
    T*       r_ptr = static_cast<T*>(result.request().ptr);

    scipy::spatial::distance::cdist(a_ptr, mA, b_ptr, mB, dim, r_ptr, metric.c_str());
    return result;
}

// ============================================================================
// KDTree — matches Python scipy.spatial.cKDTree
// ============================================================================
// Python: tree = cKDTree(points)
//         distances, indices = tree.query(q, k=1)
//
// C++ KDTree wraps the native scipy::spatial::KDTree with pybind11.

template <typename T>
struct KDTreeWrap {
    std::vector<T> data;  // owned copy of points
    scipy::spatial::KDTree<T> tree;

    KDTreeWrap(const py::array_t<T>& points)
        : tree(nullptr, 0, 0)
    {
        auto buf = points.request();
        if (buf.ndim != 2)
            throw std::runtime_error("KDTree: points must be 2D array (n_pts × dim)");
        size_t n_pts = static_cast<size_t>(buf.shape[0]);
        int    dim   = static_cast<int>(buf.shape[1]);
        const T* ptr = static_cast<const T*>(buf.ptr);
        data.assign(ptr, ptr + n_pts * dim);
        tree = std::move(scipy::spatial::KDTree<T>(data.data(), n_pts, dim));
    }

    /// Python: tree.query(q, k=1) → returns (distances, indices)
    py::object query(const py::array_t<T>& q_arr, int k = 1,
                     double eps = 0.0, double p = 2.0,
                     double distance_upper_bound = std::numeric_limits<double>::infinity()) {
        auto buf = q_arr.request();
        if (static_cast<size_t>(buf.shape[0]) != static_cast<size_t>(tree.dim))
            throw std::runtime_error("KDTree.query: query point dimension mismatch");

        const T* q = static_cast<const T*>(buf.ptr);

        if (k == 1) {
            T d; size_t idx;
            tree.query(q, d, idx);
            // scipy returns scalars for k=1, not 1-element arrays
            return py::make_tuple(py::float_(static_cast<double>(d)),
                                  py::int_(static_cast<py::ssize_t>(idx)));
        } else {
            std::vector<T> dists(k);
            std::vector<size_t> indices(k);
            tree.query(q, dists.data(), indices.data(), k);

            py::array_t<T>            py_dists({k});
            py::array_t<py::ssize_t>  py_indices({k});
            T*            d_ptr = static_cast<T*>(py_dists.request().ptr);
            py::ssize_t*  i_ptr = static_cast<py::ssize_t*>(py_indices.request().ptr);
            for (int i = 0; i < k; ++i) {
                d_ptr[i] = dists[i];
                i_ptr[i] = static_cast<py::ssize_t>(indices[i]);
            }
            return py::make_tuple(py_dists, py_indices);
        }
    }
};

// ============================================================================
// Register KDTree type with pybind11 module
// ============================================================================

template <typename T>
inline void bind_kdtree(py::module_& m, const char* name) {
    py::class_<KDTreeWrap<T>>(m, name)
        .def(py::init<const py::array_t<T>&>(), py::arg("points"))
        .def("query", &KDTreeWrap<T>::query,
             py::arg("q"), py::arg("k") = 1,
             py::arg("eps") = 0.0, py::arg("p") = 2.0,
             py::arg("distance_upper_bound") = std::numeric_limits<double>::infinity(),
             "Query the kd-tree for nearest neighbors.\n"
             "Returns (distances, indices) tuple matching Python cKDTree.query.");
}

}  // namespace spatial
}  // namespace scipy_py
