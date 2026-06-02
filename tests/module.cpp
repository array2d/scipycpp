#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../pycpp/stats_py.h"
#include "../pycpp/integrate_py.h"
#include "../pycpp/linalg_py.h"
#include "../pycpp/spatial_py.h"
#include "../pycpp/ndimage_py.h"
#include "../pycpp/signal_py.h"
#include "../pycpp/transform_py.h"
#include "numpy/svml_bridge.h"

namespace py = pybind11;

PYBIND11_MODULE(scipycpp, m) {
    m.doc() = "C++ bit-level alignment of Python scipy APIs";

    // SVML bridge init
    try {
        py::module_ np_core = py::module_::import("numpy.core._multiarray_umath");
        std::string umath_path = np_core.attr("__file__").cast<std::string>();
        numpy::svml::bridge_init(umath_path.c_str());
    } catch (...) {}

    // -- stats submodule --
    py::module_ stats = m.def_submodule("stats", "scipy.stats equivalents");
    py::class_<int> norm_cls(stats, "norm", "Frozen normal distribution");
    norm_cls.def_static("pdf",
        [](const py::array& x, py::object loc, py::object scale) -> py::object {
            auto buf = x.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                double _loc = loc.is_none() ? 0.0 : loc.cast<double>();
                double _scale = scale.is_none() ? 1.0 : scale.cast<double>();
                return scipy::stats::norm_pdf<double>(
                    x.cast<py::array_t<double>>(), _loc, _scale);
            } else {
                float _loc = loc.is_none() ? 0.0f : loc.cast<float>();
                float _scale = scale.is_none() ? 1.0f : scale.cast<float>();
                return scipy::stats::norm_pdf<float>(
                    x.cast<py::array_t<float>>(), _loc, _scale);
            }
        },
        py::arg("x"), py::arg("loc") = py::none(), py::arg("scale") = py::none(),
        "Probability density function.");

    // -- integrate submodule --
    m.def("trapezoid",
        [](const py::array_t<double>& y) { return scipy::integrate::trapezoid_py(y); });
    m.def("simpson",
        [](const py::array_t<double>& y) { return scipy::integrate::simpson(y.data(), y.size()); });

    // -- linalg submodule --
    py::module_ la = m.def_submodule("linalg", "scipy.linalg equivalents");
    la.def("solve",
        [](const py::array_t<double>& A, const py::array_t<double>& b) {
            return scipy::linalg::solve_py(A, b);
        });

    // -- spatial submodule --
    py::module_ sp = m.def_submodule("spatial", "scipy.spatial equivalents");
    py::module_ sp_dist = sp.def_submodule("distance", "scipy.spatial.distance equivalents");

    sp_dist.def("cdist",
        [](const py::array_t<double>& XA, const py::array_t<double>& XB,
           const std::string& metric) {
            return scipy_py::spatial::cdist(XA, XB, metric);
        },
        py::arg("XA"), py::arg("XB"), py::arg("metric") = "euclidean",
        "Cross-set distance matrix. Aligns with scipy.spatial.distance.cdist()");

    scipy_py::spatial::bind_kdtree<double>(sp, "KDTree");

    // -- ndimage submodule --
    py::module_ ndi = m.def_submodule("ndimage", "scipy.ndimage equivalents");
    ndi.def("gaussian_filter1d",
        [](const py::array_t<double>& input, double sigma, double truncate,
           const std::string& mode, double cval) {
            return scipy_py::ndimage::gaussian_filter1d(input, sigma, truncate, mode, cval);
        },
        py::arg("input"), py::arg("sigma") = 1.0, py::arg("truncate") = 4.0,
        py::arg("mode") = "reflect", py::arg("cval") = 0.0,
        "1-D Gaussian filter. Aligns with scipy.ndimage.gaussian_filter1d()");

    // -- signal submodule --
    py::module_ sig = m.def_submodule("signal", "scipy.signal equivalents");
    sig.def("medfilt",
        [](const py::array_t<double>& volume, int kernel_size) {
            return scipy_py::signal::medfilt(volume, kernel_size);
        },
        py::arg("volume"), py::arg("kernel_size") = 3,
        "1D median filter. Aligns with scipy.signal.medfilt()");

    // -- spatial.transform submodule --
    py::module_ sp_tf = sp.def_submodule("transform", "scipy.spatial.transform equivalents");
    scipy_py::transform::bind_rotation<double>(sp_tf, "Rotation");
}
