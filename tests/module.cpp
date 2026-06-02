#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../pycpp/stats_py.h"
#include "../pycpp/integrate_py.h"
#include "../pycpp/linalg_py.h"
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
}
