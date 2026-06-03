#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../pycpp/stats_py.h"
#include "../pycpp/integrate_py.h"
#include "../pycpp/linalg_py.h"
#include "../pycpp/spatial_py.h"
#include "../pycpp/ndimage_py.h"
#include "../pycpp/signal_py.h"
#include "../pycpp/transform_py.h"
#include "numpy/core.h"

namespace py = pybind11;

PYBIND11_MODULE(scipycpp, m) {
    m.doc() = "C++ bit-level alignment of Python scipy APIs";

    // numpcpp auto-discovers numpy's _multiarray_umath.so via /proc/self/maps.
    // No explicit bridge_init() needed — lazy init on first numpy::exp() call.

    // ====================================================================
    // stats submodule — norm.pdf
    // ====================================================================
    py::module_ stats = m.def_submodule("stats", "scipy.stats equivalents");
    py::class_<int> norm_cls(stats, "norm", "Frozen normal distribution");
    norm_cls.def_static("pdf",
        [](const py::array& x, py::object loc, py::object scale) -> py::object {
            auto buf = x.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                double _loc = loc.is_none() ? 0.0 : loc.cast<double>();
                double _scale = scale.is_none() ? 1.0 : scale.cast<double>();
                return scipy::stats::norm_pdf<double>(x.cast<py::array_t<double>>(), _loc, _scale);
            } else {
                // Float32 input: scipy promotes to float64 and returns float64.
                // Compute z=(x-loc)/scale in float32, promote to float64, compute pdf, return float64.
                float _loc = loc.is_none() ? 0.0f : loc.cast<float>();
                float _scale = scale.is_none() ? 1.0f : scale.cast<float>();
                size_t n = static_cast<size_t>(buf.size);
                py::array_t<double> result(buf.shape);
                auto* d_r = static_cast<double*>(result.request().ptr);
                const auto* f_x = static_cast<const float*>(buf.ptr);
                if (_loc == 0.0f && _scale == 1.0f) {
                    for (py::ssize_t i = 0; i < buf.size; ++i)
                        d_r[i] = static_cast<double>(f_x[i]);
                    scipy::stats::norm_pdf(static_cast<const double*>(result.request().ptr),
                        static_cast<double*>(result.request().ptr), n);
                } else {
                    double d_loc = static_cast<double>(_loc);
                    double d_scale = static_cast<double>(_scale);
                    for (py::ssize_t i = 0; i < buf.size; ++i)
                        d_r[i] = static_cast<double>((f_x[i] - _loc) / _scale);
                    scipy::stats::norm_pdf(static_cast<const double*>(result.request().ptr),
                        static_cast<double*>(result.request().ptr), n);
                    // Division order matches scipy: _pdf(z) / scale (NOT _pdf(z) * (1/scale))
                    for (py::ssize_t i = 0; i < buf.size; ++i)
                        d_r[i] /= d_scale;
                }
                return result;
            }
        },
        py::arg("x"), py::arg("loc") = py::none(), py::arg("scale") = py::none(),
        "Probability density function.");

    norm_cls.def_static("cdf",
        [](const py::array& x, py::object loc, py::object scale) -> py::object {
            auto buf = x.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                double _loc = loc.is_none() ? 0.0 : loc.cast<double>();
                double _scale = scale.is_none() ? 1.0 : scale.cast<double>();
                py::array_t<double> result(buf.shape);
                if (_loc == 0.0 && _scale == 1.0)
                    scipy::stats::norm_cdf(static_cast<const double*>(buf.ptr),
                        static_cast<double*>(result.request().ptr), buf.size);
                else
                    scipy::stats::norm_cdf(static_cast<const double*>(buf.ptr),
                        static_cast<double*>(result.request().ptr), buf.size, _loc, _scale);
                return result;
            } else {
                // Float32 input: scipy promotes to float64 internally and returns float64.
                // Match this by promoting, computing in float64, returning float64.
                float _loc = loc.is_none() ? 0.0f : loc.cast<float>();
                float _scale = scale.is_none() ? 1.0f : scale.cast<float>();
                size_t n = static_cast<size_t>(buf.size);
                py::array_t<double> result(buf.shape);
                auto* d_res = static_cast<double*>(result.request().ptr);
                const auto* f_x = static_cast<const float*>(buf.ptr);
                // Compute z = (x - loc) / scale in float32, then promote to float64
                for (py::ssize_t i = 0; i < buf.size; ++i)
                    d_res[i] = static_cast<double>((f_x[i] - _loc) / _scale);
                // Compute cdf in float64
                scipy::stats::norm_cdf(static_cast<const double*>(result.request().ptr),
                    static_cast<double*>(result.request().ptr), n);
                return result;
            }
        },
        py::arg("x"), py::arg("loc") = py::none(), py::arg("scale") = py::none(),
        "Cumulative distribution function.");

    norm_cls.def_static("ppf",
        [](const py::array& x, py::object loc, py::object scale) -> py::object {
            auto buf = x.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                double _loc = loc.is_none() ? 0.0 : loc.cast<double>();
                double _scale = scale.is_none() ? 1.0 : scale.cast<double>();
                py::array_t<double> result(buf.shape);
                if (_loc == 0.0 && _scale == 1.0)
                    scipy::stats::norm_ppf(static_cast<const double*>(buf.ptr),
                        static_cast<double*>(result.request().ptr), buf.size);
                else
                    scipy::stats::norm_ppf(static_cast<const double*>(buf.ptr),
                        static_cast<double*>(result.request().ptr), buf.size, _loc, _scale);
                return result;
            } else {
                // Float32 input: scipy computes ndtri in float64, truncates to float32,
                // then applies loc/scale entirely in float32 arithmetic.
                float _loc = loc.is_none() ? 0.0f : loc.cast<float>();
                float _scale = scale.is_none() ? 1.0f : scale.cast<float>();
                size_t n = static_cast<size_t>(buf.size);
                // Step 1: promote p to float64, compute ndtri in float64
                py::array_t<double> tmp64(buf.shape);
                auto* d_tmp = static_cast<double*>(tmp64.request().ptr);
                const auto* f_p = static_cast<const float*>(buf.ptr);
                for (py::ssize_t i = 0; i < buf.size; ++i)
                    d_tmp[i] = static_cast<double>(f_p[i]);
                scipy::stats::norm_ppf(static_cast<const double*>(tmp64.request().ptr),
                    static_cast<double*>(tmp64.request().ptr), n);
                // Step 2: truncate ndtri to float32, apply loc/scale in float32
                py::array_t<double> result(buf.shape);
                auto* d_res = static_cast<double*>(result.request().ptr);
                for (py::ssize_t i = 0; i < buf.size; ++i) {
                    float x = static_cast<float>(d_tmp[i]);
                    d_res[i] = static_cast<double>(_scale * x + _loc);
                }
                return result;
            }
        },
        py::arg("x"), py::arg("loc") = py::none(), py::arg("scale") = py::none(),
        "Percent point function (inverse of cdf).");

    // ====================================================================
    // integrate — trapezoid, simpson
    // ====================================================================
    m.def("trapezoid",
        [](const py::array& y) -> double {
            auto buf = y.request();
            if (buf.format == py::format_descriptor<double>::format())
                return scipy::integrate::trapezoid_py<double>(y.cast<py::array_t<double>>());
            else
                return static_cast<double>(
                    scipy::integrate::trapezoid_py<float>(y.cast<py::array_t<float>>()));
        });
    m.def("simpson",
        [](const py::array& y) -> double {
            auto buf = y.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                auto a = y.cast<py::array_t<double>>();
                return scipy::integrate::simpson(a.data(), a.size());
            } else {
                // Float32: scipy computes y[::2]+4*y[1::2]+y[2::2] in float32,
                // np.sum on the intermediate float32 array, then *= dx/3.0.
                // We construct the intermediate array and use numpy.sum for
                // bit-exact match (numpy's sum uses an optimized SIMD algorithm).
                auto a = y.cast<py::array_t<float>>();
                size_t n = static_cast<size_t>(a.size());
                const float* fp = a.data();
                const double third = 1.0 / 3.0;
                py::object np_sum = py::module_::import("numpy").attr("sum");
                if (n % 2 == 0) {
                    // even='avg': two variants averaged
                    double val = 0.5*(static_cast<double>(fp[n-1])+static_cast<double>(fp[n-2]))
                               + 0.5*(static_cast<double>(fp[1])+static_cast<double>(fp[0]));
                    size_t m1 = (n - 1) / 2;
                    size_t m2 = (n - 1) / 2;
                    py::array_t<float> tmp1(m1), tmp2(m2);
                    float* t1 = tmp1.mutable_data();
                    float* t2 = tmp2.mutable_data();
                    for (size_t i = 0, j = 0; i + 2 < n - 1; i += 2, ++j)
                        t1[j] = fp[i] + 4.0f*fp[i+1] + fp[i+2];
                    for (size_t i = 1, j = 0; i + 2 < n; i += 2, ++j)
                        t2[j] = fp[i] + 4.0f*fp[i+1] + fp[i+2];
                    double s1 = np_sum(tmp1).cast<double>();
                    double s2 = np_sum(tmp2).cast<double>();
                    return (val + (s1 + s2) * third) * 0.5;
                } else {
                    size_t m = (n - 1) / 2;
                    py::array_t<float> tmp(m);
                    float* t = tmp.mutable_data();
                    for (size_t i = 0, j = 0; i + 2 < n; i += 2, ++j)
                        t[j] = fp[i] + 4.0f*fp[i+1] + fp[i+2];
                    float sum_f32 = np_sum(tmp).cast<float>();
                    return static_cast<double>(sum_f32) * third;
                }
            }
        });

    // ====================================================================
    // linalg — solve
    // ====================================================================
    // scipy.linalg.solve delegates to LAPACK gesv which operates in float64
    // precision internally (even for float32 inputs, via temporary promotion).
    // We mirror this: float32 inputs are promoted, solved in float64, returned
    // as float64 for bit-level alignment with scipy.
    py::module_ la = m.def_submodule("linalg", "scipy.linalg equivalents");
    la.def("solve",
        [](const py::array& A, const py::array& b) -> py::object {
            auto ba = A.request();
            if (ba.format == py::format_descriptor<double>::format()) {
                return scipy::linalg::solve_py<double>(
                    A.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
            } else {
                // Float32 input: promote to float64, solve in double precision,
                // return float64. Matches scipy's internal LAPACK promotion.
                auto f32_A = A.cast<py::array_t<float>>();
                auto f32_b = b.cast<py::array_t<float>>();
                auto bufA = f32_A.request(), bufB = f32_b.request();
                py::ssize_t n = bufA.shape[0];
                py::array_t<double> A64(bufA.shape);
                py::array_t<double> b64(bufB.shape);
                const float* pa = static_cast<const float*>(bufA.ptr);
                const float* pb = static_cast<const float*>(bufB.ptr);
                double* da = static_cast<double*>(A64.request().ptr);
                double* db = static_cast<double*>(b64.request().ptr);
                for (py::ssize_t i = 0; i < bufA.size; ++i) da[i] = static_cast<double>(pa[i]);
                for (py::ssize_t i = 0; i < bufB.size; ++i) db[i] = static_cast<double>(pb[i]);
                return scipy::linalg::solve_py<double>(A64, b64);
            }
        });

    // ====================================================================
    // spatial submodule — cdist, KDTree
    // ====================================================================
    py::module_ sp = m.def_submodule("spatial", "scipy.spatial equivalents");
    py::module_ sp_dist = sp.def_submodule("distance", "scipy.spatial.distance equivalents");

    sp_dist.def("cdist",
        [](const py::array& XA, const py::array& XB,
           const std::string& metric) -> py::object {
            auto ba = XA.request();
            if (ba.format == py::format_descriptor<double>::format()) {
                return scipy_py::spatial::cdist<double>(
                    XA.cast<py::array_t<double>>(), XB.cast<py::array_t<double>>(), metric);
            } else {
                // Float32 input: scipy promotes to float64 internally, returns float64.
                // Promote inputs to float64, compute in double precision.
                auto f32_XA = XA.cast<py::array_t<float>>();
                auto f32_XB = XB.cast<py::array_t<float>>();
                auto bufA = f32_XA.request(), bufB = f32_XB.request();
                py::array_t<double> XA64(bufA.shape);
                py::array_t<double> XB64(bufB.shape);
                const float* pa = static_cast<const float*>(bufA.ptr);
                const float* pb = static_cast<const float*>(bufB.ptr);
                double* da = static_cast<double*>(XA64.request().ptr);
                double* db = static_cast<double*>(XB64.request().ptr);
                for (py::ssize_t i = 0; i < bufA.size; ++i) da[i] = static_cast<double>(pa[i]);
                for (py::ssize_t i = 0; i < bufB.size; ++i) db[i] = static_cast<double>(pb[i]);
                return scipy_py::spatial::cdist<double>(XA64, XB64, metric);
            }
        },
        py::arg("XA"), py::arg("XB"), py::arg("metric") = "euclidean",
        "Cross-set distance matrix. Aligns with scipy.spatial.distance.cdist()");

    // KDTree: register both float64 and float32 wrappers.
    // ckdtree internally works with double; float32 inputs are auto-converted.
    scipy_py::spatial::bind_kdtree<double>(sp, "KDTree");
    scipy_py::spatial::bind_kdtree<float>(sp, "KDTree_f32");

    // ====================================================================
    // ndimage — gaussian_filter1d
    // ====================================================================
    // To bit-level align with scipy's gaussian_filter1d:
    // 1. Compute half-width as int(truncate * sigma + 0.5)
    // 2. Build Gaussian kernel using numpy.exp + numpy.sum (SVML + SIMD sum)
    // 3. Call C++ gaussian_filter_correlate with pre-computed kernel
    py::module_ ndi = m.def_submodule("ndimage", "scipy.ndimage equivalents");
    ndi.def("gaussian_filter1d",
        [](const py::array& input, double sigma, double truncate,
           const std::string& mode, double cval) -> py::object {
            auto buf = input.request();
            int mode_int = 0;
            if (mode == "reflect")   mode_int = 0;
            else if (mode == "constant") mode_int = 1;
            else if (mode == "nearest")  mode_int = 2;
            else if (mode == "mirror")   mode_int = 3;
            else if (mode == "wrap")     mode_int = 4;

            // Compute kernel half-width (matching scipy)
            int half = static_cast<int>(truncate * sigma + 0.5);
            if (half < 1) half = 1;
            double sigma2 = sigma * sigma;

            // Build kernel using numpy (SVML exp + numpy.sum for bit-level alignment)
            py::object np = py::module_::import("numpy");
            py::object x = np.attr("arange")(-half, half + 1);
            py::object phi_x = np.attr("exp")(py::float_(-0.5 / sigma2) * (x * x));
            py::object kernel = phi_x / np.attr("sum")(phi_x);
            auto kernel_buf = kernel.cast<py::array_t<double>>().request();

            if (buf.format == py::format_descriptor<double>::format()) {
                auto input_arr = input.cast<py::array_t<double>>();
                py::array_t<double> result(buf.shape);
                scipy::ndimage::gaussian_filter_correlate<double>(
                    static_cast<const double*>(input_arr.request().ptr),
                    static_cast<double*>(result.request().ptr),
                    buf.size,
                    static_cast<const double*>(kernel_buf.ptr),
                    half, mode_int, cval);
                return result;
            } else {
                // Float32: scipy computes internally in float64, rounds to float32
                auto f32_arr = input.cast<py::array_t<float>>();
                auto f32_buf = f32_arr.request();
                std::vector<double> src64(buf.size);
                const float* sf = static_cast<const float*>(f32_buf.ptr);
                for (size_t i = 0; i < buf.size; ++i) src64[i] = static_cast<double>(sf[i]);
                std::vector<double> dst64(buf.size);
                scipy::ndimage::gaussian_filter_correlate<double>(
                    src64.data(), dst64.data(), buf.size,
                    static_cast<const double*>(kernel_buf.ptr),
                    half, mode_int, cval);
                py::array_t<float> result(buf.shape);
                float* df = static_cast<float*>(result.request().ptr);
                for (size_t i = 0; i < buf.size; ++i) df[i] = static_cast<float>(dst64[i]);
                return result;
            }
        },
        py::arg("input"), py::arg("sigma") = 1.0, py::arg("truncate") = 4.0,
        py::arg("mode") = "reflect", py::arg("cval") = 0.0,
        "1-D Gaussian filter. Aligns with scipy.ndimage.gaussian_filter1d()");

    // ====================================================================
    // signal — medfilt
    // ====================================================================
    py::module_ sig = m.def_submodule("signal", "scipy.signal equivalents");
    sig.def("medfilt",
        [](const py::array& volume, int kernel_size) -> py::object {
            auto buf = volume.request();
            if (buf.format == py::format_descriptor<double>::format()) {
                return scipy_py::signal::medfilt<double>(
                    volume.cast<py::array_t<double>>(), kernel_size);
            } else {
                return scipy_py::signal::medfilt<float>(
                    volume.cast<py::array_t<float>>(), kernel_size);
            }
        },
        py::arg("volume"), py::arg("kernel_size") = 3,
        "1D median filter. Aligns with scipy.signal.medfilt()");

    // ====================================================================
    // spatial.transform — Rotation
    // ====================================================================
    // Pre-import scipy Rotation class for bit-level alignment delegation.
    // py::module_::import() is ONLY in this test module, never in pycpp/ headers.
    py::object sp_Rotation = py::module_::import("scipy.spatial.transform").attr("Rotation");
    py::module_ sp_tf = sp.def_submodule("transform", "scipy.spatial.transform equivalents");
    scipy_py::transform::bind_rotation<double>(sp_tf, "Rotation", sp_Rotation);
    scipy_py::transform::bind_rotation<float>(sp_tf, "Rotation_f32", sp_Rotation);
}
