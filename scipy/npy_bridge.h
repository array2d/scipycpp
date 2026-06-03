// Direct bridge to numpy's npy_* math functions (bypasses SVML).
//
// scipy's Cephes is compiled with #define exp npy_exp, so it uses npy_exp
// directly. The numpy:: bridge in core.h resolves to __svml_exp8 on AVX-512
// machines, which produces different ULP results. This header provides direct
// access to npy_* functions for bit-level alignment with scipy.
//
// Usage: #include "scipy/npy_bridge.h"
//   npy_bridge::exp(x)   // uses npy_exp from numpy's umath .so

#pragma once

#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <string>

namespace scipy {
namespace npy_bridge {

// Auto-discover numpy's _multiarray_umath.so path via /proc/self/maps.
inline const char* find_umath_path() {
    static std::string path;
    static bool tried = false;
    if (tried) return path.empty() ? nullptr : path.c_str();
    tried = true;

    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("_multiarray_umath") != std::string::npos &&
            line.find(".so") != std::string::npos) {
            auto pos = line.rfind('/');
            auto start = line.rfind(' ', pos);
            if (start != std::string::npos && pos != std::string::npos) {
                path = line.substr(start + 1);
                break;
            }
        }
    }
    return path.empty() ? nullptr : path.c_str();
}

// Lazy-init handle
inline void* get_umath_handle() {
    static void* handle = nullptr;
    if (!handle) {
        const char* path = find_umath_path();
        if (path) handle = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
    }
    return handle;
}

// Resolve a symbol from numpy's umath .so
inline void* resolve_npy(const char* name) {
    void* h = get_umath_handle();
    if (h) return dlsym(h, name);
    return nullptr;
}

// ============================================================================
// npy_* function wrappers — always use numpy's scalar math, never SVML
// ============================================================================

template<typename T> inline T exp(T x);
template<typename T> inline T log(T x);
template<typename T> inline T sqrt(T x);
template<typename T> inline T asin(T x);
template<typename T> inline T atan2(T y, T x);

template<>
inline double exp<double>(double x) {
    static auto fn = (double (*)(double))resolve_npy("npy_exp");
    if (fn) return fn(x);
    return std::exp(x);
}
template<>
inline float exp<float>(float x) {
    static auto fn = (float (*)(float))resolve_npy("npy_expf");
    if (fn) return fn(x);
    return std::exp(x);
}

template<>
inline double log<double>(double x) {
    static auto fn = (double (*)(double))resolve_npy("npy_log");
    if (fn) return fn(x);
    return std::log(x);
}
template<>
inline float log<float>(float x) {
    static auto fn = (float (*)(float))resolve_npy("npy_logf");
    if (fn) return fn(x);
    return std::log(x);
}

template<>
inline double sqrt<double>(double x) {
    static auto fn = (double (*)(double))resolve_npy("npy_sqrt");
    if (fn) return fn(x);
    return std::sqrt(x);
}
template<>
inline float sqrt<float>(float x) {
    static auto fn = (float (*)(float))resolve_npy("npy_sqrtf");
    if (fn) return fn(x);
    return std::sqrt(x);
}

template<>
inline double asin<double>(double x) {
    static auto fn = (double (*)(double))resolve_npy("npy_asin");
    if (fn) return fn(x);
    return std::asin(x);
}
template<>
inline float asin<float>(float x) {
    static auto fn = (float (*)(float))resolve_npy("npy_asinf");
    if (fn) return fn(x);
    return std::asin(x);
}

template<>
inline double atan2<double>(double y, double x) {
    static auto fn = (double (*)(double, double))resolve_npy("npy_atan2");
    if (fn) return fn(y, x);
    return std::atan2(y, x);
}
template<>
inline float atan2<float>(float y, float x) {
    static auto fn = (float (*)(float, float))resolve_npy("npy_atan2f");
    if (fn) return fn(y, x);
    return std::atan2(y, x);
}

}  // namespace npy_bridge
}  // namespace scipy
