// scipy.spatial — distance metrics, KDTree.
//
// Usage: #include "scipy/spatial.h"

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstddef>

namespace scipy {
namespace spatial {
namespace distance {

template<typename T>
inline T euclidean(const T* u, const T* v, size_t n) {
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) { T d = u[i] - v[i]; sum += d * d; }
    return std::sqrt(sum);
}

template<typename T>
inline T sqeuclidean(const T* u, const T* v, size_t n) {
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) { T d = u[i] - v[i]; sum += d * d; }
    return sum;
}

template<typename T>
inline T manhattan(const T* u, const T* v, size_t n) {
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += std::abs(u[i] - v[i]);
    return sum;
}

template<typename T>
inline T chebyshev(const T* u, const T* v, size_t n) {
    T m = T(0);
    for (size_t i = 0; i < n; ++i) { T d = std::abs(u[i] - v[i]); if (d > m) m = d; }
    return m;
}

template<typename T>
inline T cosine(const T* u, const T* v, size_t n) {
    T dot = T(0), nu = T(0), nv = T(0);
    for (size_t i = 0; i < n; ++i) { dot += u[i]*v[i]; nu += u[i]*u[i]; nv += v[i]*v[i]; }
    if (nu == T(0) || nv == T(0)) return T(0);
    return T(1) - dot / (std::sqrt(nu) * std::sqrt(nv));
}

template<typename T>
inline T correlation(const T* u, const T* v, size_t n) {
    T mu = T(0), mv = T(0);
    for (size_t i = 0; i < n; ++i) { mu += u[i]; mv += v[i]; }
    mu /= T(n); mv /= T(n);
    T dot = T(0), nu = T(0), nv = T(0);
    for (size_t i = 0; i < n; ++i) {
        T du = u[i] - mu, dv = v[i] - mv;
        dot += du*dv; nu += du*du; nv += dv*dv;
    }
    if (nu == T(0) || nv == T(0)) return T(0);
    return T(1) - dot / (std::sqrt(nu) * std::sqrt(nv));
}

template<typename T>
inline T minkowski(const T* u, const T* v, size_t n, T p = T(2)) {
    if (p == T(1)) return manhattan(u, v, n);
    if (std::isinf(p)) return chebyshev(u, v, n);
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += std::pow(std::abs(u[i] - v[i]), p);
    return std::pow(sum, T(1) / p);
}

}  // namespace distance

// ============================================================================
// Simple KDTree (brute-force, small datasets; for large data use nanoflann)
// ============================================================================

template<typename T>
struct KDTree {
    const T* pts; size_t n; int dim;

    KDTree(const T* points, size_t n_pts, int dim_)
        : pts(points), n(n_pts), dim(dim_) {}

    size_t query(const T* q) const {
        if (n == 0) return 0;
        size_t best = 0;
        T best_dist = distance::sqeuclidean(q, pts, static_cast<size_t>(dim));
        for (size_t i = 1; i < n; ++i) {
            T d = distance::sqeuclidean(q, pts + i * dim, static_cast<size_t>(dim));
            if (d < best_dist) { best_dist = d; best = i; }
        }
        return best;
    }

    std::vector<size_t> query_k(const T* q, int k) const {
        std::vector<std::pair<T, size_t>> dists; dists.reserve(n);
        for (size_t i = 0; i < n; ++i)
            dists.push_back({distance::sqeuclidean(q, pts + i * dim, static_cast<size_t>(dim)), i});
        std::partial_sort(dists.begin(), dists.begin() + std::min(k, (int)n), dists.end());
        std::vector<size_t> result;
        for (int i = 0; i < k && i < (int)n; ++i) result.push_back(dists[i].second);
        return result;
    }
};

}  // namespace spatial
}  // namespace scipy
