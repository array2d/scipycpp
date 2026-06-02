// scipy.spatial — distance metrics, KDTree, cdist.
//
// Usage: #include "scipy/spatial.h"

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <limits>

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

// ============================================================================
// scipy.spatial.distance.cdist — cross-set distance matrix
// ============================================================================
// Aligns with Python: scipy.spatial.distance.cdist(XA, XB, metric='euclidean')
// XA is mA × dim (row-major), XB is mB × dim (row-major)
// dst is mA × mB (row-major), dst[i*mB + j] = distance(XA[i], XB[j])

template<typename T>
inline void cdist_euclidean(const T* XA, size_t mA,
                             const T* XB, size_t mB,
                             size_t dim, T* dst) {
    for (size_t i = 0; i < mA; ++i) {
        const T* ai = XA + i * dim;
        for (size_t j = 0; j < mB; ++j) {
            const T* bj = XB + j * dim;
            T sum = T(0);
            for (size_t d = 0; d < dim; ++d) {
                T diff = ai[d] - bj[d];
                sum += diff * diff;
            }
            dst[i * mB + j] = std::sqrt(sum);
        }
    }
}

template<typename T>
inline void cdist_sqeuclidean(const T* XA, size_t mA,
                               const T* XB, size_t mB,
                               size_t dim, T* dst) {
    for (size_t i = 0; i < mA; ++i) {
        const T* ai = XA + i * dim;
        for (size_t j = 0; j < mB; ++j) {
            const T* bj = XB + j * dim;
            T sum = T(0);
            for (size_t d = 0; d < dim; ++d) {
                T diff = ai[d] - bj[d];
                sum += diff * diff;
            }
            dst[i * mB + j] = sum;
        }
    }
}

template<typename T>
inline void cdist_manhattan(const T* XA, size_t mA,
                             const T* XB, size_t mB,
                             size_t dim, T* dst) {
    for (size_t i = 0; i < mA; ++i) {
        const T* ai = XA + i * dim;
        for (size_t j = 0; j < mB; ++j) {
            const T* bj = XB + j * dim;
            T sum = T(0);
            for (size_t d = 0; d < dim; ++d)
                sum += std::abs(ai[d] - bj[d]);
            dst[i * mB + j] = sum;
        }
    }
}

template<typename T>
inline void cdist_chebyshev(const T* XA, size_t mA,
                             const T* XB, size_t mB,
                             size_t dim, T* dst) {
    for (size_t i = 0; i < mA; ++i) {
        const T* ai = XA + i * dim;
        for (size_t j = 0; j < mB; ++j) {
            const T* bj = XB + j * dim;
            T mx = T(0);
            for (size_t d = 0; d < dim; ++d) {
                T diff = std::abs(ai[d] - bj[d]);
                if (diff > mx) mx = diff;
            }
            dst[i * mB + j] = mx;
        }
    }
}

/// Generic cdist dispatcher
template<typename T>
inline void cdist(const T* XA, size_t mA,
                   const T* XB, size_t mB,
                   size_t dim, T* dst,
                   const char* metric = "euclidean") {
    std::string m(metric);
    if (m == "euclidean")
        cdist_euclidean(XA, mA, XB, mB, dim, dst);
    else if (m == "sqeuclidean")
        cdist_sqeuclidean(XA, mA, XB, mB, dim, dst);
    else if (m == "cityblock" || m == "manhattan")
        cdist_manhattan(XA, mA, XB, mB, dim, dst);
    else if (m == "chebyshev")
        cdist_chebyshev(XA, mA, XB, mB, dim, dst);
    else
        cdist_euclidean(XA, mA, XB, mB, dim, dst);  // default
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

    /// Python: tree.query(q, k=1) → returns (distances, indices) tuple
    /// C++:   tree.query(q, dist, idx) — sets dist and idx to nearest values.
    ///        tree.query(q, dists_out, indices_out, k) — for k-nearest.
    void query(const T* q, T& dist, size_t& idx) const {
        if (n == 0) { dist = T(0); idx = 0; return; }
        idx = 0;
        dist = distance::euclidean(q, pts, static_cast<size_t>(dim));
        for (size_t i = 1; i < n; ++i) {
            T d = distance::euclidean(q, pts + i * dim, static_cast<size_t>(dim));
            if (d < dist) { dist = d; idx = i; }
        }
    }

    /// Python: tree.query(q, k=k) → (distances, indices)
    /// C++:   tree.query(q, dists_out, indices_out, k)
    void query(const T* q, T* dists_out, size_t* indices_out, int k) const {
        std::vector<std::pair<T, size_t>> pairs; pairs.reserve(n);
        for (size_t i = 0; i < n; ++i)
            pairs.push_back({distance::euclidean(q, pts + i * dim, static_cast<size_t>(dim)), i});
        int m = std::min(k, (int)n);
        std::partial_sort(pairs.begin(), pairs.begin() + m, pairs.end());
        for (int i = 0; i < m; ++i) {
            dists_out[i]   = pairs[i].first;
            indices_out[i] = pairs[i].second;
        }
    }

    /// Legacy: index-only query (backward compatible)
    size_t query_index(const T* q) const {
        T d; size_t i; query(q, d, i); return i;
    }

    /// Legacy: indices-only k-nearest (backward compatible)
    std::vector<size_t> query_k_indices(const T* q, int k) const {
        std::vector<size_t> result(std::min(k, (int)n));
        std::vector<T> dists(std::min(k, (int)n));
        query(q, dists.data(), result.data(), k);
        return result;
    }
};

}  // namespace spatial
}  // namespace scipy
