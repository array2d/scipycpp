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
// KDTree — backed by scipy's own ckdtree C++ code (O(log n) true kd-tree)
// ============================================================================
// Uses ckdtree-dev package (extracted from scipy.spatial.ckdtree v1.8.0).

#include <ckdtree/ckdtree.hpp>

template<typename T>
struct KDTree {
    ckdtree_ns::ckdtree* tree;
    std::vector<double>  data_double;  // ckdtree works with double internally
    int dim;

    KDTree(const T* points, size_t n_pts, int dim_)
        : tree(nullptr), dim(dim_)
    {
        // Convert to double (ckdtree uses double)
        data_double.resize(n_pts * dim_);
        for (size_t i = 0; i < n_pts * dim_; ++i)
            data_double[i] = static_cast<double>(points[i]);
        tree = ckdtree_build(data_double.data(),
                             static_cast<intptr_t>(n_pts),
                             static_cast<intptr_t>(dim_),
                             16 /* leafsize */);
    }

    ~KDTree() { if (tree) ckdtree_free(tree); }

    // Non-copyable (owns ckdtree pointer)
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    KDTree(KDTree&& other) noexcept : tree(other.tree), data_double(std::move(other.data_double)), dim(other.dim)
    { other.tree = nullptr; }
    KDTree& operator=(KDTree&& other) noexcept {
        if (this != &other) {
            if (tree) ckdtree_free(tree);
            tree = other.tree; other.tree = nullptr;
            data_double = std::move(other.data_double);
            dim = other.dim;
        }
        return *this;
    }

    /// Query k=1 → (dist, idx)
    void query(const T* q, T& dist, size_t& idx) const {
        query_k(q, &dist, &idx, 1);
    }

    /// Query k nearest → outputs to dists_out and indices_out
    void query(const T* q, T* dists_out, size_t* indices_out, int k) const {
        query_k(q, dists_out, indices_out, k);
    }

private:
    void query_k(const T* q, T* dists_out, size_t* indices_out, int k) const {
        std::vector<double> q64(dim);
        for (int d = 0; d < dim; ++d) q64[d] = static_cast<double>(q[d]);

        std::vector<double>   dd(k);
        std::vector<intptr_t> ii(k);
        std::vector<intptr_t> kvals(k);
        for (int i = 0; i < k; ++i) kvals[i] = i + 1;  // ckdtree needs k=[1,2,...,k]
        intptr_t ik = static_cast<intptr_t>(k);
        ckdtree_query_knn(tree, dd.data(), ii.data(), q64.data(), 1,
                          kvals.data(), ik, ik, 0.0, 2.0, INFINITY);
        for (int i = 0; i < k; ++i) {
            dists_out[i]   = static_cast<T>(dd[i]);  // ckdtree already returns actual distances
            indices_out[i] = static_cast<size_t>(ii[i]);
        }
    }

public:
    /// Legacy helpers
    size_t query_index(const T* q) const {
        T d; size_t i; query(q, d, i); return i;
    }
    std::vector<size_t> query_k_indices(const T* q, int k) const {
        std::vector<size_t> result(k);
        std::vector<T> dists(k);
        query(q, dists.data(), result.data(), k);
        return result;
    }
};

}  // namespace spatial
}  // namespace scipy
