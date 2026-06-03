// ckdtree — Standalone C++ kd-tree extracted from scipy.spatial.ckdtree (v1.8.0)
//
// Original: scipy/scipy/spatial/ckdtree/src/
// License: BSD-3-Clause (same as scipy)
//
// Numpy dependencies removed — pure C++11, header-only.
//
// Usage:
//   #include "ckdtree/ckdtree.hpp"
//   ckdtree tree = ckdtree_build(data, n, m, leafsize);
//   ckdtree_query_knn(&tree, dists, indices, query_pts, n_queries, k, ...);
//   ckdtree_free(&tree);

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>
#include <cassert>

// ============================================================================
// numpy compatibility layer
// ============================================================================
#ifndef CKDTREE_LIKELY
  #define CKDTREE_LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef CKDTREE_UNLIKELY
  #define CKDTREE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#ifndef CKDTREE_PREFETCH
  #define CKDTREE_PREFETCH(x, rw, loc) __builtin_prefetch((x), (rw), 3)
#endif

namespace ckdtree_ns {

using ckdtree_intp_t = intptr_t;

inline bool ckdtree_isinf(double x) { return std::isinf(x); }
inline double ckdtree_fmin(double x, double y) { return std::fmin(x, y); }
inline double ckdtree_fmax(double x, double y) { return std::fmax(x, y); }
inline double ckdtree_fabs(double x) { return std::fabs(x); }

// ============================================================================
// ordered_pair.h
// ============================================================================
struct ordered_pair { ckdtree_intp_t i, j; };

inline void add_ordered_pair(std::vector<ordered_pair> *results,
                             const ckdtree_intp_t i, const intptr_t j) {
    if (i > j) { ordered_pair p = {j,i}; results->push_back(p); }
    else       { ordered_pair p = {i,j}; results->push_back(p); }
}

// ============================================================================
// coo_entries.h
// ============================================================================
struct coo_entry { ckdtree_intp_t i, j; double v; };

// ============================================================================
// distance_base.h — BaseMinkowskiDistPp + specializations P1, Pinf, P2
// ============================================================================
// Forward declaration needed by distance templates
struct ckdtree;
struct Rectangle;

template <typename Dist1D>
struct BaseMinkowskiDistPp {
    static inline void interval_interval_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, const double p, double *min, double *max) {
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        *min = std::pow(*min, p); *max = std::pow(*max, p);
    }
    static inline void rect_rect_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2, const double p,
        double *min, double *max) {
        *min = 0.; *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            *min += std::pow(min_, p); *max += std::pow(max_, p);
        }
    }
    static inline double point_point_p(const ckdtree * tree,
        const double *x, const double *y, const double p, const ckdtree_intp_t k,
        const double upperbound) {
        double r = 0;
        for (ckdtree_intp_t i=0; i<k; ++i) {
            double r1 = Dist1D::point_point(tree, x, y, i);
            r += std::pow(r1, p);
            if (r>upperbound) return r;
        }
        return r;
    }
    static inline double distance_p(const double s, const double p)
    { return std::pow(s,p); }
};

template <typename Dist1D>
struct BaseMinkowskiDistP1 : public BaseMinkowskiDistPp<Dist1D> {
    static inline void interval_interval_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, const double p, double *min, double *max) {
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
    }
    static inline void rect_rect_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2, const double p,
        double *min, double *max) {
        *min = 0.; *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            *min += min_; *max += max_;
        }
    }
    static inline double point_point_p(const ckdtree * tree,
        const double *x, const double *y, const double p, const ckdtree_intp_t k,
        const double upperbound) {
        double r = 0;
        for (ckdtree_intp_t i=0; i<k; ++i) {
            r += Dist1D::point_point(tree, x, y, i);
            if (r>upperbound) return r;
        }
        return r;
    }
    static inline double distance_p(const double s, const double p) { return s; }
};

template <typename Dist1D>
struct BaseMinkowskiDistPinf : public BaseMinkowskiDistPp<Dist1D> {
    static inline void interval_interval_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, double p, double *min, double *max)
    { return rect_rect_p(tree, rect1, rect2, p, min, max); }
    static inline void rect_rect_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2, const double p,
        double *min, double *max) {
        *min = 0.; *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            *min = ckdtree_fmax(*min, min_);
            *max = ckdtree_fmax(*max, max_);
        }
    }
    static inline double point_point_p(const ckdtree * tree,
        const double *x, const double *y, const double p, const ckdtree_intp_t k,
        const double upperbound) {
        double r = 0;
        for (ckdtree_intp_t i=0; i<k; ++i) {
            r = ckdtree_fmax(r, Dist1D::point_point(tree, x, y, i));
            if (r>upperbound) return r;
        }
        return r;
    }
    static inline double distance_p(const double s, const double p) { return s; }
};

template <typename Dist1D>
struct BaseMinkowskiDistP2 : public BaseMinkowskiDistPp<Dist1D> {
    static inline void interval_interval_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, const double p, double *min, double *max) {
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        *min *= *min; *max *= *max;
    }
    static inline void rect_rect_p(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2, const double p,
        double *min, double *max) {
        *min = 0.; *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;
            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            min_ *= min_; max_ *= max_;
            *min += min_; *max += max_;
        }
    }
    static inline double point_point_p(const ckdtree * tree,
        const double *x, const double *y, const double p, const ckdtree_intp_t k,
        const double upperbound) {
        double r = 0;
        for (ckdtree_intp_t i=0; i<k; ++i) {
            double r1 = Dist1D::point_point(tree, x, y, i);
            r += r1 * r1;
            if (r>upperbound) return r;
        }
        return r;
    }
    static inline double distance_p(const double s, const double p) { return s * s; }
};

// ============================================================================
// distance.h — PlainDist1D, MinkowskiDistP2 (optimized), BoxDist1D
// ============================================================================
struct PlainDist1D {
    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x, const double min,
        const double max, const ckdtree_intp_t k) {
        double s = 0, t = x - max;
        if (t > s) s = t;
        else { t = min - x; if (t > s) s = t; }
        return s;
    }
    static inline void interval_interval(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, double *min, double *max);
    static inline double point_point(const ckdtree * tree,
        const double *x, const double *y, const ckdtree_intp_t k)
    { return ckdtree_fabs(x[k] - y[k]); }
};

typedef BaseMinkowskiDistPp<PlainDist1D> MinkowskiDistPp;
typedef BaseMinkowskiDistPinf<PlainDist1D> MinkowskiDistPinf;
typedef BaseMinkowskiDistP1<PlainDist1D> MinkowskiDistP1;
typedef BaseMinkowskiDistP2<PlainDist1D> NonOptimizedMinkowskiDistP2;

inline double sqeuclidean_distance_double(const double *u, const double *v, ckdtree_intp_t n) {
    double s = 0;
    ckdtree_intp_t i = 0;
    double acc[4] = {0., 0., 0., 0.};
    for (; i + 4 <= n; i += 4) {
        double diff0 = u[i] - v[i];
        double diff1 = u[i+1] - v[i+1];
        double diff2 = u[i+2] - v[i+2];
        double diff3 = u[i+3] - v[i+3];
        acc[0] += diff0 * diff0;
        acc[1] += diff1 * diff1;
        acc[2] += diff2 * diff2;
        acc[3] += diff3 * diff3;
    }
    s = acc[0] + acc[1] + acc[2] + acc[3];
    for(; i<n; ++i) { double d = u[i] - v[i]; s += d * d; }
    return s;
}

struct MinkowskiDistP2: NonOptimizedMinkowskiDistP2 {
    static inline double point_point_p(const ckdtree * tree,
        const double *x, const double *y, const double p,
        const ckdtree_intp_t k, const double upperbound) {
        (void)p; (void)upperbound; (void)tree;
        return sqeuclidean_distance_double(x, y, k);
    }
};

struct BoxDist1D {
    static inline const double wrap_position(const double x, const double boxsize) {
        if (boxsize <= 0) return x;
        double x1 = x - std::floor(x / boxsize) * boxsize;
        while(x1 >= boxsize) x1 -= boxsize;
        while(x1 < 0) x1 += boxsize;
        return x1;
    }
    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x, const double min,
        const double max, const ckdtree_intp_t k);

    static inline void interval_interval(const ckdtree * tree,
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k, double *min, double *max);

    static inline double point_point(const ckdtree * tree,
        const double *x, const double *y, const ckdtree_intp_t k);

private:
    static inline double wrap_distance(const double x, const double hb, const double fb) {
        if (CKDTREE_UNLIKELY(x < -hb)) return fb + x;
        else if (CKDTREE_UNLIKELY(x > hb)) return x - fb;
        return x;
    }
};

typedef BaseMinkowskiDistPp<BoxDist1D> BoxMinkowskiDistPp;
typedef BaseMinkowskiDistPinf<BoxDist1D> BoxMinkowskiDistPinf;
typedef BaseMinkowskiDistP1<BoxDist1D> BoxMinkowskiDistP1;
typedef BaseMinkowskiDistP2<BoxDist1D> BoxMinkowskiDistP2;

// ============================================================================
// ckdtree_decl.h — core data structures
// ============================================================================
struct ckdtreenode {
    ckdtree_intp_t split_dim;
    ckdtree_intp_t children;
    double         split;
    ckdtree_intp_t start_idx;
    ckdtree_intp_t end_idx;
    ckdtreenode   *less;
    ckdtreenode   *greater;
    ckdtree_intp_t _less;
    ckdtree_intp_t _greater;
};

struct ckdtree {
    std::vector<ckdtreenode> *tree_buffer;
    ckdtreenode   *ctree;
    double         *raw_data;
    ckdtree_intp_t  n, m, leafsize, size;
    double         *raw_maxes, *raw_mins;
    ckdtree_intp_t *raw_indices;
    double         *raw_boxsize_data;
};

// ============================================================================
// rectangle.h
// ============================================================================
struct Rectangle {
    const ckdtree_intp_t m;
    double * const maxes() const { return &buf[0]; }
    double * const mins() const { return &buf[0] + m; }

    Rectangle(const ckdtree_intp_t _m, const double *_mins, const double *_maxes)
        : m(_m), buf(2 * _m) {
        std::memcpy((void*)mins(), (void*)_mins, m*sizeof(double));
        std::memcpy((void*)maxes(), (void*)_maxes, m*sizeof(double));
    }
    Rectangle(const Rectangle& rect) : m(rect.m), buf(rect.buf) {}
private:
    mutable std::vector<double> buf;
};

// distance.h — implementations that need Rectangle definition
inline void PlainDist1D::interval_interval(const ckdtree * tree,
    const Rectangle& rect1, const Rectangle& rect2,
    const ckdtree_intp_t k, double *min, double *max) {
    (void)tree;
    *min = ckdtree_fmax(0., std::fmax(rect1.mins()[k] - rect2.maxes()[k],
                                       rect2.mins()[k] - rect1.maxes()[k]));
    *max = ckdtree_fmax(rect1.maxes()[k] - rect2.mins()[k],
                         rect2.maxes()[k] - rect1.mins()[k]);
}

// BoxDist1D implementations
inline const double BoxDist1D::side_distance_from_min_max(
    const ckdtree * tree, const double x, const double min,
    const double max, const ckdtree_intp_t k) {
    double fb = tree->raw_boxsize_data[k];
    if (fb <= 0) return PlainDist1D::side_distance_from_min_max(tree, x, min, max, k);

    double tmax = x - max, tmin = x - min;
    if(CKDTREE_LIKELY(tmax < 0 && tmin > 0)) return 0;

    tmax = ckdtree_fabs(tmax); tmin = ckdtree_fabs(tmin);
    if(tmin > tmax) { double t = tmin; tmin = tmax; tmax = t; }

    double hb = tree->raw_boxsize_data[k + tree->m];
    if(tmax < hb) return tmin;
    if(tmin > hb) return fb - tmax;
    tmax = fb - tmax;
    return (tmin > tmax) ? tmax : tmin;
}

inline void BoxDist1D::interval_interval(const ckdtree * tree,
    const Rectangle& rect1, const Rectangle& rect2,
    const ckdtree_intp_t k, double *min, double *max) {
    double rmin = rect1.mins()[k] - rect2.maxes()[k];
    double rmax = rect1.maxes()[k] - rect2.mins()[k];
    double fb = tree->raw_boxsize_data[k];
    double hb = tree->raw_boxsize_data[k + rect1.m];

    if (CKDTREE_UNLIKELY(fb <= 0)) {
        if(rmax <= 0 || rmin >= 0) {
            rmin = ckdtree_fabs(rmin); rmax = ckdtree_fabs(rmax);
            if(rmin < rmax) { *min = rmin; *max = rmax; }
            else { *min = rmax; *max = rmin; }
        } else {
            rmin = ckdtree_fabs(rmin); rmax = ckdtree_fabs(rmax);
            *max = ckdtree_fmax(rmax, rmin); *min = 0;
        }
        return;
    }
    if(rmax <= 0 || rmin >= 0) {
        rmin = ckdtree_fabs(rmin); rmax = ckdtree_fabs(rmax);
        if(rmin > rmax) { double t = rmin; rmin = rmax; rmax = t; }
        if(rmax < hb) { *min = rmin; *max = rmax; }
        else if(rmin > hb) { *max = fb - rmin; *min = fb - rmax; }
        else { *max = hb; *min = ckdtree_fmin(rmin, fb - rmax); }
    } else {
        rmin = -rmin;
        if(rmin > rmax) rmax = rmin;
        if(rmax > hb) rmax = hb;
        *max = rmax; *min = 0;
    }
}

inline double BoxDist1D::point_point(const ckdtree * tree,
    const double *x, const double *y, const ckdtree_intp_t k) {
    double r1 = wrap_distance(x[k] - y[k],
        tree->raw_boxsize_data[k + tree->m], tree->raw_boxsize_data[k]);
    return ckdtree_fabs(r1);
}

// ============================================================================
// build.cxx — kd-tree construction
// ============================================================================
#define tree_buffer_root(buf) (&(buf)[0][0])

static ckdtree_intp_t build_impl(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
    double *maxes, double *mins, const int _median, const int _compact) {

    const ckdtree_intp_t m = self->m;
    const double *data = self->raw_data;
    ckdtree_intp_t *indices = (ckdtree_intp_t *)(self->raw_indices);

    ckdtreenode new_node, *n, *root;
    ckdtree_intp_t node_index, _less, _greater;
    ckdtree_intp_t i, j, p, d;
    double size, split, minval, maxval;

    self->tree_buffer->push_back(new_node);
    node_index = self->tree_buffer->size() - 1;
    root = tree_buffer_root(self->tree_buffer);
    n = root + node_index;
    memset(n, 0, sizeof(n[0]));
    n->start_idx = start_idx;
    n->end_idx = end_idx;
    n->children = end_idx - start_idx;

    if (end_idx-start_idx <= self->leafsize) {
        n->split_dim = -1;
        return node_index;
    }

    if (CKDTREE_LIKELY(_compact)) {
        const double *tmp_data_point = data + indices[start_idx] * m;
        for(i=0; i<m; ++i) { maxes[i] = tmp_data_point[i]; mins[i] = tmp_data_point[i]; }
        for (j = start_idx + 1; j < end_idx; ++j) {
            tmp_data_point = data + indices[j] * m;
            for(i=0; i<m; ++i) {
                double tmp = tmp_data_point[i];
                maxes[i] = maxes[i] > tmp ? maxes[i] : tmp;
                mins[i] = mins[i] < tmp ? mins[i] : tmp;
            }
        }
    }

    d = 0; size = 0;
    for (i=0; i<m; ++i) {
        if (maxes[i] - mins[i] > size) { d = i; size = maxes[i] - mins[i]; }
    }
    maxval = maxes[d]; minval = mins[d];
    if (maxval == minval) {
        n->split_dim = -1;
        return node_index;
    }

    auto index_compare = [=](ckdtree_intp_t a, ckdtree_intp_t b) {
        return data[a * m + d] < data[b * m + d];
    };

    auto partition_pivot = [=](ckdtree_intp_t* first, ckdtree_intp_t* last, double pivot) {
        auto partition_ptr = std::partition(
            first, last, [&](ckdtree_intp_t a) { return data[a * m + d] < pivot; });
        return partition_ptr - indices;
    };

    if (CKDTREE_LIKELY(_median)) {
        const auto n_points = end_idx - start_idx;
        auto* node_indices = indices + start_idx;
        auto mid = node_indices + n_points / 2;
        std::nth_element(node_indices, mid, node_indices + n_points, index_compare);
        split = data[*mid * m + d];
        p = partition_pivot(node_indices, mid, split);
    } else {
        split = (maxval + minval) / 2;
        p = partition_pivot(indices + start_idx, indices + end_idx, split);
    }

    if (p == start_idx) {
        auto min_idx = *std::min_element(indices + start_idx, indices + end_idx, index_compare);
        split = std::nextafter(data[min_idx * m + d], HUGE_VAL);
        p = partition_pivot(indices + start_idx, indices + end_idx, split);
    } else if (p == end_idx) {
        auto max_idx = *std::max_element(indices + start_idx, indices + end_idx, index_compare);
        split = data[max_idx * m + d];
        p = partition_pivot(indices + start_idx, indices + end_idx, split);
    }

    if (CKDTREE_UNLIKELY(p == start_idx || p == end_idx)) {
        assert(!_compact);
        self->tree_buffer->pop_back();
        std::vector<double> tmp_bounds(2 * m);
        double* tmp_mins = &tmp_bounds[0];
        std::copy_n(mins, m, tmp_mins);
        double* tmp_maxes = &tmp_bounds[m];
        std::copy_n(maxes, m, tmp_maxes);
        const auto fixed_val = data[indices[start_idx]*m + d];
        tmp_mins[d] = fixed_val; tmp_maxes[d] = fixed_val;
        return build_impl(self, start_idx, end_idx, tmp_maxes, tmp_mins, _median, _compact);
    }

    if (CKDTREE_LIKELY(_compact)) {
        _less = build_impl(self, start_idx, p, maxes, mins, _median, _compact);
        _greater = build_impl(self, p, end_idx, maxes, mins, _median, _compact);
    } else {
        std::vector<double> tmp(m);
        double *mids = &tmp[0];
        for (i=0; i<m; ++i) mids[i] = maxes[i];
        mids[d] = split;
        _less = build_impl(self, start_idx, p, mids, mins, _median, _compact);
        for (i=0; i<m; ++i) mids[i] = mins[i];
        mids[d] = split;
        _greater = build_impl(self, p, end_idx, maxes, mids, _median, _compact);
    }

    root = tree_buffer_root(self->tree_buffer);
    n = root + node_index;
    n->_less = _less; n->_greater = _greater;
    n->less = root + _less; n->greater = root + _greater;
    n->split_dim = d; n->split = split;
    return node_index;
}

inline int build_ckdtree(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
                          double *maxes, double *mins, int _median, int _compact) {
    build_impl(self, start_idx, end_idx, maxes, mins, _median, _compact);
    return 0;
}

// ============================================================================
// query.cxx — k-nearest neighbor search
// ============================================================================
union heapcontents { ckdtree_intp_t intdata; void *ptrdata; };

struct heapitem { double priority; heapcontents contents; };

struct heap {
    std::vector<heapitem> _heap;
    ckdtree_intp_t n, space;
    heap (ckdtree_intp_t initial_size) : _heap(initial_size), n(0), space(initial_size) {}

    inline void push(heapitem &item) {
        n++;
        if (n > space) _heap.resize(2*space+1);
        space = _heap.size();
        ckdtree_intp_t i = n-1;
        _heap[i] = item;
        while ((i > 0) && (_heap[i].priority < _heap[(i-1)/2].priority)) {
            heapitem t = _heap[(i-1)/2];
            _heap[(i-1)/2] = _heap[i]; _heap[i] = t;
            i = (i-1)/2;
        }
    }
    inline heapitem peek() { return _heap[0]; }

    inline void remove() {
        ckdtree_intp_t i, j, k, l, nn;
        _heap[0] = _heap[n-1]; n--; nn = n;
        i=0; j=1; k=2;
        while (((j<nn) && (_heap[i].priority > _heap[j].priority)) ||
               ((k<nn) && (_heap[i].priority > _heap[k].priority))) {
            if ((k<nn) && (_heap[j].priority >_heap[k].priority)) l = k;
            else l = j;
            heapitem t = _heap[l]; _heap[l] = _heap[i]; _heap[i] = t;
            i = l; j = 2*i+1; k = 2*i+2;
        }
    }
    inline heapitem pop() { heapitem it = _heap[0]; remove(); return it; }
};

struct nodeinfo {
    const ckdtreenode *node;
    ckdtree_intp_t m;
    double min_distance;
    double buf[1];
    inline double * const side_distances() { return buf; }
    inline double * const maxes() { return buf + m; }
    inline double * const mins() { return buf + 2 * m; }
    inline void init_box(const struct nodeinfo * from) {
        std::memcpy(buf, from->buf, sizeof(double) * (3 * m));
        min_distance = from->min_distance;
    }
    inline void init_plain(const struct nodeinfo * from) {
        std::memcpy(buf, from->buf, sizeof(double) * m);
        min_distance = from->min_distance;
    }
    inline void update_side_distance(const int d, const double new_side_distance, const double p) {
        if (CKDTREE_UNLIKELY(ckdtree_isinf(p))) {
            min_distance = ckdtree_fmax(min_distance, new_side_distance);
        } else {
            min_distance += new_side_distance - side_distances()[d];
        }
        side_distances()[d] = new_side_distance;
    }
};

struct nodeinfo_pool {
    std::vector<char*> pool;
    ckdtree_intp_t alloc_size, arena_size, m;
    char *arena, *arena_ptr;

    nodeinfo_pool(ckdtree_intp_t _m) : m(_m) {
        alloc_size = sizeof(nodeinfo) + (3 * m -1)*sizeof(double);
        alloc_size = 64*(alloc_size/64)+64;
        arena_size = 4096*((64*alloc_size)/4096)+4096;
        arena = new char[arena_size];
        arena_ptr = arena;
        pool.push_back(arena);
    }
    ~nodeinfo_pool() {
        for (ckdtree_intp_t i = pool.size()-1; i >= 0; --i) delete [] pool[i];
    }
    inline nodeinfo *allocate() {
        ckdtree_intp_t m1 = (ckdtree_intp_t)arena_ptr;
        ckdtree_intp_t m0 = (ckdtree_intp_t)arena;
        if ((arena_size-(ckdtree_intp_t)(m1-m0))<alloc_size) {
            arena = new char[arena_size];
            arena_ptr = arena;
            pool.push_back(arena);
        }
        nodeinfo *ni1 = (nodeinfo*)arena_ptr;
        ni1->m = m;
        arena_ptr += alloc_size;
        return ni1;
    }
};

template <typename MinMaxDist>
static void query_single_point(const ckdtree *self,
    double *result_distances, ckdtree_intp_t *result_indices,
    const double *x, const ckdtree_intp_t *k, const ckdtree_intp_t nk,
    const ckdtree_intp_t kmax, const double eps, const double p,
    double distance_upper_bound) {

    static double inf = strtod("INF", NULL);
    nodeinfo_pool nipool(self->m);
    heap q(12), neighbors(kmax);

    const ckdtree_intp_t m = self->m;
    nodeinfo *ni1, *ni2;
    double d, epsfac;
    heapitem it, it2, neighbor;
    const ckdtreenode *node, *inode;

    ni1 = nipool.allocate();
    ni1->node = self->ctree;
    ni1->min_distance = 0;

    for (ckdtree_intp_t i=0; i<m; ++i) {
        ni1->mins()[i] = self->raw_mins[i];
        ni1->maxes()[i] = self->raw_maxes[i];
        double side_distance;
        if(self->raw_boxsize_data != NULL) {
            side_distance = BoxDist1D::side_distance_from_min_max(
                self, x[i], self->raw_mins[i], self->raw_maxes[i], i);
        } else {
            side_distance = PlainDist1D::side_distance_from_min_max(
                self, x[i], self->raw_mins[i], self->raw_maxes[i], i);
        }
        side_distance = MinMaxDist::distance_p(side_distance, p);
        ni1->side_distances()[i] = 0;
        ni1->update_side_distance(i, side_distance, p);
    }

    if (CKDTREE_LIKELY(p == 2.0)) {
        double tmp = 1. + eps; epsfac = 1. / (tmp*tmp);
    } else if (eps == 0.) epsfac = 1.;
    else if (ckdtree_isinf(p)) epsfac = 1. / (1. + eps);
    else epsfac = 1. / std::pow((1. + eps), p);

    if (CKDTREE_LIKELY(p == 2.0)) {
        distance_upper_bound = distance_upper_bound * distance_upper_bound;
    } else if ((!ckdtree_isinf(p)) && !std::isinf(distance_upper_bound))
        distance_upper_bound = std::pow(distance_upper_bound,p);

    for(;;) {
        if (ni1->node->split_dim == -1) {
            node = ni1->node;
            const ckdtree_intp_t start_idx = node->start_idx;
            const ckdtree_intp_t end_idx = node->end_idx;
            const double *data = self->raw_data;
            const ckdtree_intp_t *indices = self->raw_indices;

            CKDTREE_PREFETCH(data+indices[start_idx]*m, 0, m);
            if (start_idx < end_idx - 1)
                CKDTREE_PREFETCH(data+indices[start_idx+1]*m, 0, m);

            for (ckdtree_intp_t i=start_idx; i<end_idx; ++i) {
                if (i < end_idx - 2)
                    CKDTREE_PREFETCH(data+indices[i+2]*m, 0, m);

                d = MinMaxDist::point_point_p(self, data+indices[i]*m, x, p, m, distance_upper_bound);
                if (d < distance_upper_bound) {
                    if (neighbors.n == kmax) neighbors.remove();
                    neighbor.priority = -d;
                    neighbor.contents.intdata = indices[i];
                    neighbors.push(neighbor);
                    if (neighbors.n == kmax)
                        distance_upper_bound = -neighbors.peek().priority;
                }
            }
            if (q.n == 0) break;
            else { it = q.pop(); ni1 = (nodeinfo*)(it.contents.ptrdata); }
        } else {
            inode = ni1->node;
            const ckdtree_intp_t split_dim = inode->split_dim;
            const double split = inode->split;

            if (ni1->min_distance > distance_upper_bound*epsfac) break;

            ni2 = nipool.allocate();

            if (CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {
                ni2->init_plain(ni1);
                double side_distance;
                if (x[split_dim] < split) {
                    ni1->node = inode->less; ni2->node = inode->greater;
                    side_distance = split - x[split_dim];
                } else {
                    ni1->node = inode->greater; ni2->node = inode->less;
                    side_distance = x[split_dim] - split;
                }
                side_distance = MinMaxDist::distance_p(side_distance, p);
                ni2->update_side_distance(split_dim, side_distance, p);
            } else {
                ni2->init_box(ni1);
                ni1->maxes()[split_dim] = split;
                ni1->node = inode->less;
                double side_distance = BoxDist1D::side_distance_from_min_max(
                    self, x[split_dim], ni1->mins()[split_dim],
                    ni1->maxes()[split_dim], split_dim);
                side_distance = MinMaxDist::distance_p(side_distance, p);
                ni1->update_side_distance(split_dim, side_distance, p);

                ni2->mins()[split_dim] = split;
                ni2->node = inode->greater;
                side_distance = BoxDist1D::side_distance_from_min_max(
                    self, x[split_dim], ni2->mins()[split_dim],
                    ni2->maxes()[split_dim], split_dim);
                side_distance = MinMaxDist::distance_p(side_distance, p);
                ni2->update_side_distance(split_dim, side_distance, p);
            }

            if (ni1->min_distance > ni2->min_distance) {
                nodeinfo *tmp = ni1; ni1 = ni2; ni2 = tmp;
            }
            if (ni2->min_distance<=distance_upper_bound*epsfac) {
                it2.priority = ni2->min_distance;
                it2.contents.ptrdata = (void*) ni2;
                q.push(it2);
            }
        }
    }

    std::vector<heapitem> sorted_neighbors(kmax);
    ckdtree_intp_t nnb = neighbors.n;
    for(ckdtree_intp_t i = neighbors.n - 1; i >=0; --i)
        sorted_neighbors[i] = neighbors.pop();

    for (ckdtree_intp_t i = 0; i < nk; ++i) {
        if(CKDTREE_UNLIKELY(k[i] - 1 >= nnb)) {
            result_indices[i] = self->n;
            result_distances[i] = inf;
        } else {
            neighbor = sorted_neighbors[k[i] - 1];
            result_indices[i] = neighbor.contents.intdata;
            if (CKDTREE_LIKELY(p == 2.0))
                result_distances[i] = std::sqrt(-neighbor.priority);
            else if ((p == 1.) || (ckdtree_isinf(p)))
                result_distances[i] = -neighbor.priority;
            else
                result_distances[i] = std::pow((-neighbor.priority),(1./p));
        }
    }
}

#define HANDLE(cond, kls) \
    if(cond) { query_single_point<kls>(self, dd_row, ii_row, xx_row, k, nk, kmax, eps, p, distance_upper_bound); } else

inline int query_knn(const ckdtree *self,
    double *dd, ckdtree_intp_t *ii,
    const double *xx, const ckdtree_intp_t n,
    const ckdtree_intp_t* k, const ckdtree_intp_t nk,
    const ckdtree_intp_t kmax, const double eps,
    const double p, const double distance_upper_bound) {

    ckdtree_intp_t m = self->m;

    if(CKDTREE_LIKELY(!self->raw_boxsize_data)) {
        for (ckdtree_intp_t i=0; i<n; ++i) {
            double *dd_row = dd + (i*nk);
            ckdtree_intp_t *ii_row = ii + (i*nk);
            const double *xx_row = xx + (i*m);
            HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)
            HANDLE(p == 1, MinkowskiDistP1)
            HANDLE(ckdtree_isinf(p), MinkowskiDistPinf)
            HANDLE(1, MinkowskiDistPp) {}
        }
    } else {
        std::vector<double> row(m);
        double *xx_row = &row[0];
        for (ckdtree_intp_t i=0; i<n; ++i) {
            double *dd_row = dd + (i*nk);
            ckdtree_intp_t *ii_row = ii + (i*nk);
            const double *old_xx_row = xx + (i*m);
            for(int64_t j=0; j<m; ++j)
                xx_row[j] = BoxDist1D::wrap_position(old_xx_row[j], self->raw_boxsize_data[j]);
            HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)
            HANDLE(p == 1, BoxMinkowskiDistP1)
            HANDLE(ckdtree_isinf(p), BoxMinkowskiDistPinf)
            HANDLE(1, BoxMinkowskiDistPp) {}
        }
    }
    return 0;
}

#undef HANDLE

} // namespace ckdtree_ns

// ============================================================================
// Public C API (stdlib-style, no namespace)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/// Allocate and build a ckdtree.
/// data: n × m row-major double array (flattened).
/// leafsize: typical value 16.
/// Returns pointer to heap-allocated tree. Caller must free with ckdtree_free.
inline ckdtree_ns::ckdtree* ckdtree_build(const double *data, intptr_t n, intptr_t m, intptr_t leafsize) {
    using namespace ckdtree_ns;
    ckdtree *self = new ckdtree();
    self->n = n; self->m = m; self->leafsize = leafsize;
    self->raw_data = const_cast<double*>(data);
    self->raw_boxsize_data = NULL;

    self->raw_indices = new ckdtree_intp_t[n];
    for (intptr_t i = 0; i < n; ++i) self->raw_indices[i] = i;

    self->raw_maxes = new double[m];
    self->raw_mins  = new double[m];
    for (intptr_t i = 0; i < m; ++i) {
        self->raw_maxes[i] = data[i];
        self->raw_mins[i]  = data[i];
    }
    for (intptr_t j = 1; j < n; ++j) {
        const double *row = data + j * m;
        for (intptr_t i = 0; i < m; ++i) {
            if (row[i] > self->raw_maxes[i]) self->raw_maxes[i] = row[i];
            if (row[i] < self->raw_mins[i])  self->raw_mins[i]  = row[i];
        }
    }

    self->tree_buffer = new std::vector<ckdtreenode>();
    self->size = 0;
    build_ckdtree(self, 0, n, self->raw_maxes, self->raw_mins, 1, 1);
    self->ctree = tree_buffer_root(self->tree_buffer);
    self->size = self->tree_buffer->size();
    return self;
}

/// Query k-nearest neighbors for n_queries points.
/// dd: output distances, size n_queries × kmax
/// ii: output indices,  size n_queries × kmax
/// k:  array of k values per query, size n_queries (or single int for all)
inline int ckdtree_query_knn(const ckdtree_ns::ckdtree *self,
    double *dd, intptr_t *ii, const double *xx, intptr_t n_queries,
    const intptr_t *k, intptr_t nk, intptr_t kmax,
    double eps, double p, double distance_upper_bound) {
    return ckdtree_ns::query_knn(self, dd, ii, xx, n_queries, k, nk, kmax, eps, p, distance_upper_bound);
}

/// Free a ckdtree.
inline void ckdtree_free(ckdtree_ns::ckdtree *self) {
    if (!self) return;
    delete self->tree_buffer;
    delete[] self->raw_indices;
    delete[] self->raw_maxes;
    delete[] self->raw_mins;
    delete self;
}

#ifdef __cplusplus
}
#endif
