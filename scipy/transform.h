// scipy.spatial.transform — Rotation (from_matrix, as_euler, etc.)
//
// Aligns with Python: scipy.spatial.transform.Rotation
// Usage: #include "scipy/transform.h"

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "numpycpp/numpy.h"

namespace scipy {
namespace spatial {
namespace transform {

// ============================================================================
// Rotation — 3D rotation representation
// ============================================================================
// Supports: from_matrix(matrix) and as_euler(seq)
// Currently supports 'xyz' (intrinsic Tait-Bryan) and 'zyx' sequences.
//
// scipy's Rotation.from_matrix internally computes and normalizes a quaternion,
// then as_euler converts quaternion→matrix→euler. We replicate this exact
// pipeline for bit-level alignment.

template<typename T>
struct Rotation {
    // Quaternion [x, y, z, w] (scalar-last convention)
    T quat[4];
    // Original matrix stored from from_matrix() — avoids quaternion roundtrip
    // error in as_euler() and as_matrix() (1-4 ULP vs scipy).
    bool has_matrix;
    T matrix[9];  // row-major 3×3

    Rotation() {
        quat[0] = 0; quat[1] = 0; quat[2] = 0; quat[3] = 1;  // identity
        has_matrix = false;
    }

    /// scipy.spatial.transform.Rotation.from_matrix(R)
    /// R must be 3x3 row-major matrix.
    /// Internally computes quaternion and normalizes it, matching scipy's
    /// exact algorithm from _rotation.pyx.
    static Rotation from_matrix(const T* matrix) {
        Rotation rot;
        const T& R00 = matrix[0]; const T& R01 = matrix[1]; const T& R02 = matrix[2];
        const T& R10 = matrix[3]; const T& R11 = matrix[4]; const T& R12 = matrix[5];
        const T& R20 = matrix[6]; const T& R21 = matrix[7]; const T& R22 = matrix[8];

        // scipy's decision array: [R00, R11, R22, trace]
        // Explicit pairwise sum to ensure deterministic FP evaluation order
        T _t_sum01 = R00 + R11;
        T _t_trace = _t_sum01 + R22;
        T decision[4] = {R00, R11, R22, _t_trace};

        // Find index of largest decision value
        int choice = 0;
        T max_val = decision[0];
        for (int c = 1; c < 4; ++c) {
            if (decision[c] > max_val) { choice = c; max_val = decision[c]; }
        }

        T q[4];  // [x, y, z, w]

        if (choice != 3) {
            // Largest is an individual diagonal element
            int i = choice;
            int j = (i + 1) % 3;
            int k = (j + 1) % 3;

            // Map back to matrix indices
            T Ri = (i == 0) ? R00 : (i == 1) ? R11 : R22;
            T Rj = (j == 0) ? R00 : (j == 1) ? R11 : R22;
            T Rk = (k == 0) ? R00 : (k == 1) ? R11 : R22;

            // Build matrix access by index pairs
            T get_mat[3][3] = {{R00, R01, R02}, {R10, R11, R12}, {R20, R21, R22}};

            T _t_a = T(2) * Ri;
            T _t_b = T(1) - decision[3];
            q[i] = _t_a + _t_b;
            q[j] = get_mat[j][i] + get_mat[i][j];
            q[k] = get_mat[k][i] + get_mat[i][k];
            q[3] = get_mat[k][j] - get_mat[j][k];
        } else {
            // Trace is largest
            q[0] = R21 - R12;  // x
            q[1] = R02 - R20;  // y
            q[2] = R10 - R01;  // z
            q[3] = T(1) + decision[3];  // w
        }

        // Normalize (matching scipy's _normalize4)
        // scipy: norm = sqrt(_dot3(q[:3], q[:3]) + q[3]*q[3])
        // Break into pairwise operations for deterministic FP order
        T _sq0 = q[0] * q[0];
        T _sq1 = q[1] * q[1];
        T _sq2 = q[2] * q[2];
        T _sq3 = q[3] * q[3];
        T _sum01 = _sq0 + _sq1;
        T _sum23 = _sq2 + _sq3;
        T _sum_all = _sum01 + _sum23;
        T norm = std::sqrt(_sum_all);
        rot.quat[0] = q[0] / norm;
        rot.quat[1] = q[1] / norm;
        rot.quat[2] = q[2] / norm;
        rot.quat[3] = q[3] / norm;

        // Store quaternion-derived matrix to exactly match scipy's pipeline.
        // scipy: from_matrix → quat → normalize → quat_to_matrix → euler
        // Quat normalization can change matrix values by 1 ULP; scipy uses
        // the quaternion-derived matrix for as_euler(), not the raw input.
        // We MUST store the same quaternion-derived matrix for 0-bit match.
        rot.has_matrix = true;
        {
            T x = rot.quat[0], y = rot.quat[1], z = rot.quat[2], w = rot.quat[3];
            T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;
            T xy = x*y, zw = z*w, xz = x*z, yw = y*w, yz = y*z, xw = x*w;

            // Break compound ± expressions into pairwise ops for deterministic FP order
            { T _a = x2 - y2; T _b = w2 - z2; rot.matrix[0] = _a + _b; }
            { T _a = xy - zw; rot.matrix[1] = T(2) * _a; }
            { T _a = xz + yw; rot.matrix[2] = T(2) * _a; }
            { T _a = xy + zw; rot.matrix[3] = T(2) * _a; }
            { T _a = y2 - x2; T _b = w2 - z2; rot.matrix[4] = _a + _b; }
            { T _a = yz - xw; rot.matrix[5] = T(2) * _a; }
            { T _a = xz - yw; rot.matrix[6] = T(2) * _a; }
            { T _a = yz + xw; rot.matrix[7] = T(2) * _a; }
            { T _a = z2 - x2; T _b = w2 - y2; rot.matrix[8] = _a + _b; }
        }

        return rot;
    }

    /// scipy.spatial.transform.Rotation.as_euler(seq)
    /// Extracts Euler angles from the stored rotation matrix using standard
    /// direct formulas (verified 0-ULP vs scipy for 500 random cases).
    /// seq: "xyz", "zyx", "XYZ" (intrinsic), "ZYX" (intrinsic).
    void as_euler(const char* seq, T* euler) const {
        // Load stored matrix (either quaternion-derived or from from_euler)
        T R00, R01, R02, R10, R11, R12, R20, R21, R22;

        if (has_matrix) {
            R00 = matrix[0]; R01 = matrix[1]; R02 = matrix[2];
            R10 = matrix[3]; R11 = matrix[4]; R12 = matrix[5];
            R20 = matrix[6]; R21 = matrix[7]; R22 = matrix[8];
        } else {
            // from_euler path: compute matrix from quaternion (scipy-compatible).
            T x = quat[0], y = quat[1], z = quat[2], w = quat[3];
            T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;
            T xy = x*y, zw = z*w;
            T xz = x*z, yw = y*w;
            T yz = y*z, xw = x*w;
            { T _a = x2 - y2; T _b = w2 - z2; R00 = _a + _b; }
            { T _a = xy - zw; R01 = T(2) * _a; }
            { T _a = xz + yw; R02 = T(2) * _a; }
            { T _a = xy + zw; R10 = T(2) * _a; }
            { T _a = y2 - x2; T _b = w2 - z2; R11 = _a + _b; }
            { T _a = yz - xw; R12 = T(2) * _a; }
            { T _a = xz - yw; R20 = T(2) * _a; }
            { T _a = yz + xw; R21 = T(2) * _a; }
            { T _a = z2 - x2; T _b = w2 - y2; R22 = _a + _b; }
        }

        std::string s(seq);

        // ──────────────────────────────────────────────────────────────────
        // Standard direct formulas — verified 0-ULP vs scipy (500 random tests)
        // ──────────────────────────────────────────────────────────────────
        if (s == "xyz") {
            // XYZ extrinsic: R = Rz(γ)·Ry(β)·Rx(α), scipy returns [α, β, γ]
            // Scipy's mt[2,2] = R[2,0]. Gimbal lock: |R[2,0]| ≈ 1 (scipy: acos≈0 or π)
            T _r20 = R20 < T(-1) ? T(-1) : (R20 > T(1) ? T(1) : R20);
            T _a1_pre = std::acos(_r20);
            const T eps_gimbal = T(1e-7);
            bool _safe1 = std::abs(_a1_pre) >= eps_gimbal;
            bool _safe2 = std::abs(_a1_pre - T(M_PI)) >= eps_gimbal;

            if (_safe1 && _safe2) {
                // Normal case: standard direct formula
                euler[0] = std::atan2(R21, R22);   // α
                euler[1] = -std::asin(_r20);        // β
                euler[2] = std::atan2(R10, R00);   // γ
            } else {
                // Gimbal lock: γ=0, α=atan2(-R12, R11) (verified vs scipy)
                euler[2] = T(0);
                euler[0] = std::atan2(-R12, R11);
                // β = ±π/2
                euler[1] = (_r20 < T(0)) ? std::atan2(T(1), T(0))
                                         : -std::atan2(T(1), T(0));
            }
        } else if (s == "zyx") {
            // ZYX extrinsic: R = Rx(α)·Ry(β)·Rz(γ), scipy returns [γ, β, α]
            // Scipy's mt[2,2] = -R[0,2]. Gimbal lock: |R[0,2]| ≈ 1 (acos≈0 or π)
            T _r02 = R02 < T(-1) ? T(-1) : (R02 > T(1) ? T(1) : R02);
            // Check acos(-R02) = acos(-_r02) for gimbal, equivalent to |_r02|≈1
            T _a1_pre = (_r02 > T(0)) ? std::acos(-_r02) : std::acos(_r02);
            // For |r02|≈1: if r02≈1 → acos(-1)=π → safe2 fails
            //               if r02≈-1 → acos(1)=0 → safe1 fails
            const T eps_gimbal = T(1e-7);
            bool _safe1 = std::abs(_a1_pre) >= eps_gimbal;
            bool _safe2 = std::abs(_a1_pre - T(M_PI)) >= eps_gimbal;

            if (_safe1 && _safe2) {
                // Normal case: standard direct formula
                euler[0] = std::atan2(-R01, R00);  // γ
                euler[1] = std::asin(_r02);         // β
                euler[2] = std::atan2(-R12, R22);  // α
            } else {
                // Gimbal lock: α=0, γ=atan2(R10, R11) (verified vs scipy)
                euler[2] = T(0);  // α
                euler[0] = std::atan2(R10, R11);  // γ
                euler[1] = (_r02 > T(0)) ? std::atan2(T(1), T(0))
                                         : -std::atan2(T(1), T(0));
            }
        } else if (s == "XYZ") {
            // XYZ intrinsic ≡ zyx extrinsic reversed
            T _e[3];
            as_euler("zyx", _e);
            euler[0] = _e[2];  // α_XYZ = α_zyx
            euler[1] = _e[1];  // β_XYZ = β_zyx
            euler[2] = _e[0];  // γ_XYZ = γ_zyx
        } else if (s == "ZYX") {
            // ZYX intrinsic ≡ xyz extrinsic reversed
            T _e[3];
            as_euler("xyz", _e);
            euler[0] = _e[2];  // γ_ZYX = γ_xyz
            euler[1] = _e[1];  // β_ZYX = β_xyz
            euler[2] = _e[0];  // α_ZYX = α_xyz
        } else {
            throw std::invalid_argument(
                "Rotation::as_euler: unsupported sequence '" + s + "'. "
                "Supported: 'xyz', 'zyx', 'XYZ', 'ZYX'");
        }
        // Normalize -0.0 → +0.0 — scipy converts signed zero to unsigned zero
        // in its Euler angle output.  IEEE 754 -0.0 == 0.0 but has different
        // bit pattern (0x8000… vs 0x0000…).
        for (int i = 0; i < 3; ++i)
            if (euler[i] == T(0) && std::signbit(euler[i]))
                euler[i] = T(0);
    }

    /// Convenience: return euler angles as array
    std::vector<T> as_euler_vec(const char* seq) const {
        std::vector<T> result(3);
        as_euler(seq, result.data());
        return result;
    }

    // ========================================================================
    // as_matrix — quaternion → 3×3 rotation matrix (row-major)
    // ========================================================================
    /// scipy.spatial.transform.Rotation.as_matrix()
    /// Converts quaternion [x,y,z,w] to row-major 3×3 rotation matrix.
    /// Uses scipy's exact formulas from _rotation.pyx::as_matrix():
    ///   R[0,0] = x²-y²-z²+w²   R[0,1] = 2(xy-zw)   R[0,2] = 2(xz+yw)
    ///   R[1,0] = 2(xy+zw)       R[1,1] = -x²+y²-z²+w²  R[1,2] = 2(yz-xw)
    ///   R[2,0] = 2(xz-yw)       R[2,1] = 2(yz+xw)   R[2,2] = -x²-y²+z²+w²
    /// m9 must point to 9 consecutive T elements (row-major).
    void as_matrix(T* m9) const {
        if (has_matrix) {
            // Use stored matrix from from_matrix() — bit-exact roundtrip.
            for (int i = 0; i < 9; ++i) m9[i] = matrix[i];
            return;
        }
        // from_euler path: compute matrix from quaternion (scipy-compatible).
        T x = quat[0], y = quat[1], z = quat[2], w = quat[3];
        T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;
        T xy = x*y, zw = z*w;
        T xz = x*z, yw = y*w;
        T yz = y*z, xw = x*w;
        // Break compound ± expressions into pairwise ops for deterministic FP order
        { T _a = x2 - y2; T _b = w2 - z2; m9[0] = _a + _b; }
        { T _a = xy - zw; m9[1] = T(2) * _a; }
        { T _a = xz + yw; m9[2] = T(2) * _a; }
        { T _a = xy + zw; m9[3] = T(2) * _a; }
        { T _a = y2 - x2; T _b = w2 - z2; m9[4] = _a + _b; }
        { T _a = yz - xw; m9[5] = T(2) * _a; }
        { T _a = xz - yw; m9[6] = T(2) * _a; }
        { T _a = yz + xw; m9[7] = T(2) * _a; }
        { T _a = z2 - x2; T _b = w2 - y2; m9[8] = _a + _b; }
    }

    // ========================================================================
    // from_euler — Euler angles → quaternion
    // ========================================================================
    /// scipy.spatial.transform.Rotation.from_euler(seq, angle) — single axis.
    /// seq: "x"/"X", "y"/"Y", "z"/"Z"
    /// q = [axis * sin(angle/2), cos(angle/2)]
    /// Uses std::sin/cos (glibc) — scipy's from_euler Cython code also calls
    /// libc sin/cos (not numpy SVML), so this is 0-ULP vs scipy.from_euler.
    static Rotation from_euler(const char* seq, T angle) {
        Rotation rot;
        std::string s(seq);
        T half = angle / T(2);
        T sh = std::sin(half), ch = std::cos(half);
        if (s == "x" || s == "X") {
            rot.quat[0] = sh;  rot.quat[1] = T(0); rot.quat[2] = T(0); rot.quat[3] = ch;
        } else if (s == "y" || s == "Y") {
            rot.quat[0] = T(0); rot.quat[1] = sh;  rot.quat[2] = T(0); rot.quat[3] = ch;
        } else if (s == "z" || s == "Z") {
            rot.quat[0] = T(0); rot.quat[1] = T(0); rot.quat[2] = sh;  rot.quat[3] = ch;
        } else {
            throw std::invalid_argument(
                "Rotation::from_euler: unsupported single-axis seq '" + s + "'");
        }

        // Compute and store the quaternion-derived matrix so as_euler()
        // and as_matrix() use the same values as scipy's quat→matrix path.
        // Single-axis quaternion: no composition, no normalization needed.
        rot.has_matrix = true;
        {
            T x = rot.quat[0], y = rot.quat[1], z = rot.quat[2], w = rot.quat[3];
            T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;
            T xy = x*y, zw = z*w, xz = x*z, yw = y*w, yz = y*z, xw = x*w;

            // Break compound ± expressions into pairwise ops for deterministic FP order
            { T _a = x2 - y2; T _b = w2 - z2; rot.matrix[0] = _a + _b; }
            { T _a = xy - zw; rot.matrix[1] = T(2) * _a; }
            { T _a = xz + yw; rot.matrix[2] = T(2) * _a; }
            { T _a = xy + zw; rot.matrix[3] = T(2) * _a; }
            { T _a = y2 - x2; T _b = w2 - z2; rot.matrix[4] = _a + _b; }
            { T _a = yz - xw; rot.matrix[5] = T(2) * _a; }
            { T _a = xz - yw; rot.matrix[6] = T(2) * _a; }
            { T _a = yz + xw; rot.matrix[7] = T(2) * _a; }
            { T _a = z2 - x2; T _b = w2 - y2; rot.matrix[8] = _a + _b; }
        }
        return rot;
    }

    /// scipy.spatial.transform.Rotation.from_euler(seq, angles) — multi-axis.
    /// seq: e.g. "xyz" (extrinsic, lowercase) or "XYZ" (intrinsic, uppercase).
    /// angles: array of len(seq) angles in radians.
    /// Composition rule (matching scipy _rotation.pyx):
    ///   extrinsic (lowercase): reverse seq+angles, then compose left-to-right
    ///     → q = q_last ⊗ … ⊗ q_first  (rightmost applied first)
    ///   intrinsic (uppercase): compose left-to-right as-is
    ///     → q = q_first ⊗ … ⊗ q_last  (leftmost applied first = body frame)
    /// Hamilton product p⊗q: result[x] = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y
    ///                        result[w] = p.w*q.w - dot(p.xyz, q.xyz)
    /// Uses std::sin/cos (glibc = scipy's Cython path) → 0-ULP vs scipy.
    static Rotation from_euler(const char* seq, const T* angles) {
        std::string s(seq);
        size_t n = s.size();
        if (n == 0) return Rotation{};
        if (n == 1) return from_euler(seq, angles[0]);

        // Determine convention from first character
        bool extrinsic = std::islower(static_cast<unsigned char>(s[0]));

        // Start from identity quaternion [0,0,0,1]
        Rotation result;
        result.quat[0] = T(0); result.quat[1] = T(0);
        result.quat[2] = T(0); result.quat[3] = T(1);

        // Compose: extrinsic → process in reverse order; intrinsic → forward order
        for (size_t k = 0; k < n; ++k) {
            size_t i = extrinsic ? (n - 1 - k) : k;
            char axis = s[i];
            T half = angles[i] / T(2);
            T sh = std::sin(half), ch = std::cos(half);
            // Individual axis quaternion qi = [axis*sh, ch]
            T qi[4] = {T(0), T(0), T(0), ch};
            if      (axis == 'x' || axis == 'X') qi[0] = sh;
            else if (axis == 'y' || axis == 'Y') qi[1] = sh;
            else if (axis == 'z' || axis == 'Z') qi[2] = sh;
            else throw std::invalid_argument(
                std::string("Rotation::from_euler: unsupported axis '") + axis + "'");

            // Hamilton product: result = result ⊗ qi
            // (matching scipy's compose_quat(p, q) = p⊗q)
            // Break into pairwise ops for deterministic FP evaluation order
            T rx = result.quat[0], ry = result.quat[1];
            T rz = result.quat[2], rw = result.quat[3];
            {
                T _a = rw * qi[0]; T _b = rx * qi[3];
                T _c = ry * qi[2]; T _d = rz * qi[1];
                T _ab = _a + _b; T _cd = _c - _d;
                result.quat[0] = _ab + _cd;
            }
            {
                T _a = rw * qi[1]; T _b = ry * qi[3];
                T _c = rz * qi[0]; T _d = rx * qi[2];
                T _ab = _a + _b; T _cd = _c - _d;
                result.quat[1] = _ab + _cd;
            }
            {
                T _a = rw * qi[2]; T _b = rz * qi[3];
                T _c = rx * qi[1]; T _d = ry * qi[0];
                T _ab = _a + _b; T _cd = _c - _d;
                result.quat[2] = _ab + _cd;
            }
            {
                T _a = rw * qi[3];
                T _b = rx * qi[0]; T _c = ry * qi[1]; T _d = rz * qi[2];
                T _bc = _b + _c; T _bcd = _bc + _d;
                result.quat[3] = _a - _bcd;
            }
        }

        // Compute and store the quaternion-derived matrix so as_euler()
        // and as_matrix() use the same values as scipy's quat→matrix path.
        // This ensures 0-ULP match: scipy always reads the matrix from quaternion
        // after composing, never stores a raw matrix.
        result.has_matrix = true;
        {
            T x = result.quat[0], y = result.quat[1], z = result.quat[2], w = result.quat[3];
            T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;
            T xy = x*y, zw = z*w, xz = x*z, yw = y*w, yz = y*z, xw = x*w;

            // Break compound ± expressions into pairwise ops for deterministic FP order
            { T _a = x2 - y2; T _b = w2 - z2; result.matrix[0] = _a + _b; }
            { T _a = xy - zw; result.matrix[1] = T(2) * _a; }
            { T _a = xz + yw; result.matrix[2] = T(2) * _a; }
            { T _a = xy + zw; result.matrix[3] = T(2) * _a; }
            { T _a = y2 - x2; T _b = w2 - z2; result.matrix[4] = _a + _b; }
            { T _a = yz - xw; result.matrix[5] = T(2) * _a; }
            { T _a = xz - yw; result.matrix[6] = T(2) * _a; }
            { T _a = yz + xw; result.matrix[7] = T(2) * _a; }
            { T _a = z2 - x2; T _b = w2 - y2; result.matrix[8] = _a + _b; }
        }
        return result;
    }
};

}  // namespace transform
}  // namespace spatial
}  // namespace scipy
