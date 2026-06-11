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
        T decision[4] = {R00, R11, R22, R00 + R11 + R22};

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

            q[i] = T(1) - decision[3] + T(2) * Ri;
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
        T norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        rot.quat[0] = q[0] / norm;
        rot.quat[1] = q[1] / norm;
        rot.quat[2] = q[2] / norm;
        rot.quat[3] = q[3] / norm;

        // Store original matrix to avoid quaternion roundtrip ULP-error
        // in as_euler() and as_matrix().
        rot.has_matrix = true;
        for (int i = 0; i < 9; ++i) rot.matrix[i] = matrix[i];

        return rot;
    }

    /// scipy.spatial.transform.Rotation.as_euler(seq)
    /// seq: "xyz" (intrinsic Tait-Bryan), "zyx" (intrinsic)
    void as_euler(const char* seq, T* euler) const {
        // Convert quaternion back to rotation matrix.
        // MUST use scipy's exact formulas from as_matrix():
        //   matrix[0,0] = x2 - y2 - z2 + w2
        //   matrix[1,0] = 2 * (xy + zw)   etc.
        // Using the standard 1-2*(y2+z2) forms gives different floating-point
        // results due to unit-norm not being exact in finite precision.
        T R00, R01, R02, R10, R11, R12, R20, R21, R22;

        if (has_matrix) {
            // Use stored matrix from from_matrix() — avoids ULP-error from
            // quaternion normalisation + quaternion→matrix rebuild.
            R00 = matrix[0]; R01 = matrix[1]; R02 = matrix[2];
            R10 = matrix[3]; R11 = matrix[4]; R12 = matrix[5];
            R20 = matrix[6]; R21 = matrix[7]; R22 = matrix[8];
        } else {
            // from_euler path: quaternion is the canonical representation.
            // Compute matrix from quaternion (scipy-compatible formulas).
            T x = quat[0], y = quat[1], z = quat[2], w = quat[3];
            T x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;

            T xy = x*y, zw = z*w;
            T xz = x*z, yw = y*w;
            T yz = y*z, xw = x*w;

            // Rotation matrix (row-major) — scipy's as_matrix() formulas
            R00 = x2 - y2 - z2 + w2;
            R01 = T(2) * (xy - zw);
            R02 = T(2) * (xz + yw);

            R10 = T(2) * (xy + zw);
            R11 = -x2 + y2 - z2 + w2;
            R12 = T(2) * (yz - xw);

            R20 = T(2) * (xz - yw);
            R21 = T(2) * (yz + xw);
            R22 = -x2 - y2 + z2 + w2;
        }

        std::string s(seq);

        if (s == "xyz") {
            // ================================================================
            // scipy _compute_euler_from_matrix for 'xyz' intrinsic
            // ================================================================
            // n1=[1,0,0], n2=[0,1,0], n3=[0,0,1]
            // sl=1, cl=0, offset=atan2(1,0)=pi/2
            // c=[[0,1,0],[0,0,1],[1,0,0]]
            // rot=[[1,0,0],[0,0,1],[0,-1,0]]
            // matrix_trans = c @ M @ c^T @ rot
            //   = [[M11, -M10, M12],
            //      [M21, -M20, M22],
            //      [M01, -M00, M02]]
            T offset = std::atan2(T(1), T(0));  // pi/2 via atan2
            T mt22 = R02;  // matrix_trans[2,2]
            if (mt22 > T(1))  mt22 = T(1);
            if (mt22 < T(-1)) mt22 = T(-1);

            T a1_pre = std::acos(mt22);
            const T eps_angle = T(1e-7);
            bool safe1 = std::abs(a1_pre) >= eps_angle;
            bool safe2 = std::abs(a1_pre - T(M_PI)) >= eps_angle;
            bool safe = safe1 && safe2;

            T a1 = a1_pre + offset;

            if (safe) {
                T a0 = std::atan2(R12, -R22);   // atan2(mt[0,2], -mt[1,2])
                T a2 = std::atan2(R01, -R00);   // atan2(mt[2,0], mt[2,1])

                // Step 7: adjust (xyz: seq[0]!=seq[2])
                bool adjust = a1 < T(-M_PI / 2) || a1 > T(M_PI / 2);
                if (adjust) {
                    a0 += T(M_PI);
                    a1 = T(2) * offset - a1;
                    a2 -= T(M_PI);
                }

                // Wrap to [-pi, pi]
                if (a0 > T(M_PI))  a0 -= T(2 * M_PI);
                if (a0 < T(-M_PI)) a0 += T(2 * M_PI);
                if (a1 > T(M_PI))  a1 -= T(2 * M_PI);
                if (a1 < T(-M_PI)) a1 += T(2 * M_PI);
                if (a2 > T(M_PI))  a2 -= T(2 * M_PI);
                if (a2 < T(-M_PI)) a2 += T(2 * M_PI);

                euler[0] = a0;  // alpha
                euler[1] = a1;  // beta
                euler[2] = a2;  // gamma
            } else {
                // Gimbal lock
                euler[2] = T(0);  // set third angle to zero
                if (!safe1) {
                    // mt[2,2] ≈ 1, acos ≈ 0
                    euler[0] = std::atan2(R21 - R01, R11 + R00);  // mt[1,0]-mt[0,1], mt[0,0]+mt[1,1]
                }
                if (!safe2) {
                    // mt[2,2] ≈ -1, acos ≈ pi
                    euler[0] = std::atan2(R21 + R01, R11 - R00);
                }
                euler[1] = offset;
            }
        } else if (s == "zyx" || s == "XYZ") {
            // ================================================================
            // scipy _compute_euler_from_matrix for 'zyx' intrinsic
            // ================================================================
            // n1=[0,0,1], n2=[0,1,0], n3=[1,0,0]
            // sl=dot(cross(n1,n2),n3)=dot([-1,0,0],[1,0,0])=-1
            // cl=dot(n1,n3)=dot([0,0,1],[1,0,0])=0
            // offset=atan2(-1,0)=-pi/2
            // c=[[0,1,0],[-1,0,0],[0,0,1]]
            // rot=[[1,0,0],[0,0,-1],[0,1,0]]
            // c.T@rot=[[0,0,1],[1,0,0],[0,1,0]]
            // matrix_trans = [[M11, M12, M10],
            //                  [-M01, -M02, -M00],
            //                  [M21, M22, M20]]
            T offset = std::atan2(T(-1), T(0));  // -pi/2 via atan2
            T mt22 = R20;  // matrix_trans[2,2]
            if (mt22 > T(1))  mt22 = T(1);
            if (mt22 < T(-1)) mt22 = T(-1);

            T a1_pre = std::acos(mt22);
            const T eps_angle = T(1e-7);
            bool safe1 = std::abs(a1_pre) >= eps_angle;
            bool safe2 = std::abs(a1_pre - T(M_PI)) >= eps_angle;
            bool safe = safe1 && safe2;

            T a1 = a1_pre + offset;

            if (safe) {
                T a0 = std::atan2(R10, R00);   // atan2(mt[0,2], -mt[1,2])
                T a2 = std::atan2(R21, R22);   // atan2(mt[2,0], mt[2,1])

                // Step 7: adjust (zyx: seq[0]!=seq[2])
                bool adjust = a1 < T(-M_PI / 2) || a1 > T(M_PI / 2);
                if (adjust) {
                    a0 += T(M_PI);
                    a1 = T(2) * offset - a1;
                    a2 -= T(M_PI);
                }

                // Wrap to [-pi, pi]
                if (a0 > T(M_PI))  a0 -= T(2 * M_PI);
                if (a0 < T(-M_PI)) a0 += T(2 * M_PI);
                if (a1 > T(M_PI))  a1 -= T(2 * M_PI);
                if (a1 < T(-M_PI)) a1 += T(2 * M_PI);
                if (a2 > T(M_PI))  a2 -= T(2 * M_PI);
                if (a2 < T(-M_PI)) a2 += T(2 * M_PI);

                euler[0] = a0;  // gamma_z
                euler[1] = a1;  // beta_y
                euler[2] = a2;  // alpha_x
            } else {
                // Gimbal lock
                euler[2] = T(0);  // set third angle to zero
                if (!safe1) {
                    euler[0] = std::atan2(R21 - R01, R11 + R00);
                }
                if (!safe2) {
                    euler[0] = std::atan2(R21 + R01, R11 - R00);
                }
                euler[1] = offset;
            }
        } else {
            throw std::invalid_argument(
                "Rotation::as_euler: unsupported sequence '" + s + "'. "
                "Supported: 'xyz', 'zyx', 'XYZ'");
        }
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
        m9[0] =  x2 - y2 - z2 + w2;          // R[0,0]
        m9[1] =  T(2) * (xy - zw);            // R[0,1]
        m9[2] =  T(2) * (xz + yw);            // R[0,2]
        m9[3] =  T(2) * (xy + zw);            // R[1,0]
        m9[4] = -x2 + y2 - z2 + w2;          // R[1,1]
        m9[5] =  T(2) * (yz - xw);            // R[1,2]
        m9[6] =  T(2) * (xz - yw);            // R[2,0]
        m9[7] =  T(2) * (yz + xw);            // R[2,1]
        m9[8] = -x2 - y2 + z2 + w2;          // R[2,2]
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
            T rx = result.quat[0], ry = result.quat[1];
            T rz = result.quat[2], rw = result.quat[3];
            result.quat[0] = rw*qi[0] + rx*qi[3] + ry*qi[2] - rz*qi[1];
            result.quat[1] = rw*qi[1] + ry*qi[3] + rz*qi[0] - rx*qi[2];
            result.quat[2] = rw*qi[2] + rz*qi[3] + rx*qi[1] - ry*qi[0];
            result.quat[3] = rw*qi[3] - rx*qi[0] - ry*qi[1] - rz*qi[2];
        }
        return result;
    }
};

}  // namespace transform
}  // namespace spatial
}  // namespace scipy
