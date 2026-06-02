// scipy.spatial.transform — Rotation (from_matrix, as_euler, etc.)
//
// Aligns with Python: scipy.spatial.transform.Rotation
// Usage: #include "scipy/transform.h"

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace scipy {
namespace spatial {
namespace transform {

// ============================================================================
// Rotation — 3D rotation representation
// ============================================================================
// Supports: from_matrix(matrix) and as_euler(seq)
// Currently supports 'xyz' (intrinsic Tait-Bryan) and 'zyx' sequences.

template<typename T>
struct Rotation {
    // Rotation matrix (3x3 row-major)
    T R[9];

    Rotation() {
        R[0] = 1; R[1] = 0; R[2] = 0;
        R[3] = 0; R[4] = 1; R[5] = 0;
        R[6] = 0; R[7] = 0; R[8] = 1;
    }

    /// scipy.spatial.transform.Rotation.from_matrix(R)
    /// R must be 3x3 row-major matrix.
    static Rotation from_matrix(const T* matrix) {
        Rotation rot;
        for (int i = 0; i < 9; ++i) rot.R[i] = matrix[i];
        return rot;
    }

    /// scipy.spatial.transform.Rotation.as_euler(seq)
    /// seq: "xyz" (intrinsic, equivalent to extrinsic ZYX)
    ///        "zyx" (intrinsic, equivalent to extrinsic XYZ)
    ///
    /// Intrinsic "xyz": R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
    ///   Sequence: first rotate alpha about x, then beta about y, then gamma about z.
    ///   Returns [alpha, beta, gamma].
    ///
    /// Intrinsic "zyx": R = Rx(alpha) @ Ry(beta) @ Rz(gamma)
    ///   Sequence: first rotate gamma about z, then beta about y, then alpha about x.
    ///   Returns [alpha, beta, gamma].
    void as_euler(const char* seq, T* euler) const {
        std::string s(seq);
        const T eps = T(0.999999);

        if (s == "xyz") {
            // R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
            // R[2,0] = -sin(beta)
            T sb = -R[6];  // -R[2,0]
            if (std::abs(sb) < eps) {
                T alpha = std::atan2(R[7], R[8]);   // R[2,1], R[2,2]
                T beta  = std::asin(sb);
                T gamma = std::atan2(R[3], R[0]);    // R[1,0], R[0,0]
                euler[0] = alpha;
                euler[1] = beta;
                euler[2] = gamma;
            } else {
                // Gimbal lock: cos(beta) ≈ 0
                T gamma = T(0);
                if (sb <= -T(1)) {
                    T beta  = T(-M_PI / 2);
                    T alpha = std::atan2(-R[1], R[4]);  // -R[0,1], R[1,1]
                    euler[0] = alpha;
                    euler[1] = beta;
                    euler[2] = gamma;
                } else {
                    T beta  = T(M_PI / 2);
                    T alpha = std::atan2(-R[1], R[4]);
                    euler[0] = alpha;
                    euler[1] = beta;
                    euler[2] = gamma;
                }
            }
        } else if (s == "zyx" || s == "XYZ") {
            // R = Rx(alpha_x) @ Ry(beta_y) @ Rz(gamma_z)
            // where alpha_x applied last, gamma_z applied first.
            // scipy 'zyx' returns [z_angle, y_angle, x_angle]
            // R[0,2] = sin(beta_y)
            T sb = R[2];  // R[0,2]
            if (std::abs(sb) < eps) {
                T alpha_x = std::atan2(-R[5], R[8]);  // -R[1,2], R[2,2]
                T beta_y  = std::asin(sb);
                T gamma_z = std::atan2(-R[1], R[0]);  // -R[0,1], R[0,0]
                euler[0] = gamma_z;  // z-angle first
                euler[1] = beta_y;   // y-angle
                euler[2] = alpha_x;  // x-angle last
            } else {
                // Gimbal lock
                T gamma_z = T(0);
                if (sb <= -T(1)) {
                    T beta_y  = T(-M_PI / 2);
                    T alpha_x = std::atan2(R[3], R[4]);  // R[1,0], R[1,1]
                    euler[0] = gamma_z;
                    euler[1] = beta_y;
                    euler[2] = alpha_x;
                } else {
                    T beta_y  = T(M_PI / 2);
                    T alpha_x = std::atan2(R[3], R[4]);
                    euler[0] = gamma_z;
                    euler[1] = beta_y;
                    euler[2] = alpha_x;
                }
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
};

}  // namespace transform
}  // namespace spatial
}  // namespace scipy
