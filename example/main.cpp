// Include only the scipy modules we use (stats.h requires SVML runtime bridge init)
#include "scipy/integrate.h"
#include "scipy/linalg.h"
#include "scipy/special.h"
#include "scipy/spatial.h"
#include "scipy/ndimage.h"
#include "scipy/signal.h"
#include "scipy/transform.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // --- Integration ---
    {
        std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0};
        std::cout << "trapezoid: "
                  << scipy::integrate::trapezoid(y.data(), static_cast<size_t>(y.size()))
                  << std::endl;
        std::cout << "simpson:   "
                  << scipy::integrate::simpson(y.data(), static_cast<size_t>(y.size()))
                  << std::endl;

        auto [val, err] = scipy::integrate::quad(
            [](double x) { return x * x; }, 0.0, 4.0);
        std::cout << "quad(x^2, 0,4): " << val << " (err=" << err << ")" << std::endl;
    }

    // --- Special ---
    {
        std::cout << "erf(1.0) = " << scipy::special::erf(1.0) << std::endl;
        std::cout << "gamma(5.0) = " << scipy::special::gamma(5.0) << std::endl;
    }

    // --- Linalg ---
    {
        double A[4] = {2.0, 1.0, 1.0, 3.0};
        double b[2] = {5.0, 6.0};
        double xs[2];
        if (scipy::linalg::solve(A, xs, b, 2))
            std::cout << "solve: x=[" << xs[0] << ", " << xs[1] << "]" << std::endl;
        std::cout << "det: " << scipy::linalg::det(A, 2) << std::endl;
    }

    // --- Spatial Distance ---
    {
        // cdist — cross-set distance matrix
        std::vector<double> XA = {0,0, 1,1, 2,2};       // 3x2
        std::vector<double> XB = {0,1, 1,2};             // 2x2
        std::vector<double> dm(3*2);
        scipy::spatial::distance::cdist(XA.data(), 3, XB.data(), 2, 2, dm.data());
        std::cout << "cdist(euclidean): [" << dm[0] << ", " << dm[1] << ", ...]" << std::endl;

        // KDTree — nearest neighbor query with distances
        double pts[] = {0,0, 1,1, 2,2, 3,3};
        double q[]   = {0.5, 0.5};
        scipy::spatial::KDTree<double> tree(pts, 4, 2);
        double dist; size_t idx;
        tree.query(q, dist, idx);
        std::cout << "KDTree.query → dist=" << dist << " idx=" << idx << std::endl;

        // k-nearest
        double dists[2]; size_t indices[2];
        tree.query(q, dists, indices, 2);
        std::cout << "KDTree.query(k=2) → dist=[" << dists[0] << "," << dists[1]
                  << "] idx=[" << indices[0] << "," << indices[1] << "]" << std::endl;
    }

    // --- ndimage ---
    {
        std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> out(5);
        scipy::ndimage::gaussian_filter1d(a.data(), out.data(), 5, 1.0);
        std::cout << "gaussian_filter1d(sigma=1):";
        for (auto& v : out) std::cout << " " << v;
        std::cout << std::endl;
    }

    // --- Signal ---
    {
        std::vector<double> a = {1.0, 5.0, 2.0, 8.0, 3.0};
        std::vector<double> out(5);
        scipy::signal::medfilt(a.data(), out.data(), 5, 3);
        std::cout << "medfilt(k=3):";
        for (auto& v : out) std::cout << " " << v;
        std::cout << std::endl;
    }

    // --- Spatial Transform ---
    {
        using T = double;
        // Rotation matrix from euler angles: Rx(20°) @ Ry(30°) @ Rz(45°)
        auto deg = [](T d) { return d * M_PI / 180.0; };
        T rx = deg(20), ry = deg(30), rz = deg(45);
        T cx = std::cos(rx), sx = std::sin(rx);
        T cy = std::cos(ry), sy = std::sin(ry);
        T cz = std::cos(rz), sz = std::sin(rz);

        T R[9] = {
            cy*cz,               -cy*sz,              sy,
            cx*sz + sx*sy*cz,     cx*cz - sx*sy*sz,  -sx*cy,
            sx*sz - cx*sy*cz,     sx*cz + cx*sy*sz,   cx*cy
        };

        auto rot = scipy::spatial::transform::Rotation<T>::from_matrix(R);
        auto euler = rot.as_euler_vec("xyz");
        std::cout << "Rotation.as_euler(xyz): ["
                  << euler[0]*180/M_PI << "°, "
                  << euler[1]*180/M_PI << "°, "
                  << euler[2]*180/M_PI << "°]" << std::endl;
    }

    return 0;
}
