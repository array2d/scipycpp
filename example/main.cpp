#include <scipy/core.h>
#include <iostream>
#include <vector>
#include <complex>

int main() {
    // --- Integration ---
    std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0};
    std::cout << "trapezoid: " << scipy::integrate::trapezoid(y.data(), y.size()) << std::endl;
    std::cout << "simpson:   " << scipy::integrate::simpson(y.data(), nullptr, y.size()) << std::endl;

    auto [val, err] = scipy::integrate::quad([](double x) { return x * x; }, 0.0, 4.0);
    std::cout << "quad(x^2, 0,4): " << val << " (err=" << err << ")" << std::endl;

    // --- Stats ---
    std::vector<double> x = {-1.0, 0.0, 1.0};
    std::vector<double> pdf(3);
    scipy::stats::norm_pdf(x.data(), pdf.data(), 3);
    std::cout << "norm.pdf: " << pdf[0] << " " << pdf[1] << " " << pdf[2] << std::endl;

    // --- Linalg ---
    double A[4] = {2.0, 1.0, 1.0, 3.0};
    double b[2] = {5.0, 6.0};
    double xs[2];
    scipy::linalg::solve(A, xs, b, 2);
    std::cout << "solve: x=[" << xs[0] << ", " << xs[1] << "]" << std::endl;

    // --- FFT ---
    std::vector<std::complex<double>> sig = {{1,0}, {2,0}, {3,0}, {4,0}};
    std::vector<std::complex<double>> spec(4);
    scipy::fft::fft(sig.data(), spec.data(), 4);
    std::cout << "fft:";
    for (auto& v : spec) std::cout << " (" << v.real() << "," << v.imag() << ")";
    std::cout << std::endl;

    return 0;
}
