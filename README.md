# scipycpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

## Overview

`scipycpp` is a **header-only C++ library** implementing scipy's core APIs with
**bit-level precision alignment** against Python scipy.

Built on proven C++ libraries:
- **[numpycpp](https://github.com/array2d/numpycpp)** — numpy primitives + SVML bridge
- **[Eigen3](https://eigen.tuxfamily.org/)** — linear algebra (solve, inv, det, eig, svd, cholesky)
- **[pocketfft](https://github.com/mreineck/pocketfft)** — FFT (same library numpy/scipy use internally)

## Quick Start

```cpp
#include "scipy/core.h"

// Integration
auto [val, err] = scipy::integrate::quad([](double x){ return x*x; }, 0.0, 4.0);

// Stats
std::vector<double> pdf(3);
scipy::stats::norm_pdf(data, pdf.data(), 3);

// Linalg (Eigen3-backed)
double A[4] = {2,1,1,3}, b[2] = {5,6}, x[2];
scipy::linalg::solve(A, x, b, 2);

// FFT (pocketfft-backed — same as numpy/scipy)
scipy::fft::fft(signal, spectrum, 4);
```

### Dependencies

```bash
# Install dependencies
sudo dpkg -i numpycpp-dev-*.deb
sudo apt-get install libeigen3-dev
sudo dpkg -i pocketfft-dev-*.deb

# Install scipycpp
mkdir build && cd build
cmake .. && make deb
sudo dpkg -i scipycpp-dev-*.deb
```

```cmake
find_package(scipycpp REQUIRED)
target_link_libraries(myapp PRIVATE scipycpp::scipycpp)
```

## Modules

| Module | Backend | Key APIs |
|--------|---------|----------|
| `integrate` | numpcpp + pure C++ | quad, simpson, trapezoid |
| `optimize` | pure C++ | minimize_scalar (Brent), root_scalar |
| `interpolate` | pure C++ | interp1d, CubicSpline |
| `signal` | numpcpp | convolve, correlate |
| `stats` | **numpycpp SVML** | norm.pdf/cdf/ppf (bit-exact) |
| `spatial` | pure C++ | euclidean, cosine, manhattan, KDTree |
| `special` | C++17 std | erf, gamma, beta, digamma, logsumexp |
| `linalg` | **Eigen3** | solve, inv, det, eigvalsh, cholesky, svd |
| `fft` | **pocketfft** | fft, ifft, rfft, irfft |

## Testing

```bash
cd tests && make && make test
```

## License

MIT
