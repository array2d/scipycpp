// Native C++ scipy implementations — header-only, zero pybind11 dependency.
//
// Based on: numpcpp (numpy primitives), Eigen3 (linalg), pocketfft (FFT).
//
// This is the main entry point that includes all scipy submodules.

#pragma once

#include "integrate.h"
#include "optimize.h"
#include "interpolate.h"
#include "signal.h"
#include "stats.h"
#include "spatial.h"
#include "special.h"
#include "linalg.h"
#include "fft.h"
#include "ndimage.h"
#include "transform.h"
