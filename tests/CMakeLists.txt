# Build scipycpp Python test module (bit-level alignment tests).
#
# Flags follow the numpcpp README bit-exact backend spec exactly.
# Build:
#   cmake -S tests -B tests/build
#   cmake --build tests/build
# Test:
#   cd tests && python3 -m pytest test_all.py -q --tb=short --no-header

cmake_minimum_required(VERSION 3.16)
project(scipycpp_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Find dependencies -------------------------------------------------------
# Python3 (for pybind11 extension)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 — prefer the copy installed alongside Python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmake_result
    ERROR_QUIET
)
if(NOT (_pybind11_cmake_result EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}"))
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_cmake_result
        ERROR_QUIET
    )
endif()
if(_pybind11_cmake_result EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED
        PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Eigen3 for linalg.h
find_package(Eigen3 REQUIRED)

# numpcpp for numpy.h
find_package(numpycpp REQUIRED)

# OpenMP (used in ndimage.h, etc.)
find_package(OpenMP)

# AVX-512 runtime probe — compile AND run a tiny binary that executes a ZMM
# instruction.  check_cxx_source_runs returns TRUE only if the binary exits 0,
# so it correctly distinguishes machines where CPUID says avx512f but the
# hypervisor traps ZMM execution (e.g. some Azure VM SKUs used by GitHub Actions).
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mfma")
check_cxx_source_runs("
    #include <immintrin.h>
    int main() {
        __m512d a = _mm512_set1_pd(2.0);
        __m512d b = _mm512_sqrt_pd(a);
        (void)b;
        return 0;
    }
" SCIPYCPP_AVX512_WORKS)
unset(CMAKE_REQUIRED_FLAGS)

# ---- Python extension module -------------------------------------------------
pybind11_add_module(scipycpp MODULE module.cpp)

# Include paths: parent dir (for scipy/, pycpp/) + Eigen3 + numpcpp
target_include_directories(scipycpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${EIGEN3_INCLUDE_DIR}
)

# Link transitive dependencies
target_link_libraries(scipycpp PRIVATE
    Eigen3::Eigen
    numpycpp::numpycpp
)

# ── bit-exact compile flags (from numpcpp README) ──────────────────────────
target_compile_options(scipycpp PRIVATE
    -O2
    -ffp-contract=off          # REQUIRED — prevents silent FMA fusion
    -msse4.1                   # REQUIRED — _mm_insert_epi32 in linalg.h
    -mfma                      # REQUIRED — _mm512_fmadd_* in AVX-512 loops
    # Disable GCC builtin replacements — defensive: prevents future GCC versions
    # from silently substituting npy_* call sites with builtins.
    -fno-builtin-exp   -fno-builtin-log   -fno-builtin-sin
    -fno-builtin-cos   -fno-builtin-tan   -fno-builtin-pow
    -fno-builtin-sqrt  -fno-builtin-atan2 -fno-builtin-log2
    -fno-builtin-log10 -fno-builtin-asin  -fno-builtin-acos
    -fno-builtin-atan  -fno-builtin-exp2
    -fno-builtin-cbrt  -fno-builtin-expm1 -fno-builtin-log1p
)

if(SCIPYCPP_AVX512_WORKS)
    target_compile_options(scipycpp PRIVATE
        -mavx512f -mprefer-vector-width=256)
    message(STATUS "AVX-512: usable — SVML + wide-loop path enabled")
else()
    message(STATUS "AVX-512: not usable (missing or hypervisor-trapped) — scalar npy_* path (still bit-exact)")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(scipycpp PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(scipycpp PRIVATE dl)

# Place .so next to the test scripts for easy import
set_target_properties(scipycpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
    SUFFIX ".so"
)

# ---- Test target -------------------------------------------------------------
add_custom_target(test
    COMMAND "${Python3_EXECUTABLE}" -m pytest
            "${CMAKE_CURRENT_SOURCE_DIR}/test_all.py" -q --tb=short --no-header
    DEPENDS scipycpp
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running bit-exact alignment tests"
)

message(STATUS "scipycpp test module")
message(STATUS "  Python: ${Python3_VERSION}  (${Python3_EXECUTABLE})")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  Build: cmake --build <build_dir>")
message(STATUS "  Test:  cmake --build <build_dir> --target test")
