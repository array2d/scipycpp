cmake_minimum_required(VERSION 3.16)

# Auto-detect scipy version from the Python environment
execute_process(
    COMMAND python3 -c "import scipy; print(scipy.__version__)"
    OUTPUT_VARIABLE SCIPY_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _scipy_ver_result
)
if(_scipy_ver_result OR SCIPY_VERSION STREQUAL "")
    message(FATAL_ERROR "Cannot detect scipy version. Is 'scipy' installed (pip install scipy)?")
endif()
message(STATUS "scipy version detected: ${SCIPY_VERSION}")

project(scipycpp VERSION ${SCIPY_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- INTERFACE target for find_package consumers ----------------------------
add_library(scipycpp INTERFACE)
target_include_directories(scipycpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/scipycpp>
)
target_compile_features(scipycpp INTERFACE cxx_std_17)

# Transitive dependencies: scipycpp → Eigen3, numpcpp
target_link_libraries(scipycpp INTERFACE
    Eigen3::Eigen
    numpycpp::numpycpp
)

# ---- Install — header-only C++ library --------------------------------------
install(DIRECTORY scipy
    DESTINATION include/scipycpp
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(DIRECTORY pycpp
    DESTINATION include/scipycpp
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".pytest_cache" EXCLUDE
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    scipycpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/scipycpp-config.cmake
    INSTALL_DESTINATION lib/cmake/scipycpp
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/scipycpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/scipycpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/scipycpp-config-version.cmake
    DESTINATION lib/cmake/scipycpp
)

# ---- Docs & examples -------------------------------------------------------
install(FILES README.md DESTINATION share/scipycpp)
install(DIRECTORY example/ DESTINATION share/scipycpp/example)

# ---- CPack DEB packaging ----------------------------------------------------
set(CPACK_PACKAGE_NAME                     "scipycpp-dev")
set(CPACK_PACKAGE_VERSION                  ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR                   "array2d")
set(CPACK_PACKAGE_CONTACT                  "array2d")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY      "C++ header-only scipy API — based on numpcpp+Eigen3+pocketfft")
set(CPACK_GENERATOR                        "DEB")
set(CPACK_DEB_COMPONENT_INSTALL            OFF)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER        "array2d")
set(CPACK_DEBIAN_PACKAGE_SECTION           "libdevel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY          "optional")
set(CPACK_DEBIAN_FILE_NAME                 "scipycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")

set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

# Recommend the required dependencies
set(CPACK_DEBIAN_PACKAGE_DEPENDS "numpycpp-dev, libeigen3-dev")

include(CPack)
add_custom_target(deb
    COMMAND cpack -G DEB --config CPackConfig.cmake
    COMMENT "Packaging scipycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")

message(STATUS "scipycpp v${PROJECT_VERSION}  (header-only C++ library)")
message(STATUS "  Dependencies: numpcpp + Eigen3")
message(STATUS "  Bundled: pocketfft (FFT) + ckdtree (KDTree)")
message(STATUS "  DEB:  make deb  →  scipycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")
message(STATUS "  Test: cmake -S tests -B tests/build && cmake --build tests/build --target test")
