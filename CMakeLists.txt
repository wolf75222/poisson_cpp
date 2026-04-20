cmake_minimum_required(VERSION 3.20)
project(poisson_cpp
  VERSION 0.2.0
  DESCRIPTION "Modern C++20 Poisson solvers: Thomas, SOR, spectral (DST), AMR quadtree + multigrid"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Eigen: prefer a system install (brew/apt) which exposes Eigen3::Eigen.
# Supports Eigen 3.4+ and 5.x (Homebrew now ships 5.x on macOS).
find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
  # Fallback: clone and make headers available manually, without executing
  # Eigen's own CMake (which registers hundreds of internal tests).
  message(STATUS "Eigen3 not found; downloading headers via FetchContent")
  FetchContent_Declare(
    eigen_headers
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
  FetchContent_MakeAvailable(eigen_headers)
  # Modern CMake (4.x) processes Eigen's own CMakeLists during MakeAvailable
  # and creates Eigen3::Eigen itself; only fall back to a manual alias when
  # that didn't happen.
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3_Eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
      $<BUILD_INTERFACE:${eigen_headers_SOURCE_DIR}>)
  endif()
endif()

# nlohmann/json for snapshot I/O.
find_package(nlohmann_json 3.10 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found; fetching via FetchContent")
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# FFTW3 (GPL) for the spectral solvers. Optional: the rest of the library
# builds without it.
find_package(FFTW3 MODULE)
if(FFTW3_FOUND)
  set(POISSON_HAVE_FFTW3 ON)
  message(STATUS "FFTW3 found: ${FFTW3_LIBRARY}")
else()
  set(POISSON_HAVE_FFTW3 OFF)
  message(WARNING
    "FFTW3 not found, DSTSolver1D/2D disabled. Install FFTW then rebuild:\n"
    "  Debian/Ubuntu/Colab : sudo apt-get install -y libfftw3-dev\n"
    "  macOS Homebrew      : brew install fftw\n"
    "  conda / mamba       : conda install -c conda-forge fftw\n"
    "  Fedora / RHEL       : sudo dnf install fftw-devel")
endif()

# OpenMP for parallel red-black SOR / Gauss-Seidel sweeps. Gracefully optional:
# the library degrades to single-threaded when OpenMP is absent.
# OpenMP parallel sweeps: beneficial for N >= ~384 (observed ~1.6–1.9× on
# Apple M-series, 8 threads). Adds ~10% overhead on small grids due to
# libomp link-in cost, so default OFF to preserve small-N performance;
# enable with `-DPOISSON_USE_OPENMP=ON` for large-problem workloads.
option(POISSON_USE_OPENMP "Enable OpenMP parallel sweeps" OFF)
if(POISSON_USE_OPENMP)
  # Homebrew-specific hints on macOS (AppleClang does not ship with libomp).
  if(APPLE AND NOT DEFINED OpenMP_ROOT)
    set(OpenMP_ROOT /opt/homebrew/opt/libomp)
  endif()
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    set(POISSON_HAVE_OPENMP ON)
    message(STATUS "OpenMP found: parallel sweeps enabled")
  else()
    set(POISSON_HAVE_OPENMP OFF)
    message(STATUS "OpenMP not found: using single-threaded sweeps")
  endif()
else()
  set(POISSON_HAVE_OPENMP OFF)
endif()

# -----------------------------------------------------------------------------
# Library target (header-only + compiled .cpp)
# -----------------------------------------------------------------------------
set(POISSON_SOURCES
  src/linalg/thomas.cpp
  src/fv/solver1d.cpp
  src/fv/dielectric.cpp
  src/fv/solver2d.cpp
  src/amr/quadtree.cpp
  src/amr/solver.cpp
  src/iter/poisson_cg.cpp
  src/mg/vcycle.cpp)

if(POISSON_HAVE_FFTW3)
  list(APPEND POISSON_SOURCES
    src/spectral/dst1d.cpp
    src/spectral/dst2d.cpp)
endif()

add_library(poisson ${POISSON_SOURCES})
add_library(poisson::poisson ALIAS poisson)

# Required so the static archive can be linked into the pybind11 shared
# module on Linux. macOS toolchains compile PIC by default, which is why
# this only surfaces in CI / Colab builds.
set_target_properties(poisson PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(POISSON_HAVE_FFTW3)
  target_link_libraries(poisson PUBLIC FFTW3::fftw3)
  target_compile_definitions(poisson PUBLIC POISSON_HAVE_FFTW3=1)
endif()

if(POISSON_HAVE_OPENMP)
  target_link_libraries(poisson PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(poisson PUBLIC POISSON_HAVE_OPENMP=1)
endif()

target_include_directories(poisson
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(poisson PUBLIC Eigen3::Eigen)
# nlohmann_json is not a dependency of the compiled library — it is only
# used by the header `poisson/io/json_io.hpp` (excluded from install) and
# by downstream tests that need JSON snapshot I/O.

target_compile_features(poisson PUBLIC cxx_std_20)
target_compile_options(poisson PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion>)
target_compile_options(poisson PUBLIC
  $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:
    -O3 -march=native>)

# -----------------------------------------------------------------------------
# Examples and tests
# -----------------------------------------------------------------------------
option(POISSON_BUILD_EXAMPLES "Build CLI demo examples" ON)
if(POISSON_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

option(POISSON_BUILD_BENCHMARKS "Build simple chrono-based benchmarks" ON)
if(POISSON_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

option(POISSON_BUILD_TESTS "Build Catch2 unit tests" ON)
if(POISSON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Optional Python bindings via pybind11.
# -----------------------------------------------------------------------------
option(POISSON_BUILD_PYTHON "Build pybind11 Python bindings" OFF)
if(POISSON_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 2.11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found; fetching via FetchContent")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
  endif()
  add_subdirectory(python)
endif()

# -----------------------------------------------------------------------------
# Install + package config (for downstream find_package(poisson))
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS poisson
  EXPORT  poissonTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/poisson
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "io/json_io.hpp" EXCLUDE)

install(EXPORT poissonTargets
  FILE        poissonTargets.cmake
  NAMESPACE   poisson::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poisson)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/poissonConfigVersion.cmake"
  VERSION   ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/poissonConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/poissonConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poisson)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/poissonConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/poissonConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poisson)

if(POISSON_HAVE_FFTW3)
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW3.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poisson)
endif()
