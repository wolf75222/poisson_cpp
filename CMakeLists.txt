cmake_minimum_required(VERSION 3.20)
project(poisson_cpp
  VERSION 0.1.0
  DESCRIPTION "Modern C++20 Poisson solvers: Thomas, SOR, spectral (DST), AMR quadtree + multigrid"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
  add_library(Eigen3_Eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
  target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${eigen_headers_SOURCE_DIR}>)
endif()

# -----------------------------------------------------------------------------
# Library target (header-only + compiled .cpp)
# -----------------------------------------------------------------------------
add_library(poisson
  src/linalg/thomas.cpp
  src/fv/solver1d.cpp
)
add_library(poisson::poisson ALIAS poisson)

target_include_directories(poisson
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(poisson PUBLIC Eigen3::Eigen)

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
add_subdirectory(examples)

option(POISSON_BUILD_TESTS "Build Catch2 unit tests" ON)
if(POISSON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
