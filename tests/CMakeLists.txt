include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.3)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

set(TEST_SOURCES
  test_thomas.cpp
  test_solver1d.cpp
  test_dielectric.cpp
  test_solver2d.cpp)

if(POISSON_HAVE_FFTW3)
  list(APPEND TEST_SOURCES test_dst1d.cpp test_dst2d.cpp)
endif()

add_executable(tests_poisson ${TEST_SOURCES})
target_link_libraries(tests_poisson PRIVATE poisson::poisson Catch2::Catch2WithMain)
target_compile_options(tests_poisson PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic>)

catch_discover_tests(tests_poisson)
