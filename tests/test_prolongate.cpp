#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include "poisson/mg/vcycle.hpp"

TEST_CASE("prolongate_const replicates each coarse value to 4 fine cells",
          "[mg][prolongate]") {
  Eigen::MatrixXd c(2, 2);
  c << 1.0, 2.0,
       3.0, 4.0;
  const Eigen::MatrixXd f = poisson::mg::prolongate_const(c);
  REQUIRE(f.rows() == 4);
  REQUIRE(f.cols() == 4);
  // Each 2x2 block of f matches one coarse cell.
  REQUIRE(f.block(0, 0, 2, 2).isConstant(1.0));
  REQUIRE(f.block(2, 0, 2, 2).isConstant(3.0));
  REQUIRE(f.block(0, 2, 2, 2).isConstant(2.0));
  REQUIRE(f.block(2, 2, 2, 2).isConstant(4.0));
}

TEST_CASE("prolongate_bilinear exactly reproduces an affine function",
          "[mg][prolongate]") {
  // Set c(I, J) = a + b*I + c*J at coarse centers. Bilinear interpolation
  // at fine centers (offset by ±1/2 in coarse units) should be exact.
  const int M = 8;
  const double a = 1.25, bx = 0.5, by = -0.3;
  Eigen::MatrixXd c(M, M);
  for (int J = 0; J < M; ++J) {
    for (int I = 0; I < M; ++I) {
      c(I, J) = a + bx * (I + 0.5) + by * (J + 0.5);
    }
  }
  const Eigen::MatrixXd f = poisson::mg::prolongate_bilinear(c);
  const int N = 2 * M;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const double xf = (i + 0.5) / 2.0;  // in coarse units
      const double yf = (j + 0.5) / 2.0;
      const double expected = a + bx * xf + by * yf;
      // Near the physical edges the bilinear formula uses ghost = 0 and
      // deviates; check only interior fine cells that lie inside all four
      // contributing coarse cells' convex hull.
      if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        REQUIRE(std::abs(f(i, j) - expected) < 1e-12);
      }
    }
  }
}
