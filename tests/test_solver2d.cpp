#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver2d.hpp"

using poisson::Grid2D;
using poisson::fv::Solver2D;

TEST_CASE("Solver2D: zero charge gives linear profile in x, constant in y",
          "[fv][2d]") {
  const int N = 32;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, /*eps=*/1.0, /*uL=*/10.0, /*uR=*/100.0);

  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  const auto report = solver.solve(V, rho, {.tol = 1e-10, .max_iter = 20'000});

  REQUIRE(report.iterations > 0);
  REQUIRE(report.residual < 1e-10);

  // Analytical: V(x) = uL + (uR - uL) * x / Lx, with x_i = (i + 0.5) * dx.
  const double dx = grid.dx();
  for (int i = 0; i < N; ++i) {
    const double x = (i + 0.5) * dx;
    const double V_theo = 10.0 + (100.0 - 10.0) * x / 1.0;
    for (int j = 0; j < N; ++j) {
      REQUIRE(std::abs(V(i, j) - V_theo) < 1e-4);
    }
  }
}

TEST_CASE("Solver2D: symmetric in y under y-independent source", "[fv][2d]") {
  const int N = 24;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, /*eps=*/1.0, /*uL=*/0.0, /*uR=*/0.0);

  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  // rho depends only on i (through x), not on j.
  Eigen::MatrixXd rho(N, N);
  for (int i = 0; i < N; ++i) {
    const double x = (i + 0.5) * grid.dx();
    rho.row(i).setConstant(std::sin(3.14159265358979323846 * x));
  }
  solver.solve(V, rho, {.tol = 1e-10});

  // Neumann in y: V should be independent of j in steady state.
  for (int i = 0; i < N; ++i) {
    const double v0 = V(i, 0);
    for (int j = 0; j < N; ++j) {
      REQUIRE(std::abs(V(i, j) - v0) < 1e-6);
    }
  }
}

TEST_CASE("Solver2D: auto omega converges", "[fv][2d]") {
  const int N = 16;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, /*eps=*/1.0, /*uL=*/0.0, /*uR=*/1.0);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  const auto report = solver.solve(V, rho, {.omega = -1, .tol = 1e-10});
  REQUIRE(report.iterations < 200);   // omega_opt is fast
  REQUIRE(report.residual < 1e-10);
}
