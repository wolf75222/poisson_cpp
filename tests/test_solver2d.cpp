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

TEST_CASE("Solver2D: inhomogeneous eps preserves D across y-independent layers",
          "[fv][2d]") {
  // Layered eps along x (like the 1D dielectric test), rho = 0. With Dirichlet
  // uL,uR in x and Neumann in y, V depends on x only and eps_face * dV/dx is
  // the same (D-field continuity).
  const int N = 32;
  Grid2D grid(1.0, 1.0, N, N);

  Eigen::MatrixXd eps = Eigen::MatrixXd::Ones(N, N);
  eps.topRows(5).setConstant(4.0);
  eps.bottomRows(5).setConstant(4.0);

  Solver2D solver(grid, eps, /*uL=*/0.0, /*uR=*/10.0);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  const auto report = solver.solve(V, rho, {.tol = 1e-11, .max_iter = 50'000});
  REQUIRE(report.residual < 1e-11);

  // V must be y-independent.
  for (int i = 0; i < N; ++i) {
    for (int j = 1; j < N; ++j) {
      REQUIRE(std::abs(V(i, j) - V(i, 0)) < 1e-6);
    }
  }

  // D = eps_face * (V_{i+1} - V_i)/dx must be constant along x (Neumann in y
  // removes transverse flow, so the x-flux is conserved).
  const double dx = grid.dx();
  Eigen::VectorXd D(N - 1);
  for (int i = 0; i < N - 1; ++i) {
    const double eps_face = 2.0 * eps(i, 0) * eps(i + 1, 0) /
                            (eps(i, 0) + eps(i + 1, 0));
    D(i) = eps_face * (V(i + 1, 0) - V(i, 0)) / dx;
  }
  const double rel = (D.maxCoeff() - D.minCoeff()) / std::abs(D.mean());
  REQUIRE(rel < 1e-8);
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
