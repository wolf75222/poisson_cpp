#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/spectral/dst1d.hpp"

using poisson::spectral::DSTSolver1D;

TEST_CASE("DSTSolver1D: recovers a discrete eigenmode to machine precision",
          "[spectral][1d]") {
  // If rho_i = eps0 * lambda_k * sin(k i pi / (N+1)), then V = sin(...).
  const int N = 64;
  const double L = 1.0, eps0 = 1.0;
  const int k_mode = 4;
  const double h = L / (N + 1);

  Eigen::VectorXd V_theo(N);
  for (int i = 1; i <= N; ++i) {
    V_theo(i - 1) = std::sin(k_mode * i * std::numbers::pi / (N + 1));
  }
  const double s = std::sin(k_mode * std::numbers::pi / (2.0 * (N + 1)));
  const double lam_k = 4.0 / (h * h) * s * s;
  const Eigen::VectorXd rho = eps0 * lam_k * V_theo;

  DSTSolver1D solver(N, L, eps0);
  const Eigen::VectorXd V = solver.solve(rho);
  REQUIRE((V - V_theo).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("DSTSolver1D: manufactured sin(pi x/L) gives O(h^2) error",
          "[spectral][1d]") {
  // V(x) = sin(pi x/L), rho(x) = eps0 (pi/L)^2 sin(pi x/L).
  // Continuous eigenvalue vs discrete eigenvalue differs by O(h^2),
  // so the solver returns V to O(h^2).
  const double L = 1.0, eps0 = 1.0;
  double err_prev = 0.0;
  for (int N : {31, 63, 127, 255}) {
    const double h = L / (N + 1);
    Eigen::VectorXd rho(N);
    Eigen::VectorXd V_theo(N);
    for (int i = 1; i <= N; ++i) {
      const double x = i * h;
      const double sx = std::sin(std::numbers::pi * x / L);
      V_theo(i - 1) = sx;
      rho(i - 1) = eps0 * (std::numbers::pi / L) * (std::numbers::pi / L) * sx;
    }
    DSTSolver1D solver(N, L, eps0);
    const Eigen::VectorXd V = solver.solve(rho);
    const double err = (V - V_theo).cwiseAbs().maxCoeff();
    if (err_prev > 0.0) REQUIRE(err < 0.35 * err_prev);  // ~1/4 per halving
    err_prev = err;
  }
}
