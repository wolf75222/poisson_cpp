#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"

using poisson::Grid1D;
using poisson::fv::solve_poisson_1d;

TEST_CASE("solve_poisson_1d: zero charge gives affine potential", "[fv][1d]") {
  const int N = 50;
  const double L = 1.0, uL = 10.0, uR = 0.0;
  Grid1D grid(L, N);
  Eigen::VectorXd rho = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd V = solve_poisson_1d(rho, uL, uR, grid);

  for (int i = 0; i < N; ++i) {
    const double x = grid.x(i);
    const double V_theo = uL + (uR - uL) * x / L;
    REQUIRE(std::abs(V(i) - V_theo) < 1e-12);
  }
}

TEST_CASE("solve_poisson_1d: uniform charge gives parabolic potential",
          "[fv][1d]") {
  const int N = 50;
  const double L = 1.0, uL = 10.0, uR = 0.0, eps0 = 1.0;
  const double rho_val = 100.0;
  Grid1D grid(L, N);
  Eigen::VectorXd rho = Eigen::VectorXd::Constant(N, rho_val);
  Eigen::VectorXd V = solve_poisson_1d(rho, uL, uR, grid, eps0);

  // Analytical: V(x) = -rho/(2 eps0) x^2 + B x + uL with V(L) = uR.
  const double B = (uR - uL + rho_val * L * L / (2.0 * eps0)) / L;
  for (int i = 0; i < N; ++i) {
    const double x = grid.x(i);
    const double V_theo = -rho_val * x * x / (2.0 * eps0) + B * x + uL;
    REQUIRE(std::abs(V(i) - V_theo) < 1e-12);
  }
}

TEST_CASE("solve_poisson_1d: manufactured solution converges as O(h^2)",
          "[fv][1d]") {
  // phi(x) = sin(pi x / L) + A sin(k pi x / L), with Dirichlet 0 on both sides.
  const double L = 1.0, eps0 = 1.0, A = 5.0;
  const int k = 2;
  auto phi_theo = [&](double x) {
    return std::sin(std::numbers::pi * x / L)
         + A * std::sin(k * std::numbers::pi * x / L);
  };
  auto rho_theo = [&](double x) {
    const double w1 = std::numbers::pi / L;
    const double w2 = k * std::numbers::pi / L;
    return eps0 * (w1 * w1 * std::sin(w1 * x) + w2 * w2 * A * std::sin(w2 * x));
  };

  double err_prev = 0.0;
  for (int N : {51, 101, 201, 401}) {
    Grid1D grid(L, N);
    Eigen::VectorXd rho(N);
    for (int i = 0; i < N; ++i) rho(i) = rho_theo(grid.x(i));
    Eigen::VectorXd V = solve_poisson_1d(rho, 0.0, 0.0, grid, eps0);

    double err = 0.0;
    for (int i = 0; i < N; ++i) {
      err = std::max(err, std::abs(V(i) - phi_theo(grid.x(i))));
    }
    // O(h^2): halving h (doubling N-1) should roughly quarter the error.
    if (err_prev > 0.0) {
      REQUIRE(err < 0.35 * err_prev);  // ~1/4 with slack
    }
    err_prev = err;
  }
}
