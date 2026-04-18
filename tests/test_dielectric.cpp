#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/dielectric.hpp"

using poisson::Grid1D;
using poisson::fv::harmonic_mean;
using poisson::fv::solve_poisson_1d;

TEST_CASE("harmonic_mean computes 2ab/(a+b) element-wise", "[dielectric]") {
  Eigen::VectorXd a(3); a << 1.0, 2.0, 5.0;
  Eigen::VectorXd b(3); b << 3.0, 2.0, 5.0;
  const Eigen::VectorXd h = harmonic_mean(a, b);
  REQUIRE(std::abs(h(0) - 2.0 * 1.0 * 3.0 / 4.0) < 1e-14);
  REQUIRE(std::abs(h(1) - 2.0) < 1e-14);  // both equal to 2
  REQUIRE(std::abs(h(2) - 5.0) < 1e-14);
}

TEST_CASE("solve_poisson_1d (dielectric) preserves D across interfaces",
          "[dielectric]") {
  // Setup: layers eps_r = 5 near boundaries, 1 in the middle, rho = 0.
  // Without charge, eps_r dV/dx must be constant across the whole domain.
  const int N = 50;
  const double L = 1.0, uL = 10.0, uR = 0.0, eps0 = 1.0;
  Grid1D grid(L, N);

  const int n_diel = 5;
  Eigen::VectorXd eps_r = Eigen::VectorXd::Ones(N);
  eps_r.head(n_diel).setConstant(5.0);
  eps_r.tail(n_diel).setConstant(5.0);

  Eigen::VectorXd rho = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd V = solve_poisson_1d(rho, eps_r, uL, uR, grid, eps0);

  const double dx = grid.dx();
  // D_face = eps_face * (V_{i+1} - V_i) / dx, with eps_face = harmonic mean.
  const Eigen::VectorXd eps_face =
      harmonic_mean(eps_r.segment(1, N - 1), eps_r.segment(0, N - 1));
  Eigen::VectorXd D(N - 1);
  for (int i = 0; i < N - 1; ++i) {
    D(i) = eps0 * eps_face(i) * (V(i + 1) - V(i)) / dx;
  }
  const double rel = (D.maxCoeff() - D.minCoeff()) / std::abs(D.mean());
  REQUIRE(rel < 1e-12);  // conservation to machine precision
}

TEST_CASE("solve_poisson_1d (dielectric) reduces to uniform case when eps_r = 1",
          "[dielectric]") {
  const int N = 40;
  Grid1D grid(1.0, N);
  Eigen::VectorXd eps_r = Eigen::VectorXd::Ones(N);
  Eigen::VectorXd rho = Eigen::VectorXd::Constant(N, 3.14);

  Eigen::VectorXd V_diel = solve_poisson_1d(rho, eps_r, 2.0, -1.0, grid, 1.0);

  // Analytical: V(x) = -rho/(2 eps0) x^2 + B x + uL, with uL=2, uR=-1, L=1.
  // B = uR - uL + rho * L^2 / (2 eps0) = -3 + 1.57 = -1.43
  const double rho_val = 3.14, L = 1.0, uL = 2.0, uR = -1.0;
  const double B = (uR - uL + rho_val * L * L / 2.0) / L;
  for (int i = 0; i < N; ++i) {
    const double x = grid.x(i);
    const double V_theo = -rho_val * x * x / 2.0 + B * x + uL;
    REQUIRE(std::abs(V_diel(i) - V_theo) < 1e-12);
  }
}
