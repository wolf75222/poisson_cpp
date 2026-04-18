#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/spectral/dst2d.hpp"

using poisson::spectral::DSTSolver2D;

TEST_CASE("DSTSolver2D: recovers a 2D discrete eigenmode to machine precision",
          "[spectral][2d]") {
  const int Nx = 32, Ny = 32;
  const double Lx = 1.0, Ly = 1.0, eps0 = 1.0;
  const int k = 1, l = 1;
  const double hx = Lx / (Nx + 1);
  const double hy = Ly / (Ny + 1);

  Eigen::MatrixXd V_theo(Nx, Ny);
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      V_theo(i - 1, j - 1) =
          std::sin(k * i * std::numbers::pi / (Nx + 1))
          * std::sin(l * j * std::numbers::pi / (Ny + 1));
    }
  }

  const double sx = std::sin(k * std::numbers::pi / (2.0 * (Nx + 1)));
  const double sy = std::sin(l * std::numbers::pi / (2.0 * (Ny + 1)));
  const double lam_kl = 4.0 / (hx * hx) * sx * sx + 4.0 / (hy * hy) * sy * sy;
  const Eigen::MatrixXd rho = eps0 * lam_kl * V_theo;

  DSTSolver2D solver(Nx, Ny, Lx, Ly, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);
  REQUIRE((V - V_theo).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("DSTSolver2D: continuous mode gives O(h^2) error", "[spectral][2d]") {
  // V = sin(pi x/Lx) sin(pi y/Ly), rho = eps0 ((pi/Lx)^2 + (pi/Ly)^2) V.
  const double Lx = 1.0, Ly = 1.0, eps0 = 1.0;
  double err_prev = 0.0;
  for (int N : {31, 63, 127}) {
    const double hx = Lx / (N + 1), hy = Ly / (N + 1);
    Eigen::MatrixXd rho(N, N);
    Eigen::MatrixXd V_theo(N, N);
    const double a = std::numbers::pi / Lx;
    const double b = std::numbers::pi / Ly;
    for (int j = 1; j <= N; ++j) {
      for (int i = 1; i <= N; ++i) {
        const double x = i * hx, y = j * hy;
        const double v = std::sin(a * x) * std::sin(b * y);
        V_theo(i - 1, j - 1) = v;
        rho(i - 1, j - 1) = eps0 * (a * a + b * b) * v;
      }
    }
    DSTSolver2D solver(N, N, Lx, Ly, eps0);
    const Eigen::MatrixXd V = solver.solve(rho);
    const double err = (V - V_theo).cwiseAbs().maxCoeff();
    if (err_prev > 0.0) REQUIRE(err < 0.35 * err_prev);
    err_prev = err;
  }
}
