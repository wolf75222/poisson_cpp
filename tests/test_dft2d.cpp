#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/spectral/dft2d.hpp"

using poisson::spectral::DFTSolver2D;

TEST_CASE("DFTSolver2D: recovers a 2D periodic eigenmode to machine precision",
          "[spectral][2d][periodic]") {
  // The periodic discrete Laplacian eigenvector at wavenumber (k, l):
  //   V(i, j) = sin(2π k i / Nx) sin(2π l j / Ny)
  // has eigenvalue λ_{k,l} = -(4/hx²) sin²(π k / Nx) - (4/hy²) sin²(π l / Ny)
  // (negative). Solving eps0 ∇²V = -ρ ⇒ ρ = -eps0 λ V = +eps0 |λ| V.
  // We then expect solver.solve(ρ) = V (mean-zero ⇒ no pinning needed).
  const int Nx = 32, Ny = 32;
  const double Lx = 1.0, Ly = 1.0, eps0 = 1.0;
  const int k = 2, l = 3;
  const double hx = Lx / Nx, hy = Ly / Ny;
  const double sx = std::sin(std::numbers::pi * k / Nx);
  const double sy = std::sin(std::numbers::pi * l / Ny);
  const double abs_lam = 4.0 / (hx * hx) * sx * sx +
                          4.0 / (hy * hy) * sy * sy;

  Eigen::MatrixXd V_theo(Nx, Ny);
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      V_theo(i, j) = std::sin(2.0 * std::numbers::pi * k * i / Nx) *
                      std::sin(2.0 * std::numbers::pi * l * j / Ny);
    }
  }
  // V_theo has zero mean (sin integrates to 0 over the full period).
  REQUIRE(std::abs(V_theo.mean()) < 1e-12);

  const Eigen::MatrixXd rho = eps0 * abs_lam * V_theo;
  DFTSolver2D solver(Nx, Ny, Lx, Ly, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);
  REQUIRE((V - V_theo).cwiseAbs().maxCoeff() < 1e-11);
}

TEST_CASE("DFTSolver2D: continuous mode, periodic, O(h²) error",
          "[spectral][2d][periodic][convergence]") {
  // V(x, y) = sin(2π x / Lx) sin(2π y / Ly) is exactly periodic on
  // [0, Lx) × [0, Ly). Continuous Laplacian:
  //   ∇²V = -((2π/Lx)² + (2π/Ly)²) V
  // ⇒ ρ = -eps0 ∇²V = +eps0 ((2π/Lx)² + (2π/Ly)²) V
  // The discrete solver should converge to the continuous V at 2nd
  // order in h (5-point stencil truncation).
  const double Lx = 1.0, Ly = 1.0, eps0 = 1.0;
  const double kx = 2.0 * std::numbers::pi / Lx;
  const double ky = 2.0 * std::numbers::pi / Ly;
  double err_prev = 0.0;
  for (int N : {32, 64, 128}) {
    const double hx = Lx / N, hy = Ly / N;
    Eigen::MatrixXd rho(N, N), V_theo(N, N);
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        const double x = (i + 0.5) * hx;   // cell-centred sampling
        const double y = (j + 0.5) * hy;
        V_theo(i, j) = std::sin(kx * x) * std::sin(ky * y);
        rho(i, j)    = eps0 * (kx * kx + ky * ky) * V_theo(i, j);
      }
    }
    DFTSolver2D solver(N, N, Lx, Ly, eps0);
    const Eigen::MatrixXd V = solver.solve(rho);
    const double err = (V - V_theo).cwiseAbs().maxCoeff();
    if (err_prev > 0.0) {
      // Halving N should drop the error by ~4 (2nd order). Allow some
      // slack but require at least 3x improvement.
      REQUIRE(err_prev / err > 3.0);
    }
    err_prev = err;
  }
}

TEST_CASE("DFTSolver2D: input with non-zero mean is handled (mean subtracted)",
          "[spectral][2d][periodic][mean]") {
  // Periodic Poisson has a constant null mode. If the input doesn't
  // have zero mean, no exact solution exists ; the conventional fix is
  // to subtract the mean and pin V(0,0)=0. Verify the solver does
  // exactly that : solving for ρ = const must produce V = 0 (the mean
  // is the only thing that survives, and the solver pins it to 0).
  const int Nx = 16, Ny = 16;
  Eigen::MatrixXd rho = Eigen::MatrixXd::Constant(Nx, Ny, 1.5);
  DFTSolver2D solver(Nx, Ny, 1.0, 1.0, 1.0);
  const Eigen::MatrixXd V = solver.solve(rho);
  REQUIRE(V.cwiseAbs().maxCoeff() < 1e-12);
}
