// Physical / mathematical invariants that MUST hold for a correct Poisson
// solver, independent of any reference data. Each test is either
//   (a) polynomial exactness: for a polynomial V of degree <= 2 in each
//       variable, the 5-point Laplacian is a perfect approximation of
//       d^2/dx^2 (the leading truncation term h^2 V''''/12 vanishes), so
//       the discrete solver must reproduce V at the grid nodes to
//       machine precision;
//   (b) linearity / superposition: A is a linear operator;
//   (c) reflection symmetry: a symmetric source must produce a symmetric
//       potential (otherwise the solver has a normalisation bug);
//   (d) Green's-function reciprocity: V at point B from a source at A
//       equals V at point A from a source at B (exchange of roles).
//
// These identities are true BY CONSTRUCTION — once written they do not need
// to be recomputed against any external oracle. They are our "bank of
// verified facts" about the solvers.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <random>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/linalg/thomas.hpp"

#ifdef POISSON_HAVE_FFTW3
#include "poisson/spectral/dst1d.hpp"
#include "poisson/spectral/dst2d.hpp"
#endif

using poisson::fv::Solver2D;
using poisson::Grid2D;

// ============================================================================
//  A. Polynomial exactness (DST family)
// ============================================================================

#ifdef POISSON_HAVE_FFTW3

TEST_CASE("[invariant] DSTSolver1D: exact on V(x) = x(L - x)",
          "[invariant][spectral][1d]") {
  // V''(x) = -2  =>  rho(x) = -eps0 * V''(x) = 2 eps0 (constant).
  // V vanishes at x = 0 and x = L, matching the DST-I BCs.
  const int N = 63;
  const double L = 1.0, eps0 = 1.3;
  const double h = L / (N + 1);

  const Eigen::VectorXd rho = Eigen::VectorXd::Constant(N, 2.0 * eps0);
  Eigen::VectorXd V_exact(N);
  for (int i = 1; i <= N; ++i) {
    const double x = i * h;
    V_exact(i - 1) = x * (L - x);
  }

  poisson::spectral::DSTSolver1D solver(N, L, eps0);
  const Eigen::VectorXd V = solver.solve(rho);
  REQUIRE((V - V_exact).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("[invariant] DSTSolver2D: exact on V(x,y) = x(Lx-x) y(Ly-y)",
          "[invariant][spectral][2d]") {
  const int N = 47;
  const double L = 1.0, eps0 = 1.0;
  const double h = L / (N + 1);

  Eigen::MatrixXd rho(N, N);
  Eigen::MatrixXd V_exact(N, N);
  for (int j = 1; j <= N; ++j) {
    const double y = j * h;
    for (int i = 1; i <= N; ++i) {
      const double x = i * h;
      // V(x,y) = x(L-x) y(L-y), with -Laplacian V = 2[y(L-y) + x(L-x)]
      rho(i - 1, j - 1) = 2.0 * eps0 * (y * (L - y) + x * (L - x));
      V_exact(i - 1, j - 1) = x * (L - x) * y * (L - y);
    }
  }

  poisson::spectral::DSTSolver2D solver(N, N, L, L, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);
  REQUIRE((V - V_exact).cwiseAbs().maxCoeff() < 1e-13);
}

TEST_CASE("[invariant] DSTSolver2D: superposition principle",
          "[invariant][spectral][2d]") {
  const int N = 31;
  const double L = 1.0;
  const double h = L / (N + 1);
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> U(-1.0, 1.0);

  Eigen::MatrixXd rho1(N, N), rho2(N, N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      rho1(i, j) = U(rng);
      rho2(i, j) = U(rng);
    }
  }
  const double alpha = 2.718, beta = -1.414;

  poisson::spectral::DSTSolver2D solver(N, N, L, L);
  const Eigen::MatrixXd V1 = solver.solve(rho1);
  const Eigen::MatrixXd V2 = solver.solve(rho2);
  const Eigen::MatrixXd V_combined =
      solver.solve(alpha * rho1 + beta * rho2);
  const Eigen::MatrixXd V_expected = alpha * V1 + beta * V2;
  (void)h;
  REQUIRE((V_combined - V_expected).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("[invariant] DSTSolver2D: zero input -> zero output",
          "[invariant][spectral][2d]") {
  const int N = 31;
  poisson::spectral::DSTSolver2D solver(N, N, 1.0, 1.0);
  const Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  const Eigen::MatrixXd V = solver.solve(rho);
  REQUIRE(V.cwiseAbs().maxCoeff() == 0.0);   // exact, not approximate
}

TEST_CASE("[invariant] DSTSolver2D: reflection symmetry preserved",
          "[invariant][spectral][2d]") {
  // With a source that is symmetric under (x, y) -> (Lx - x, y) AND under
  // (x, y) -> (x, Ly - y), the solution must share both symmetries.
  const int N = 33;
  const double L = 1.0;
  const double h = L / (N + 1);

  Eigen::MatrixXd rho(N, N);
  for (int j = 1; j <= N; ++j) {
    const double y = j * h;
    for (int i = 1; i <= N; ++i) {
      const double x = i * h;
      rho(i - 1, j - 1) = std::exp(
          -((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.01);
    }
  }

  poisson::spectral::DSTSolver2D solver(N, N, L, L);
  const Eigen::MatrixXd V = solver.solve(rho);

  // V(i, j) = V(N-1-i, j)  [reflection in x]
  // V(i, j) = V(i, N-1-j)  [reflection in y]
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      REQUIRE(std::abs(V(i, j) - V(N - 1 - i, j)) < 1e-13);
      REQUIRE(std::abs(V(i, j) - V(i, N - 1 - j)) < 1e-13);
    }
  }
}

TEST_CASE("[invariant] DSTSolver2D: Green's-function reciprocity G(A,B) = G(B,A)",
          "[invariant][spectral][2d]") {
  // Approximate point sources with narrow Gaussians. For a linear, self-adjoint
  // operator (Laplacian + Dirichlet BCs), the Green's function is symmetric:
  // the potential at B due to a unit source at A equals the potential at A
  // due to a unit source at B. This is a cornerstone theorem of electrostatics.
  const int N = 127;
  const double L = 1.0;
  const double h = L / (N + 1);
  const double sigma = 0.03;

  auto gaussian = [&](int ic, int jc) {
    const double xc = ic * h, yc = jc * h;
    Eigen::MatrixXd rho(N, N);
    for (int j = 1; j <= N; ++j) {
      const double y = j * h;
      for (int i = 1; i <= N; ++i) {
        const double x = i * h;
        const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
        rho(i - 1, j - 1) = std::exp(-r2 / (2.0 * sigma * sigma));
      }
    }
    return rho;
  };

  // Two well-separated points, both in the interior.
  const int iA = 32, jA = 48;   // ≈ (0.25, 0.375)
  const int iB = 80, jB = 64;   // ≈ (0.625, 0.50)

  poisson::spectral::DSTSolver2D solver(N, N, L, L);
  const Eigen::MatrixXd V_from_A = solver.solve(gaussian(iA, jA));
  const Eigen::MatrixXd V_from_B = solver.solve(gaussian(iB, jB));

  const double V_at_B_from_A = V_from_A(iB - 1, jB - 1);
  const double V_at_A_from_B = V_from_B(iA - 1, jA - 1);

  // Reciprocity must hold to machine precision (same discrete operator).
  REQUIRE(std::abs(V_at_B_from_A - V_at_A_from_B)
          < 1e-13 * std::max(std::abs(V_at_B_from_A), 1.0));
}

#endif   // POISSON_HAVE_FFTW3

// ============================================================================
//  B. FV Solver2D invariants (cell-centered, Dirichlet in x, Neumann in y)
// ============================================================================

TEST_CASE("[invariant] Solver2D: zero input -> zero output",
          "[invariant][fv][2d]") {
  const int N = 16;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, 1.0, 0.0, 0.0);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  const Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  solver.solve(V, rho, {.tol = 1e-14, .max_iter = 1});
  // V stays zero under one iteration from zero initial guess.
  REQUIRE(V.cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("[invariant] Solver2D: linearity under source scaling",
          "[invariant][fv][2d]") {
  const int N = 24;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, 1.0, 0.0, 0.0);

  Eigen::MatrixXd rho(N, N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const double x = (i + 0.5) * grid.dx();
      const double y = (j + 0.5) * grid.dy();
      rho(i, j) = std::exp(
          -((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02);
    }
  }

  // A linear operator L satisfies L(alpha rho) = alpha L(rho). We run each
  // SOR to tight tolerance, then check V(alpha rho) == alpha V(rho).
  Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd Va = Eigen::MatrixXd::Zero(N, N);
  solver.solve(V1, rho,       {.tol = 1e-12, .max_iter = 30000});
  solver.solve(Va, 3.5 * rho, {.tol = 1e-12, .max_iter = 30000});
  REQUIRE((Va - 3.5 * V1).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("[invariant] Solver2D: reflection symmetry in y (Neumann axis)",
          "[invariant][fv][2d]") {
  // Source independent of y and centered in x => V must be perfectly
  // symmetric under y -> Ly - y (Neumann BCs preserve this).
  const int N = 24;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, 1.0, 0.0, 0.0);

  Eigen::MatrixXd rho(N, N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const double x = (i + 0.5) * grid.dx();
      // symmetric in x around 0.5
      rho(i, j) = std::sin(std::numbers::pi * x);
    }
  }
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  solver.solve(V, rho, {.tol = 1e-12, .max_iter = 30000});

  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      // symmetry in y
      REQUIRE(std::abs(V(i, j) - V(i, N - 1 - j)) < 1e-8);
      // symmetry in x (rho symmetric around x = 0.5)
      REQUIRE(std::abs(V(i, j) - V(N - 1 - i, j)) < 1e-8);
    }
  }
}

// ============================================================================
//  C. Thomas solver invariants
// ============================================================================

TEST_CASE("[invariant] Thomas: linearity under RHS combination",
          "[invariant][thomas]") {
  const int N = 200;
  Eigen::VectorXd a(N), b(N), c(N), d1(N), d2(N);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(0, 1);
  for (int i = 0; i < N; ++i) {
    a(i) = U(rng);
    b(i) = U(rng) + 3.0;   // diagonally dominant
    c(i) = U(rng);
    d1(i) = U(rng);
    d2(i) = U(rng);
  }

  const double alpha = 2.5, beta = -1.7;
  const Eigen::VectorXd x1 = poisson::linalg::thomas(a, b, c, d1);
  const Eigen::VectorXd x2 = poisson::linalg::thomas(a, b, c, d2);
  const Eigen::VectorXd x_combined =
      poisson::linalg::thomas(a, b, c, alpha * d1 + beta * d2);
  REQUIRE((x_combined - (alpha * x1 + beta * x2)).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("[invariant] Thomas: zero RHS -> zero solution",
          "[invariant][thomas]") {
  const int N = 50;
  Eigen::VectorXd a(N), b(N), c(N);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> U(0, 1);
  for (int i = 0; i < N; ++i) {
    a(i) = U(rng);
    b(i) = U(rng) + 3.0;
    c(i) = U(rng);
  }
  const Eigen::VectorXd d = Eigen::VectorXd::Zero(N);
  const Eigen::VectorXd x = poisson::linalg::thomas(a, b, c, d);
  REQUIRE(x.cwiseAbs().maxCoeff() == 0.0);
}
