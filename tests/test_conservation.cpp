// Physical conservation laws for electrostatics, as described in the course
// notes `CourseOnPoisson/Notes/poisson_equation.pdf` (Sections 2.3, 2.4,
// 3.2, 4.2, 4.3). These are identities that any physically-consistent
// Poisson solver must satisfy; they test the solver against the
// *mathematical structure* of the physics, not against external data.
//
//   1. Gauss's law (integral form):        ε₀ ∮ ∂V/∂n dℓ = Q_enclosed
//   2. Electrostatic energy identity:      ½ ∫ ρ V dA = ½ ε₀ ∫ |∇V|² dA
//   3. Green's first identity (Dirichlet): ε₀ ∫ |∇V|² dA = ∫ ρ V dA
//   4. Normal-D continuity at dielectric interface: ε₁ E₁·n = ε₂ E₂·n
//   5. Discrete flux conservation per cell (FV solver)

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/dielectric.hpp"
#include "poisson/fv/solver2d.hpp"

#ifdef POISSON_HAVE_FFTW3
#include "poisson/spectral/dst2d.hpp"
#endif

// ============================================================================
//  1. Gauss's law (integral form)
// ============================================================================

#ifdef POISSON_HAVE_FFTW3

TEST_CASE("[conservation] Gauss's law: ε₀ flux(∂Ω) = Q enclosed",
          "[conservation][spectral][2d]") {
  // With V = 0 on the full boundary ∂Ω, the divergence theorem applied to
  // -ε₀ ∇²V = ρ gives
  //     ε₀ ∮_∂Ω (-∇V)·n_out dℓ = ∫∫_Ω ρ dA   [= Q_enclosed]
  // i.e. the outward electric flux equals Q/ε₀. With the forward-difference
  // boundary estimate ∂V/∂n ≈ V_near/h, the discrete check converges at O(h).
  const int N = 511;   // fine enough that the O(h) flux estimate is accurate
  const double L = 1.0, eps0 = 1.3;
  const double h = L / (N + 1);

  // Gaussian blob well inside the box (σ/distance-to-wall = 6 → boundary
  // value of ρ is below 1e-7, so Q_enclosed ≈ Q_total to double precision).
  const double xc = 0.5, yc = 0.5, sigma = 0.08;
  Eigen::MatrixXd rho(N, N);
  for (int j = 1; j <= N; ++j) {
    const double y = j * h;
    for (int i = 1; i <= N; ++i) {
      const double x = i * h;
      const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
      rho(i - 1, j - 1) = std::exp(-r2 / (2.0 * sigma * sigma));
    }
  }

  poisson::spectral::DSTSolver2D solver(N, N, L, L, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);

  // Q_enclosed via midpoint rule on the grid.
  const double Q_enc = rho.sum() * h * h;

  // Outward flux: for V=0 at boundary, ∂V/∂n ≈ V_near_boundary / h, and
  // integrating over the edge of length L picks up a factor (h for dy) /
  // (h for the derivative) = 1. So flux through one edge = sum of V values
  // along the adjacent interior row.
  const double flux_W = V.row(0).sum();
  const double flux_E = V.row(N - 1).sum();
  const double flux_S = V.col(0).sum();
  const double flux_N = V.col(N - 1).sum();
  const double total_flux = flux_W + flux_E + flux_S + flux_N;

  const double lhs = eps0 * total_flux;
  const double rhs = Q_enc;
  const double rel_err = std::abs(lhs - rhs) / std::abs(rhs);
  // O(h) boundary estimate: at N=511 we expect ~5% error; at N=1023, ~2.5%.
  REQUIRE(rel_err < 0.06);
}

// ============================================================================
//  2. Electrostatic energy identity (two forms must agree exactly)
// ============================================================================

TEST_CASE("[conservation] Electrostatic energy: ½∫ρV = ½ε₀∫|∇V|²",
          "[conservation][spectral][2d]") {
  // For Dirichlet V = 0 on ∂Ω, integration by parts gives
  //     ∫ ρ V dA = ε₀ ∫ |∇V|² dA
  // This is the *discrete* analogue using the same 5-point stencil that
  // defines the solver, so it holds to machine precision.
  const int N = 127;
  const double L = 1.0, eps0 = 1.7;
  const double h = L / (N + 1);

  Eigen::MatrixXd rho(N, N);
  for (int j = 1; j <= N; ++j) {
    for (int i = 1; i <= N; ++i) {
      const double x = i * h, y = j * h;
      const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
      rho(i - 1, j - 1) = std::exp(-r2 / 0.01);
    }
  }

  poisson::spectral::DSTSolver2D solver(N, N, L, L, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);

  // W_interaction = ½ ∫ ρ V dA
  const double W_int = 0.5 * h * h * (rho.array() * V.array()).sum();

  // W_field = ½ ε₀ ∫ |∇V|² dA, via edge-wise (V_right - V_left)² summation.
  // In discrete form, |∇V|² dA = (ΔV / h)² × h² = (ΔV)², so no extra h factor.
  // Boundary edges include the ghost value V=0 at x=0, x=L, y=0, y=L.
  double E = 0.0;
  for (int j = 0; j < N; ++j) {
    E += V(0, j) * V(0, j);                // left boundary edge
    E += V(N - 1, j) * V(N - 1, j);        // right boundary edge
    for (int i = 0; i < N - 1; ++i) {
      const double d = V(i + 1, j) - V(i, j);
      E += d * d;
    }
  }
  for (int i = 0; i < N; ++i) {
    E += V(i, 0) * V(i, 0);                // bottom boundary edge
    E += V(i, N - 1) * V(i, N - 1);        // top boundary edge
    for (int j = 0; j < N - 1; ++j) {
      const double d = V(i, j + 1) - V(i, j);
      E += d * d;
    }
  }
  const double W_field = 0.5 * eps0 * E;

  const double rel_err = std::abs(W_int - W_field) / std::abs(W_int);
  // Exact discrete identity (summation by parts on the 5-point stencil):
  // any deviation is pure round-off accumulated on N² operations.
  REQUIRE(rel_err < 1e-12);

  // Sanity: W > 0 (physically, electrostatic energy is non-negative).
  REQUIRE(W_int > 0.0);
  REQUIRE(W_field > 0.0);
}

// ============================================================================
//  3. Green's first identity for Dirichlet BCs
// ============================================================================

TEST_CASE("[conservation] Green's first identity: ε₀ ∫|∇V|² = ∫ρV",
          "[conservation][spectral][2d]") {
  // From the notes Section 2.4: for V = 0 on ∂Ω,
  //   ∫ V Δ V dA = - ∫ |∇V|² dA   (integration by parts, boundary term vanishes)
  // Multiplying by -ε₀ and using -ε₀ Δ V = ρ:
  //   ε₀ ∫ |∇V|² dA = ∫ ρ V dA
  // Same *identity* as energy equivalence but without the ½; we verify both
  // sides independently here to double-check the solver's self-adjointness.
  const int N = 127;
  const double L = 1.0, eps0 = 2.3;
  const double h = L / (N + 1);
  Eigen::MatrixXd rho(N, N);
  for (int j = 1; j <= N; ++j)
    for (int i = 1; i <= N; ++i)
      rho(i - 1, j - 1) = std::sin(std::numbers::pi * i * h)
                         * std::sin(std::numbers::pi * j * h);

  poisson::spectral::DSTSolver2D solver(N, N, L, L, eps0);
  const Eigen::MatrixXd V = solver.solve(rho);

  // LHS: ε₀ ∫ |∇V|² dA
  double grad_sq = 0.0;
  for (int j = 0; j < N; ++j) {
    grad_sq += V(0, j) * V(0, j) + V(N - 1, j) * V(N - 1, j);
    for (int i = 0; i < N - 1; ++i) {
      const double d = V(i + 1, j) - V(i, j);
      grad_sq += d * d;
    }
  }
  for (int i = 0; i < N; ++i) {
    grad_sq += V(i, 0) * V(i, 0) + V(i, N - 1) * V(i, N - 1);
    for (int j = 0; j < N - 1; ++j) {
      const double d = V(i, j + 1) - V(i, j);
      grad_sq += d * d;
    }
  }
  const double lhs = eps0 * grad_sq;

  // RHS: ∫ ρ V dA
  const double rhs = h * h * (rho.array() * V.array()).sum();

  REQUIRE(std::abs(lhs - rhs) / std::abs(rhs) < 1e-12);
}

#endif   // POISSON_HAVE_FFTW3

// ============================================================================
//  4. Normal-D continuity at a dielectric interface (1D)
// ============================================================================

TEST_CASE("[conservation] 1D dielectric: D_n is continuous across interfaces",
          "[conservation][fv][1d]") {
  // With ρ = 0, Maxwell's equation ∇·D = 0 forces D = ε E to be piecewise
  // constant across layers. With Dirichlet BCs in 1D, the total charge
  // would need to be zero, so D is a single constant over the whole domain.
  // Any variation exceeding round-off signals lost conservation.
  using poisson::Grid1D;
  using poisson::fv::harmonic_mean;
  using poisson::fv::solve_poisson_1d;

  const int N = 200;
  const double L = 1.0, uL = 15.0, uR = 0.0, eps0 = 1.0;
  Grid1D grid(L, N);

  // A piecewise eps profile with 3 layers:
  //   [0, 0.3)   -> eps_r = 5
  //   [0.3, 0.7) -> eps_r = 1
  //   [0.7, 1.0] -> eps_r = 2
  Eigen::VectorXd eps_r(N);
  for (int i = 0; i < N; ++i) {
    const double x = grid.x(i);
    if      (x < 0.3) eps_r(i) = 5.0;
    else if (x < 0.7) eps_r(i) = 1.0;
    else              eps_r(i) = 2.0;
  }
  const Eigen::VectorXd rho = Eigen::VectorXd::Zero(N);
  const Eigen::VectorXd V = solve_poisson_1d(rho, eps_r, uL, uR, grid, eps0);

  // D at each face (N - 1 interior faces): D_face = ε_face (V_{i+1} - V_i)/dx
  const double dx = grid.dx();
  const Eigen::VectorXd eps_face =
      harmonic_mean(eps_r.segment(1, N - 1), eps_r.segment(0, N - 1));
  Eigen::VectorXd D(N - 1);
  for (int i = 0; i < N - 1; ++i) {
    D(i) = eps0 * eps_face(i) * (V(i + 1) - V(i)) / dx;
  }

  // D must be a single constant across every face, including the two
  // material interfaces at x ≈ 0.3 and x ≈ 0.7.
  const double D_mean = D.mean();
  const double D_var  = (D.maxCoeff() - D.minCoeff()) / std::abs(D_mean);
  REQUIRE(D_var < 1e-12);   // machine precision
}

// ============================================================================
//  5. Discrete flux conservation at each cell (FV Solver2D)
// ============================================================================

TEST_CASE("[conservation] Solver2D FV: per-cell flux balance to tolerance",
          "[conservation][fv][2d]") {
  // For the converged SOR iterate, the discrete divergence theorem
  //   Σ_faces F_face = h² ρ_cell
  // must hold cell-by-cell to within the SOR tolerance. Any residual above
  // this bound would mean the iteration stopped in an unconverged state.
  using poisson::Grid2D;
  using poisson::fv::Solver2D;

  const int N = 32;
  Grid2D grid(1.0, 1.0, N, N);
  Solver2D solver(grid, /*eps=*/1.0, /*uL=*/0.0, /*uR=*/0.0);

  // Two Gaussian blobs of opposite sign (electric dipole-like source).
  Eigen::MatrixXd rho(N, N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const double x = (i + 0.5) * grid.dx();
      const double y = (j + 0.5) * grid.dy();
      rho(i, j) =  std::exp(-((x - 0.3) * (x - 0.3)
                              + (y - 0.5) * (y - 0.5)) / 0.01)
                  - std::exp(-((x - 0.7) * (x - 0.7)
                               + (y - 0.5) * (y - 0.5)) / 0.01);
    }
  }
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  const auto report = solver.solve(V, rho, {.tol = 1e-12, .max_iter = 50000});
  REQUIRE(report.residual < 1e-10);

  // Verify flux balance on a few randomly-picked interior cells.
  const double dx = grid.dx(), dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  (void)inv_dx2; (void)inv_dy2;   // (not directly needed below)

  double worst_residual = 0.0;
  for (int j = 1; j < N - 1; ++j) {
    for (int i = 1; i < N - 1; ++i) {
      // Discrete divergence-theorem form:
      //   (V(i+1) - V(i)) - (V(i) - V(i-1)) all over dx² gives -∂²V/∂x²
      const double lap =
          (V(i + 1, j) + V(i - 1, j) - 2.0 * V(i, j)) / (dx * dx) +
          (V(i, j + 1) + V(i, j - 1) - 2.0 * V(i, j)) / (dy * dy);
      const double local_residual = std::abs(-lap - rho(i, j));
      if (local_residual > worst_residual) worst_residual = local_residual;
    }
  }
  // After converged SOR to 1e-12 iteration tolerance, each cell's flux
  // balance holds to essentially the same precision.
  REQUIRE(worst_residual < 1e-7);
}
