#pragma once

#include <optional>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"

namespace poisson::fv {

/// Finite-volume 2D Poisson solver with SOR red-black relaxation.
///
/// Discretization: cell-centered finite volumes on a uniform Nx x Ny mesh.
/// Boundary conditions: Dirichlet in x (V = uL at i = 0 face, V = uR at
/// i = Nx face), homogeneous Neumann in y. Permittivity can be spatially
/// variable; face values use the harmonic mean of the two adjacent cells.
///
/// Stencil coefficients (Ve, Vw, Vn, Vs and Vc = Ve + Vw + Vn + Vs) are
/// precomputed at construction, so repeated `solve()` calls with different
/// right-hand sides are cheap.
///
/// Port of `sor_2d` from `CourseOnPoisson/notebooks/TP3_Poisson_2D.ipynb`.
struct SORParams {
  /// Over-relaxation factor. If <= 0, the optimal
  /// omega_opt = 2 / (1 + sin(pi / max(Nx, Ny))) is used automatically.
  double omega = -1.0;
  double tol = 1e-8;
  int max_iter = 20'000;
};

struct SORReport {
  int iterations;
  double residual;   ///< max |V^{k+1} - V^k|_inf at the last iteration.
};

class Solver2D {
 public:
  using Params = SORParams;
  using Report = SORReport;

  /// Build a solver with spatially variable permittivity `eps`
  /// (absolute, i.e. eps0 * eps_r).
  Solver2D(const Grid2D& grid,
           Eigen::MatrixXd eps,
           double uL,
           double uR);

  /// Build a solver with constant permittivity `eps` (default 1.0).
  Solver2D(const Grid2D& grid, double eps, double uL, double uR);

  /// Run SOR starting from the initial guess in `V` (modified in place).
  ///
  /// \param V   initial guess, shape (Nx, Ny). Overwritten with the solution.
  /// \param rho right-hand side, shape (Nx, Ny).
  /// \param p   convergence parameters.
  Report solve(Eigen::Ref<Eigen::MatrixXd> V,
               Eigen::Ref<const Eigen::MatrixXd> rho,
               Params p = {}) const;

  const Grid2D& grid() const noexcept { return grid_; }
  double uL() const noexcept { return uL_; }
  double uR() const noexcept { return uR_; }

 private:
  Grid2D grid_;
  double uL_, uR_;
  /// Per-cell stencil coefficients (shape Nx x Ny). Boundary cells already
  /// include the Dirichlet ghost-cell contribution (Vw(0,:) = 2 eps(0,:)/dx²
  /// and Ve(Nx-1,:) = 2 eps(Nx-1,:)/dx²).
  Eigen::MatrixXd Ve_, Vw_, Vn_, Vs_, Vc_;
  Eigen::MatrixXd Vc_inv_;   ///< 1/Vc_, precomputed to avoid hot-loop divisions
};

}  // namespace poisson::fv
