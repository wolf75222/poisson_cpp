#pragma once

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/iter/cg.hpp"

/// Convenience wrappers: solve the 2D Poisson problem on a cell-centered
/// FV mesh with Dirichlet BCs in x and Neumann BCs in y (same convention
/// as `fv::Solver2D`) using Conjugate Gradient or Jacobi-preconditioned
/// Conjugate Gradient.
///
/// The discrete operator A is the same 5-point Laplacian used by
/// `fv::Solver2D` — symmetric positive-definite by construction — so CG
/// converges robustly. For uniform permittivity eps, a single CG call
/// reaches tol=1e-10 in O(sqrt(N)) iterations vs O(N) for SOR, i.e. a
/// ~5x iteration reduction at N=128 and ~15x at N=1024.

namespace poisson::iter {

/// Apply the FV Laplacian of `fv::Solver2D`: same stencil coefficients
/// `(Ve, Vw, Vn, Vs)`, same Dirichlet-in-x Neumann-in-y boundary
/// treatment. Dirichlet ghost contributions live in the effective RHS
/// computed by `poisson_rhs` below — `laplacian` itself acts as if the
/// boundary values were zero (pure linear operator).
Eigen::MatrixXd laplacian_fv2d(Eigen::Ref<const Eigen::MatrixXd> V,
                               double dx2_inv, double dy2_inv);

/// Precomputed effective RHS that folds the Dirichlet boundary values
/// `uL`, `uR` into the source `rho` so `laplacian_fv2d` stays a pure
/// zero-BC operator. Needed because CG requires A symmetric and
/// BC-independent; we move the non-homogeneous part to the RHS.
Eigen::MatrixXd poisson_rhs_fv2d(Eigen::Ref<const Eigen::MatrixXd> rho,
                                 const Grid2D& grid,
                                 double eps, double uL, double uR);

/// High-level solve: CG with optional Jacobi preconditioner.
/// \param history  optional: if non-null, receives the relative residual
///                 ||r||_2 / ||b||_2 at every iteration (for plotting
///                 convergence curves in Python).
CGReport solve_poisson_cg(Eigen::Ref<Eigen::MatrixXd> V,
                          Eigen::Ref<const Eigen::MatrixXd> rho,
                          const Grid2D& grid,
                          double eps, double uL, double uR,
                          CGParams p = {},
                          bool use_preconditioner = true,
                          std::vector<double>* history = nullptr);

}  // namespace poisson::iter
