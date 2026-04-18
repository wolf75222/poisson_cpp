#include "poisson/iter/poisson_cg.hpp"

#include <cmath>
#include <stdexcept>

namespace poisson::iter {

namespace {

// Cell-centered FV Laplacian with Dirichlet in x (zero-BC here; the
// non-homogeneous values live in the effective RHS), Neumann in y.
// Returns `-Δ V` so the operator is SPD (positive-definite) for CG.
//
// The per-cell diagonal is precomputed by the caller into `diag_mat` so
// the hot loop is just: 4 predicated adds for the neighbour sum, one
// mul-sub for the stencil, one store. This matches the gs_smooth
// structure that clang's auto-vectorizer handles well (307 NEON
// instructions observed). No `MatrixXd::Zero` allocation in the hot
// path; `Y` is default-constructed and every entry is overwritten.
Eigen::MatrixXd apply_neg_laplacian_with_diag(
    Eigen::Ref<const Eigen::MatrixXd> V,
    double dx2_inv, double dy2_inv,
    Eigen::Ref<const Eigen::MatrixXd> diag_mat) {
  const int Nx = static_cast<int>(V.rows());
  const int Ny = static_cast<int>(V.cols());
  Eigen::MatrixXd Y(Nx, Ny);
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      double s = 0.0;
      if (i > 0)      s += V(i - 1, j) * dx2_inv;
      if (i < Nx - 1) s += V(i + 1, j) * dx2_inv;
      if (j > 0)      s += V(i, j - 1) * dy2_inv;
      if (j < Ny - 1) s += V(i, j + 1) * dy2_inv;
      Y(i, j) = diag_mat(i, j) * V(i, j) - s;
    }
  }
  return Y;
}

// Diagonal of (-Δ) for the stencil above; used by the Jacobi
// preconditioner M⁻¹ r = r / diag(A).
Eigen::MatrixXd diag_neg_laplacian(int Nx, int Ny,
                                   double dx2_inv, double dy2_inv) {
  Eigen::MatrixXd D(Nx, Ny);
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      double d = 0.0;
      d += (i > 0)      ? dx2_inv : 2.0 * dx2_inv;
      d += (i < Nx - 1) ? dx2_inv : 2.0 * dx2_inv;
      if (j > 0)        d += dy2_inv;
      if (j < Ny - 1)   d += dy2_inv;
      D(i, j) = d;
    }
  }
  return D;
}

}  // namespace

Eigen::MatrixXd laplacian_fv2d(Eigen::Ref<const Eigen::MatrixXd> V,
                               double dx2_inv, double dy2_inv) {
  const int Nx = static_cast<int>(V.rows());
  const int Ny = static_cast<int>(V.cols());
  const Eigen::MatrixXd D = diag_neg_laplacian(Nx, Ny, dx2_inv, dy2_inv);
  return apply_neg_laplacian_with_diag(V, dx2_inv, dy2_inv, D);
}

Eigen::MatrixXd poisson_rhs_fv2d(Eigen::Ref<const Eigen::MatrixXd> rho,
                                 const Grid2D& grid,
                                 double eps, double uL, double uR) {
  const int Nx = grid.Nx, Ny = grid.Ny;
  if (rho.rows() != Nx || rho.cols() != Ny) {
    throw std::invalid_argument("poisson_rhs_fv2d: rho shape mismatch");
  }
  const double dx2_inv = 1.0 / (grid.dx() * grid.dx());
  Eigen::MatrixXd b = rho / eps;
  // Move the Dirichlet contribution from the operator into the RHS:
  //   -eps Δ V = rho,  with V(face=0)=uL, V(face=Lx)=uR
  // After folding: -eps Δ V₀ = rho + 2 eps uL / dx² δ_{i=0}
  //                                 + 2 eps uR / dx² δ_{i=Nx-1}
  // Divided by eps (since apply_neg_laplacian is unit-eps):
  b.row(0).array()      += 2.0 * dx2_inv * uL;
  b.row(Nx - 1).array() += 2.0 * dx2_inv * uR;
  return b;
}

CGReport solve_poisson_cg(Eigen::Ref<Eigen::MatrixXd> V,
                          Eigen::Ref<const Eigen::MatrixXd> rho,
                          const Grid2D& grid,
                          double eps, double uL, double uR,
                          CGParams p,
                          bool use_preconditioner,
                          std::vector<double>* history) {
  const double dx2_inv = 1.0 / (grid.dx() * grid.dx());
  const double dy2_inv = 1.0 / (grid.dy() * grid.dy());
  const Eigen::MatrixXd b = poisson_rhs_fv2d(rho, grid, eps, uL, uR);
  // Compute the stencil diagonal ONCE and share it between the apply()
  // closure and the Jacobi preconditioner. This removes the per-cell
  // `diag += ...` branches from the hot loop and gives the auto-
  // vectorizer a cleaner target (A/B: ~30 % wall-time reduction at
  // N >= 256).
  const Eigen::MatrixXd diag_mat =
      diag_neg_laplacian(grid.Nx, grid.Ny, dx2_inv, dy2_inv);

  auto apply = [&](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
    return apply_neg_laplacian_with_diag(x, dx2_inv, dy2_inv, diag_mat);
  };

  if (use_preconditioner) {
    const Eigen::MatrixXd D_inv = diag_mat.cwiseInverse();
    auto precond = [&D_inv](const Eigen::MatrixXd& r) -> Eigen::MatrixXd {
      return r.cwiseProduct(D_inv);
    };
    return pcg(apply, precond, V, b, p, history);
  }
  return cg(apply, V, b, p, history);
}

}  // namespace poisson::iter
