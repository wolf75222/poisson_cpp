#include "poisson/iter/poisson_cg.hpp"

#include <cmath>
#include <stdexcept>

namespace poisson::iter {

namespace {

// Cell-centered FV Laplacian with Dirichlet in x (zero-BC here; the
// non-homogeneous values live in the effective RHS), Neumann in y.
// Stencil coefficients mirror `fv::Solver2D` for eps ≡ 1:
//   L V = (V_{i+1,j} + V_{i-1,j} - 2 V_{i,j}) / dx²
//       + (V_{i,j+1} + V_{i,j-1} - 2 V_{i,j}) / dy²
// Boundary: Dirichlet ghost-cell coefficient 2/dx² at i=0 and i=Nx-1;
// Neumann: no ghost contribution at j=0 and j=Ny-1 (skips the missing
// neighbour).
//
// Sign convention: we return `-Δ V` so the resulting operator is SPD
// (positive-definite) as required by CG. The right-hand side is ρ/ε.
Eigen::MatrixXd apply_neg_laplacian(Eigen::Ref<const Eigen::MatrixXd> V,
                                    double dx2_inv, double dy2_inv) {
  const int Nx = static_cast<int>(V.rows());
  const int Ny = static_cast<int>(V.cols());
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(Nx, Ny);
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      double s = 0.0, diag = 0.0;
      // West / east in x (Dirichlet ghost: +2 V_i/dx² as if V_ghost=-V_i
      // around face value 0, which matches fv::Solver2D for zero uL/uR).
      if (i > 0)      { s += V(i - 1, j) * dx2_inv; diag += dx2_inv; }
      else             { diag += 2.0 * dx2_inv; }
      if (i < Nx - 1) { s += V(i + 1, j) * dx2_inv; diag += dx2_inv; }
      else             { diag += 2.0 * dx2_inv; }
      // South / north in y (Neumann: no ghost, just skip).
      if (j > 0)      { s += V(i, j - 1) * dy2_inv; diag += dy2_inv; }
      if (j < Ny - 1) { s += V(i, j + 1) * dy2_inv; diag += dy2_inv; }

      Y(i, j) = diag * V(i, j) - s;    // -Δ V
    }
  }
  return Y;
}

// Diagonal of (-Δ) for the stencil above — used by the Jacobi
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
  return apply_neg_laplacian(V, dx2_inv, dy2_inv);
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
                          bool use_preconditioner) {
  const double dx2_inv = 1.0 / (grid.dx() * grid.dx());
  const double dy2_inv = 1.0 / (grid.dy() * grid.dy());
  const Eigen::MatrixXd b = poisson_rhs_fv2d(rho, grid, eps, uL, uR);

  auto apply = [=](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
    return apply_neg_laplacian(x, dx2_inv, dy2_inv);
  };

  if (use_preconditioner) {
    const Eigen::MatrixXd D_inv =
        diag_neg_laplacian(grid.Nx, grid.Ny, dx2_inv, dy2_inv)
            .cwiseInverse();
    auto precond = [D_inv](const Eigen::MatrixXd& r) -> Eigen::MatrixXd {
      return r.cwiseProduct(D_inv);
    };
    return pcg(apply, precond, V, b, p);
  }
  return cg(apply, V, b, p);
}

}  // namespace poisson::iter
