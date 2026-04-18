#include "poisson/fv/solver2d.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson::fv {

namespace {

// Element-wise harmonic mean of two arrays: 2 a b / (a + b).
Eigen::MatrixXd harmonic_mean_mat(const Eigen::MatrixXd& a,
                                  const Eigen::MatrixXd& b) {
  return (2.0 * a.array() * b.array()) / (a.array() + b.array());
}

}  // namespace

Solver2D::Solver2D(const Grid2D& grid,
                   Eigen::MatrixXd eps,
                   double uL,
                   double uR)
    : grid_(grid), uL_(uL), uR_(uR) {
  if (eps.rows() != grid.Nx || eps.cols() != grid.Ny) {
    throw std::invalid_argument("Solver2D: eps shape must be (Nx, Ny)");
  }
  if ((eps.array() <= 0.0).any()) {
    throw std::invalid_argument("Solver2D: eps must be strictly positive");
  }

  const int Nx = grid.Nx, Ny = grid.Ny;
  const double dx2 = grid.dx() * grid.dx();
  const double dy2 = grid.dy() * grid.dy();

  // Face permittivities (harmonic mean), shapes (Nx-1, Ny) and (Nx, Ny-1).
  const Eigen::MatrixXd eps_face_x = harmonic_mean_mat(
      eps.bottomRows(Nx - 1), eps.topRows(Nx - 1));
  const Eigen::MatrixXd eps_face_y = harmonic_mean_mat(
      eps.rightCols(Ny - 1), eps.leftCols(Ny - 1));

  Ve_.setZero(Nx, Ny);
  Vw_.setZero(Nx, Ny);
  Vn_.setZero(Nx, Ny);
  Vs_.setZero(Nx, Ny);

  Ve_.topRows(Nx - 1)    = eps_face_x / dx2;
  Vw_.bottomRows(Nx - 1) = eps_face_x / dx2;
  Vn_.leftCols(Ny - 1)   = eps_face_y / dy2;
  Vs_.rightCols(Ny - 1)  = eps_face_y / dy2;

  // Dirichlet boundaries in x: electrode on the face, distance to center h/2
  // → coefficient 2 eps / dx^2 (standard ghost-cell form).
  Vw_.row(0)      = 2.0 * eps.row(0) / dx2;
  Ve_.row(Nx - 1) = 2.0 * eps.row(Nx - 1) / dx2;
  // Neumann in y: Vn on the last row and Vs on the first are kept at zero.

  Vc_ = Ve_ + Vw_ + Vn_ + Vs_;
}

Solver2D::Solver2D(const Grid2D& grid, double eps, double uL, double uR)
    : Solver2D(grid, Eigen::MatrixXd::Constant(grid.Nx, grid.Ny, eps), uL, uR) {
}

Solver2D::Report Solver2D::solve(Eigen::Ref<Eigen::MatrixXd> V,
                                 Eigen::Ref<const Eigen::MatrixXd> rho,
                                 Params p) const {
  const int Nx = grid_.Nx, Ny = grid_.Ny;
  if (V.rows() != Nx || V.cols() != Ny) {
    throw std::invalid_argument("Solver2D::solve: V shape must be (Nx, Ny)");
  }
  if (rho.rows() != Nx || rho.cols() != Ny) {
    throw std::invalid_argument("Solver2D::solve: rho shape must be (Nx, Ny)");
  }

  double w = p.omega;
  if (w <= 0.0) {
    w = 2.0 / (1.0 + std::sin(std::numbers::pi / std::max(Nx, Ny)));
  }

  // Scratch buffer for the neighbour-sum across the whole grid, reused
  // across iterations to avoid allocations in the hot loop.
  Eigen::MatrixXd neighbour_sum(Nx, Ny);

  double max_diff = 0.0;
  int iter = 0;
  for (iter = 0; iter < p.max_iter; ++iter) {
    max_diff = 0.0;

    for (int color = 0; color < 2; ++color) {
      // Fully vectorised neighbour sum including ghost-cell Dirichlet terms.
      neighbour_sum.setZero();
      neighbour_sum.bottomRows(Nx - 1).array() +=
          Vw_.bottomRows(Nx - 1).array() * V.topRows(Nx - 1).array();
      neighbour_sum.topRows(Nx - 1).array() +=
          Ve_.topRows(Nx - 1).array() * V.bottomRows(Nx - 1).array();
      neighbour_sum.rightCols(Ny - 1).array() +=
          Vs_.rightCols(Ny - 1).array() * V.leftCols(Ny - 1).array();
      neighbour_sum.leftCols(Ny - 1).array() +=
          Vn_.leftCols(Ny - 1).array() * V.rightCols(Ny - 1).array();
      neighbour_sum.row(0).array()      += Vw_.row(0).array()      * uL_;
      neighbour_sum.row(Nx - 1).array() += Ve_.row(Nx - 1).array() * uR_;

      // Update cells of the current color. Eigen stores matrices in
      // column-major order, so we iterate with j outer and i inner for
      // cache locality. The inner loop has no colour branch: we step i
      // by 2 starting at the right offset for column j.
      const double one_minus_w = 1.0 - w;
      for (int j = 0; j < Ny; ++j) {
        const int istart = (j + color) & 1;
        for (int i = istart; i < Nx; i += 2) {
          const double V_gs = (neighbour_sum(i, j) + rho(i, j)) / Vc_(i, j);
          const double V_new = one_minus_w * V(i, j) + w * V_gs;
          const double diff = std::abs(V_new - V(i, j));
          if (diff > max_diff) max_diff = diff;
          V(i, j) = V_new;
        }
      }
    }

    if (max_diff < p.tol) {
      ++iter;
      break;
    }
  }

  return {iter, max_diff};
}

}  // namespace poisson::fv
