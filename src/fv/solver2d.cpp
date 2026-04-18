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
  Vc_inv_ = Vc_.cwiseInverse();
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

  // Fold the Dirichlet boundary contribution into an effective right-hand
  // side:  rhs_bc = rho + Vw(0,:) * uL * e_0 + Ve(Nx-1,:) * uR * e_{Nx-1}.
  // This lets the hot loop use unconditional predicated adds (like
  // `if (i > 0) s += Vw_(i,j) * V(i-1,j)`) instead of ternaries with
  // distinct operands. The auto-vectorizer accepts the first form — as
  // verified in mg::gs_smooth, which emits 307 NEON instructions —
  // and bails on the latter.
  Eigen::MatrixXd rhs_bc = rho;
  rhs_bc.row(0).array()      += Vw_.row(0).array()      * uL_;
  rhs_bc.row(Nx - 1).array() += Ve_.row(Nx - 1).array() * uR_;

  const double one_minus_w = 1.0 - w;
  double max_diff = 0.0;
  int iter = 0;

  // Separate serial and parallel sweep paths with a C++ `if`. A single
  // `#pragma omp parallel for if(N < T)` pays a small dispatch overhead
  // even when the runtime clause disables parallelism, so we duplicate the
  // inner loop and guard the whole parallel variant behind `#ifdef`.
  constexpr int kOmpThreshold = 384;   // empirical break-even on Apple M-series
  for (iter = 0; iter < p.max_iter; ++iter) {
    double md = 0.0;
    for (int color = 0; color < 2; ++color) {
      double color_max = 0.0;
#if defined(POISSON_HAVE_OPENMP)
      if (Ny >= kOmpThreshold) {
#       pragma omp parallel for reduction(max:color_max) schedule(static)
        for (int j = 0; j < Ny; ++j) {
          for (int i = (j + color) & 1; i < Nx; i += 2) {
            double s = 0.0;
            if (i > 0)      s += Vw_(i, j) * V(i - 1, j);
            if (i < Nx - 1) s += Ve_(i, j) * V(i + 1, j);
            if (j > 0)      s += Vs_(i, j) * V(i, j - 1);
            if (j < Ny - 1) s += Vn_(i, j) * V(i, j + 1);

            const double V_gs = (s + rhs_bc(i, j)) * Vc_inv_(i, j);
            const double V_i  = V(i, j);
            const double V_new = one_minus_w * V_i + w * V_gs;
            const double diff = std::abs(V_new - V_i);
            if (diff > color_max) color_max = diff;
            V(i, j) = V_new;
          }
        }
      } else
#endif
      {
        // Serial path. Duplicated by design so the compiler sees a single
        // "pure" loop with no OpenMP instrumentation at this call site.
        // `unroll_count(4)` reduces loop overhead (CS:APP loop-opt §2) and
        // `std::max` for the reduction emits a branchless `fmax` which
        // plays better with SIMD than the `if (diff > color_max)` pattern.
        for (int j = 0; j < Ny; ++j) {
#         if defined(__clang__)
#         pragma clang loop unroll_count(4)
#         endif
          for (int i = (j + color) & 1; i < Nx; i += 2) {
            double s = 0.0;
            if (i > 0)      s += Vw_(i, j) * V(i - 1, j);
            if (i < Nx - 1) s += Ve_(i, j) * V(i + 1, j);
            if (j > 0)      s += Vs_(i, j) * V(i, j - 1);
            if (j < Ny - 1) s += Vn_(i, j) * V(i, j + 1);

            const double V_gs = (s + rhs_bc(i, j)) * Vc_inv_(i, j);
            const double V_i  = V(i, j);
            const double V_new = one_minus_w * V_i + w * V_gs;
            const double diff = std::abs(V_new - V_i);
            color_max = std::max(color_max, diff);
            V(i, j) = V_new;
          }
        }
      }
      if (color_max > md) md = color_max;
    }

    max_diff = md;
    if (md < p.tol) { ++iter; break; }
  }

  return {iter, max_diff};
}

}  // namespace poisson::fv
