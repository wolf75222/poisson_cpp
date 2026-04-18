#include "poisson/mg/vcycle.hpp"

#include <cmath>
#include <stdexcept>

namespace poisson::mg {

namespace {

// Per-cell diagonal of the FV cell-centered Laplacian with Dirichlet 0 on
// all four faces: 4 in the interior, 5 on edges, 6 in corners.
Eigen::MatrixXd diag_fv(int N) {
  Eigen::MatrixXd Vc = Eigen::MatrixXd::Constant(N, N, 4.0);
  Vc.row(0).array()      += 1.0;
  Vc.row(N - 1).array()  += 1.0;
  Vc.col(0).array()      += 1.0;
  Vc.col(N - 1).array()  += 1.0;
  return Vc;
}

// Sum over 4 neighbours. Neighbours that lie outside the grid contribute 0
// (consistent with the ghost-cell Dirichlet 0 convention).
Eigen::MatrixXd neighbour_sum(Eigen::Ref<const Eigen::MatrixXd> V) {
  const Eigen::Index N = V.rows();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(N, N);
  s.bottomRows(N - 1)     += V.topRows(N - 1);
  s.topRows(N - 1)        += V.bottomRows(N - 1);
  s.rightCols(N - 1)      += V.leftCols(N - 1);
  s.leftCols(N - 1)       += V.rightCols(N - 1);
  return s;
}

}  // namespace

void gs_smooth(Eigen::Ref<Eigen::MatrixXd> V,
               Eigen::Ref<const Eigen::MatrixXd> rho,
               double h,
               int n_iter) {
  if (V.rows() != V.cols())
    throw std::invalid_argument("gs_smooth: V must be square");
  if (rho.rows() != V.rows() || rho.cols() != V.cols())
    throw std::invalid_argument("gs_smooth: rho shape mismatch");

  const int N = static_cast<int>(V.rows());
  const Eigen::MatrixXd Vc_inv = diag_fv(N).cwiseInverse();
  const double h2 = h * h;

  // True in-place red-black Gauss-Seidel: sweep one color, using the most
  // recently updated values of the other color (no scratch buffer, half the
  // work compared to computing the full V_gs and discarding one color).
  // Boundary cells have 3 or 2 neighbours — handled with explicit bounds
  // checks; the branch predictor handles them well because the pattern is
  // the same for every inner iteration.
  for (int it = 0; it < n_iter; ++it) {
    for (int color = 0; color < 2; ++color) {
      for (int j = 0; j < N; ++j) {
        for (int i = (j + color) & 1; i < N; i += 2) {
          double s = 0.0;
          if (i > 0)     s += V(i - 1, j);
          if (i < N - 1) s += V(i + 1, j);
          if (j > 0)     s += V(i, j - 1);
          if (j < N - 1) s += V(i, j + 1);
          V(i, j) = (s + h2 * rho(i, j)) * Vc_inv(i, j);
        }
      }
    }
  }
}

Eigen::MatrixXd laplacian_fv(Eigen::Ref<const Eigen::MatrixXd> V, double h) {
  const int N = static_cast<int>(V.rows());
  const Eigen::MatrixXd Vc = diag_fv(N);
  return (Vc.array() * V.array() - neighbour_sum(V).array()) / (h * h);
}

Eigen::MatrixXd restrict_avg(Eigen::Ref<const Eigen::MatrixXd> r) {
  const Eigen::Index N = r.rows();
  if (r.cols() != N) throw std::invalid_argument("restrict: not square");
  if (N % 2 != 0)    throw std::invalid_argument("restrict: N must be even");
  const Eigen::Index M = N / 2;
  Eigen::MatrixXd c(M, M);
  for (Eigen::Index j = 0; j < M; ++j) {
    for (Eigen::Index i = 0; i < M; ++i) {
      c(i, j) = 0.25 *
                (r(2 * i,     2 * j) + r(2 * i + 1, 2 * j) +
                 r(2 * i, 2 * j + 1) + r(2 * i + 1, 2 * j + 1));
    }
  }
  return c;
}

Eigen::MatrixXd prolongate_const(Eigen::Ref<const Eigen::MatrixXd> c) {
  const Eigen::Index M = c.rows();
  const Eigen::Index N = 2 * M;
  Eigen::MatrixXd f(N, N);
  for (Eigen::Index j = 0; j < M; ++j) {
    for (Eigen::Index i = 0; i < M; ++i) {
      const double v = c(i, j);
      f(2 * i,     2 * j)     = v;
      f(2 * i + 1, 2 * j)     = v;
      f(2 * i,     2 * j + 1) = v;
      f(2 * i + 1, 2 * j + 1) = v;
    }
  }
  return f;
}

Eigen::MatrixXd prolongate_bilinear(Eigen::Ref<const Eigen::MatrixXd> c) {
  // Cell-centered convention: coarse cell (I, J) occupies the square
  // [I*2h, (I+1)*2h] and its center is at ((I+0.5)*2h, (J+0.5)*2h).
  // Fine cell (i, j) has center ((i+0.5)*h, (j+0.5)*h). The enclosing coarse
  // cell has (I, J) = (i/2, j/2), and the fine cell's offset inside it is
  // (i%2, j%2). A standard bilinear interpolation between the four coarse
  // cells around the fine center uses weights (9/16, 3/16, 3/16, 1/16).
  const Eigen::Index M = c.rows();
  const Eigen::Index N = 2 * M;
  auto at = [&](Eigen::Index I, Eigen::Index J) -> double {
    // Homogeneous Dirichlet 0 ghost for out-of-range coarse indices.
    if (I < 0 || I >= M || J < 0 || J >= M) return 0.0;
    return c(I, J);
  };
  Eigen::MatrixXd f(N, N);
  for (Eigen::Index j = 0; j < N; ++j) {
    const Eigen::Index J = j / 2;
    const Eigen::Index dJ = (j % 2 == 0) ? -1 : +1;
    for (Eigen::Index i = 0; i < N; ++i) {
      const Eigen::Index I = i / 2;
      const Eigen::Index dI = (i % 2 == 0) ? -1 : +1;
      f(i, j) =
          9.0 / 16.0 * at(I,      J)
        + 3.0 / 16.0 * at(I + dI, J)
        + 3.0 / 16.0 * at(I,      J + dJ)
        + 1.0 / 16.0 * at(I + dI, J + dJ);
    }
  }
  return f;
}

Eigen::MatrixXd vcycle_uniform(Eigen::MatrixXd V,
                               const Eigen::MatrixXd& rho,
                               double h,
                               int n_pre,
                               int n_post,
                               int n_min) {
  gs_smooth(V, rho, h, n_pre);
  if (V.rows() <= n_min) {
    gs_smooth(V, rho, h, 50);        // "coarse enough" solve
    return V;
  }
  const Eigen::MatrixXd r = rho - laplacian_fv(V, h);
  const Eigen::MatrixXd r_c = restrict_avg(r);
  Eigen::MatrixXd delta_c = Eigen::MatrixXd::Zero(r_c.rows(), r_c.cols());
  delta_c = vcycle_uniform(std::move(delta_c), r_c, 2.0 * h,
                            n_pre, n_post, n_min);
  V += prolongate_bilinear(delta_c);
  gs_smooth(V, rho, h, n_post);
  return V;
}

void vcycle_amr_composite(amr::AMRArrays& a,
                          const amr::Quadtree& tree,
                          CompositeParams p) {
  // 1. Pre-smoothing on AMR (in place).
  amr::sor(a, {.omega = p.omega, .tol = 0.0, .max_iter = p.n_pre,
               .eps0 = p.eps0});

  // 2. AMR residual (in volts).
  const Eigen::VectorXd r = amr::residual(a, p.eps0);

  // 3. Volume-weighted restriction AMR -> uniform coarse grid at level_min.
  const int level_min = tree.level_min();
  const int N_c = 1 << level_min;
  const double h_c = tree.L() / static_cast<double>(N_c);
  Eigen::MatrixXd r_c = Eigen::MatrixXd::Zero(N_c, N_c);
  for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(a.keys.size()); ++n) {
    const amr::CellKey key = a.keys[static_cast<std::size_t>(n)];
    const uint8_t lv = amr::level_of(key);
    const uint32_t i = amr::i_of(key), j = amr::j_of(key);
    const int shift = static_cast<int>(lv) - level_min;
    const int I = static_cast<int>(i >> shift);
    const int J = static_cast<int>(j >> shift);
    const double weight = std::pow(4.0, level_min - static_cast<int>(lv));
    r_c(I, J) += weight * r(n);
  }

  // 4. Coarse solve: n V-cycles on A delta = r_c.
  // The uniform solver is formulated as `A V = rho` with
  //   (Vc V - sum) / h^2 = rho  =>  units of rho = V / m^2.
  // Our r_c is in volts, so feed rho_eff = r_c / h_c^2.
  const Eigen::MatrixXd rho_eff = r_c / (h_c * h_c);
  Eigen::MatrixXd delta_c = Eigen::MatrixXd::Zero(N_c, N_c);
  for (int k = 0; k < p.n_coarse_cycles; ++k) {
    delta_c = vcycle_uniform(std::move(delta_c), rho_eff, h_c, 3, 3, 4);
  }

  // 5. Bilinear prolongation from coarse grid to AMR leaves' centers.
  auto get_coarse = [&](int ii, int jj) -> double {
    if (ii < 0 || ii >= N_c || jj < 0 || jj >= N_c) return 0.0;
    return delta_c(ii, jj);
  };
  for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(a.keys.size()); ++n) {
    const amr::CellKey key = a.keys[static_cast<std::size_t>(n)];
    const auto [x, y] = tree.cell_center(key);
    const double u = x / h_c - 0.5;
    const double v = y / h_c - 0.5;
    const int i0 = static_cast<int>(std::floor(u));
    const int j0 = static_cast<int>(std::floor(v));
    const double fu = u - i0;
    const double fv = v - j0;
    const double d =
        (1.0 - fu) * (1.0 - fv) * get_coarse(i0,     j0) +
        fu        * (1.0 - fv) * get_coarse(i0 + 1, j0) +
        (1.0 - fu) * fv        * get_coarse(i0,     j0 + 1) +
        fu        * fv        * get_coarse(i0 + 1, j0 + 1);
    a.V(n) += d;
  }

  // 6. Post-smoothing on AMR.
  amr::sor(a, {.omega = p.omega, .tol = 0.0, .max_iter = p.n_post,
               .eps0 = p.eps0});
}

}  // namespace poisson::mg
