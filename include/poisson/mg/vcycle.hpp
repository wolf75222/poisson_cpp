#pragma once

#include <Eigen/Core>

#include "poisson/amr/quadtree.hpp"
#include "poisson/amr/solver.hpp"

namespace poisson::mg {

/// One Gauss-Seidel red-black smoothing sweep on the uniform cell-centered
/// FV Poisson operator with Dirichlet V = 0 on the full boundary.
/// Updates V in place. `rho` is the right-hand side (same units as V / h^2).
void gs_smooth(Eigen::Ref<Eigen::MatrixXd> V,
               Eigen::Ref<const Eigen::MatrixXd> rho,
               double h,
               int n_iter);

/// Apply the uniform FV Laplacian A V.
Eigen::MatrixXd laplacian_fv(Eigen::Ref<const Eigen::MatrixXd> V, double h);

/// Restriction by 4-cell averaging: (N, N) -> (N/2, N/2). N must be even.
Eigen::MatrixXd restrict_avg(Eigen::Ref<const Eigen::MatrixXd> r);

/// Piecewise-constant prolongation (order 0): (N/2, N/2) -> (N, N).
Eigen::MatrixXd prolongate_const(Eigen::Ref<const Eigen::MatrixXd> c);

/// Bilinear prolongation (order 2): (M, M) -> (2M, 2M), for cell-centered FV.
/// Each fine cell gets a weighted combination of the enclosing coarse cell
/// and its two neighbours in the direction of the fine cell's offset.
Eigen::MatrixXd prolongate_bilinear(Eigen::Ref<const Eigen::MatrixXd> c);

/// Recursive V-cycle multigrid on a uniform grid. Returns the updated V.
///
/// Coarsens by factor 2 down to size <= n_min; smoothes more aggressively at
/// the coarsest level (no "exact solve" yet, just extra GS sweeps). Port of
/// `vcycle` from `CourseOnPoisson/notebooks/TP5_AMR_Poisson_2D.ipynb`.
Eigen::MatrixXd vcycle_uniform(Eigen::MatrixXd V,
                               const Eigen::MatrixXd& rho,
                               double h,
                               int n_pre = 2,
                               int n_post = 2,
                               int n_min = 4);

/// Parameters for the 2-grid composite V-cycle on an AMR mesh.
struct CompositeParams {
  int n_pre = 3;             ///< pre-smoothing SOR sweeps on AMR
  int n_post = 3;            ///< post-smoothing SOR sweeps on AMR
  int n_coarse_cycles = 4;   ///< number of uniform V-cycles on the coarse problem
  double omega = 1.85;       ///< SOR omega for AMR smoothing
  double eps0 = 1.0;
};

/// One composite 2-grid V-cycle on an AMR tree:
///   1. SOR pre-smoothing on AMR leaves;
///   2. AMR residual r;
///   3. volume-weighted restriction r -> r_c on a uniform 2^level_min grid;
///   4. `n_coarse_cycles` uniform V-cycles to approximately solve A δ = r_c;
///   5. bilinear prolongation of δ back to AMR leaves (V += δ);
///   6. SOR post-smoothing on AMR leaves.
/// The AMR array state is updated in place.
void vcycle_amr_composite(amr::AMRArrays& a,
                          const amr::Quadtree& tree,
                          CompositeParams p = {});

}  // namespace poisson::mg
