#pragma once

#include <vector>

#include <Eigen/Core>

#include "poisson/amr/quadtree.hpp"

namespace poisson::amr {

/// Flat, array-based view of a quadtree suitable for fast SOR iterations.
///
/// Converts the leaves of a `Quadtree` into parallel arrays plus precomputed
/// finite-volume stencil weights that handle heterogeneous cell sizes at
/// coarse-fine interfaces and Dirichlet ghost cells at the domain boundary.
///
/// Weight convention (cf. bilan of `TP5_AMR_Poisson_2D.ipynb`):
///   - boundary (Dirichlet V = 0) : diag += 2, off = 0
///   - same-level neighbour       : diag += 1, off = 1
///   - coarser neighbour          : diag += 2/3, off = 2/3
///   - 2 finer neighbours         : diag += 4/3, off = 2/3 each
struct AMRArrays {
  std::vector<CellKey> keys;                ///< leaf key for index i
  Eigen::VectorXd V;
  Eigen::VectorXd rho;
  Eigen::VectorXd h;                        ///< per-leaf cell size
  Eigen::VectorXd Vc;                       ///< diagonal coefficient
  Eigen::Matrix<int64_t, Eigen::Dynamic, 4> nb0;   ///< first neighbour index (-1 if absent)
  Eigen::Matrix<int64_t, Eigen::Dynamic, 4> nb1;   ///< second neighbour index (-1 if absent)
  Eigen::Matrix<double,  Eigen::Dynamic, 4> w0;    ///< off-diag weight for nb0
  Eigen::Matrix<double,  Eigen::Dynamic, 4> w1;    ///< off-diag weight for nb1
};

/// Build the array-based AMR view from a fully built Quadtree.
AMRArrays extract_arrays(const Quadtree& tree);

/// Write V back from the flat array into the corresponding tree leaves.
void writeback(Quadtree& tree,
               const std::vector<CellKey>& keys,
               Eigen::Ref<const Eigen::VectorXd> V);

/// SOR parameters for the AMR solver. See `AMRArrays` for the stencil.
struct SORParams {
  double omega = 1.85;
  double tol = 1e-8;
  int max_iter = 20'000;
  double eps0 = 1.0;
};

struct SORReport {
  int iterations;
  double residual;
};

/// Run a Gauss-Seidel-like SOR on the AMR arrays, in place.
SORReport sor(AMRArrays& a, SORParams p = {});

/// Compute the FV residual on AMR arrays (in V).
/// r_i = sum_off w * V_neigh + h_i^2 rho_i / eps0 - Vc_i V_i
Eigen::VectorXd residual(const AMRArrays& a, double eps0 = 1.0);

}  // namespace poisson::amr
