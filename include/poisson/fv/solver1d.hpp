#pragma once

#include <Eigen/Core>

#include "poisson/core/grid.hpp"

namespace poisson::fv {

/// Solve eps0 V'' = -rho on [0, L] with Dirichlet BC V(0) = uL, V(L) = uR.
///
/// Uses the finite-volume 3-point stencil (equivalent to FD for uniform
/// grids). The discrete system is tridiagonal and solved by the Thomas
/// algorithm in O(N).
///
/// \param rho  Charge density at the N grid nodes.
/// \param uL   Dirichlet value at x = 0.
/// \param uR   Dirichlet value at x = L.
/// \param grid Node-centered grid (N nodes spanning [0, L]).
/// \param eps0 Permittivity (default 1.0 for normalized units).
/// \returns Potential V at the N grid nodes, with V(0) = uL and V(N-1) = uR.
[[nodiscard]] Eigen::VectorXd solve_poisson_1d(
    Eigen::Ref<const Eigen::VectorXd> rho,
    double uL,
    double uR,
    const Grid1D& grid,
    double eps0 = 1.0);

}  // namespace poisson::fv
