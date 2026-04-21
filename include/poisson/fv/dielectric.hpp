#pragma once

#include <Eigen/Core>

#include "poisson/core/grid.hpp"

namespace poisson::fv {

/// Element-wise harmonic mean: 2 a b / (a + b).
///
/// Used to compute the effective permittivity at the face between two
/// adjacent cells with different relative permittivities: this is the
/// formula that preserves the normal component of the displacement
/// D = eps0 eps_r grad V across a dielectric interface.
[[nodiscard]] inline Eigen::VectorXd harmonic_mean(
    Eigen::Ref<const Eigen::VectorXd> a,
    Eigen::Ref<const Eigen::VectorXd> b) {
  return (2.0 * a.array() * b.array()) / (a.array() + b.array());
}

/// Solve eps0 d/dx[eps_r(x) dV/dx] = -rho with Dirichlet BC V(0) = uL,
/// V(L) = uR, using the finite-volume 3-point stencil and the harmonic
/// mean of eps_r at cell faces.
///
/// \param rho   Charge density at the N grid nodes.
/// \param eps_r Relative permittivity per node, shape (N,). Must be > 0.
/// \param uL    Dirichlet value at x = 0.
/// \param uR    Dirichlet value at x = L.
/// \param grid  Node-centered grid.
/// \param eps0  Vacuum permittivity (default 1.0 for normalized units).
/// \returns Potential V at the N grid nodes.
[[nodiscard]] Eigen::VectorXd solve_poisson_1d(
    Eigen::Ref<const Eigen::VectorXd> rho,
    Eigen::Ref<const Eigen::VectorXd> eps_r,
    double uL,
    double uR,
    const Grid1D& grid,
    double eps0 = 1.0);

}  // namespace poisson::fv
