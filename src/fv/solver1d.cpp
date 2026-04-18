#include "poisson/fv/solver1d.hpp"

#include <stdexcept>

#include "poisson/linalg/thomas.hpp"

namespace poisson::fv {

Eigen::VectorXd solve_poisson_1d(Eigen::Ref<const Eigen::VectorXd> rho,
                                 double uL,
                                 double uR,
                                 const Grid1D& grid,
                                 double eps0) {
  const Eigen::Index N = grid.N;
  if (rho.size() != N) {
    throw std::invalid_argument("solve_poisson_1d: rho size must match grid.N");
  }

  const double dx = grid.dx();
  const double alpha = eps0 / (dx * dx);

  // Build tridiagonal system: interior rows eps0 V'' = -rho give
  //   alpha V_{i-1} - 2 alpha V_i + alpha V_{i+1} = -rho_i
  Eigen::VectorXd a = Eigen::VectorXd::Constant(N, alpha);
  Eigen::VectorXd b = Eigen::VectorXd::Constant(N, -2.0 * alpha);
  Eigen::VectorXd c = Eigen::VectorXd::Constant(N, alpha);
  Eigen::VectorXd d = -rho;

  // Dirichlet rows: identity on boundary nodes.
  a(0) = 0.0;
  b(0) = 1.0;
  c(0) = 0.0;
  d(0) = uL;

  a(N - 1) = 0.0;
  b(N - 1) = 1.0;
  c(N - 1) = 0.0;
  d(N - 1) = uR;

  return linalg::thomas(a, b, c, d);
}

}  // namespace poisson::fv
