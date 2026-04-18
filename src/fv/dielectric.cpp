#include "poisson/fv/dielectric.hpp"

#include <stdexcept>

#include "poisson/linalg/thomas.hpp"

namespace poisson::fv {

Eigen::VectorXd solve_poisson_1d(Eigen::Ref<const Eigen::VectorXd> rho,
                                 Eigen::Ref<const Eigen::VectorXd> eps_r,
                                 double uL,
                                 double uR,
                                 const Grid1D& grid,
                                 double eps0) {
  const Eigen::Index N = grid.N;
  if (rho.size() != N || eps_r.size() != N) {
    throw std::invalid_argument(
        "solve_poisson_1d: rho and eps_r must have size grid.N");
  }
  if ((eps_r.array() <= 0.0).any()) {
    throw std::invalid_argument("solve_poisson_1d: eps_r must be strictly positive");
  }

  const double dx = grid.dx();

  // Harmonic-mean permittivity at interior faces (N-1 values).
  const Eigen::VectorXd eps_face =
      harmonic_mean(eps_r.segment(1, N - 1), eps_r.segment(0, N - 1)) * eps0
      / (dx * dx);

  Eigen::VectorXd a = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd c = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd d = -rho;

  // Interior rows: V_w V_{i-1} - (V_w + V_e) V_i + V_e V_{i+1} = -rho_i
  a.segment(1, N - 1) = eps_face;
  c.segment(0, N - 1) = eps_face;
  for (Eigen::Index i = 1; i < N - 1; ++i) {
    b(i) = -(a(i) + c(i));
  }

  // Dirichlet rows: identity.
  a(0) = 0.0;        b(0) = 1.0;        c(0) = 0.0;        d(0) = uL;
  a(N - 1) = 0.0;    b(N - 1) = 1.0;    c(N - 1) = 0.0;    d(N - 1) = uR;

  return linalg::thomas(a, b, c, d);
}

}  // namespace poisson::fv
