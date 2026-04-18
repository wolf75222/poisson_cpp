#include "poisson/spectral/dst1d.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson::spectral {

DSTSolver1D::DSTSolver1D(int N, double L, double eps0)
    : N_(N), L_(L), eps0_(eps0), in_(N), out_(N) {
  if (N < 2) throw std::invalid_argument("DSTSolver1D: N must be >= 2");
  if (!(L > 0.0)) throw std::invalid_argument("DSTSolver1D: L must be > 0");
  if (!(eps0 > 0.0)) throw std::invalid_argument("DSTSolver1D: eps0 must be > 0");

  const double h = L / static_cast<double>(N + 1);
  // Discrete eigenvalues of the 3-point Laplacian:
  //   lambda_k = 4 sin^2(k pi / (2(N+1))) / h^2, k = 1..N.
  Eigen::VectorXd lam(N);
  for (int k = 1; k <= N; ++k) {
    const double s = std::sin(k * std::numbers::pi / (2.0 * (N + 1)));
    lam(k - 1) = 4.0 / (h * h) * s * s;
  }
  lam_inv_ = 1.0 / (eps0 * lam.array());

  // Plans: DST-I (FFTW_RODFT00). Forward and inverse share the same shape.
  plan_fwd_ = FFTWPlan(::fftw_plan_r2r_1d(N, in_.data(), out_.data(),
                                          FFTW_RODFT00, FFTW_MEASURE));
  plan_inv_ = FFTWPlan(::fftw_plan_r2r_1d(N, in_.data(), out_.data(),
                                          FFTW_RODFT00, FFTW_MEASURE));
  // FFTW_MEASURE may reuse the buffers, so `in_` and `out_` may contain junk
  // after planning. Reset them explicitly.
  in_.setZero();
  out_.setZero();
}

Eigen::VectorXd DSTSolver1D::solve(Eigen::Ref<const Eigen::VectorXd> rho) const {
  if (rho.size() != N_) {
    throw std::invalid_argument("DSTSolver1D::solve: rho size must be N");
  }

  // Forward DST-I: rho -> rho_hat.
  in_ = rho;
  plan_fwd_.execute();
  // FFTW's RODFT00 normalization: DST(DST(x)) = 2(N+1) x. We normalize
  // symmetrically (orthonormal DST-I) here so that division-then-inverse
  // returns the original scale.
  const double norm = 1.0 / std::sqrt(2.0 * (N_ + 1));
  Eigen::VectorXd rho_hat = out_ * norm;

  // Divide by eps0 * lambda_k in the spectral domain.
  Eigen::VectorXd V_hat = rho_hat.cwiseProduct(lam_inv_);

  // Inverse DST-I.
  in_ = V_hat;
  plan_inv_.execute();
  return out_ * norm;
}

}  // namespace poisson::spectral
