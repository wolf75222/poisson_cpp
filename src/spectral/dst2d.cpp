#include "poisson/spectral/dst2d.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson::spectral {

namespace {

// Eigenvalues of the 1D discrete Laplacian of size N spanning [0, L]:
//   lambda_k = 4 sin^2(k pi / (2(N+1))) / h^2, h = L/(N+1), k = 1..N.
Eigen::VectorXd discrete_eigenvalues(int N, double L) {
  const double h = L / static_cast<double>(N + 1);
  Eigen::VectorXd lam(N);
  for (int k = 1; k <= N; ++k) {
    const double s = std::sin(k * std::numbers::pi / (2.0 * (N + 1)));
    lam(k - 1) = 4.0 / (h * h) * s * s;
  }
  return lam;
}

}  // namespace

DSTSolver2D::DSTSolver2D(int Nx, int Ny, double Lx, double Ly, double eps0)
    : Nx_(Nx), Ny_(Ny), in_(Nx, Ny), out_(Nx, Ny) {
  if (Nx < 2 || Ny < 2) throw std::invalid_argument("DSTSolver2D: Nx,Ny >= 2");
  if (!(Lx > 0.0 && Ly > 0.0))
    throw std::invalid_argument("DSTSolver2D: Lx, Ly must be > 0");
  if (!(eps0 > 0.0))
    throw std::invalid_argument("DSTSolver2D: eps0 must be > 0");

  const Eigen::VectorXd lam_x = discrete_eigenvalues(Nx, Lx);
  const Eigen::VectorXd lam_y = discrete_eigenvalues(Ny, Ly);
  // lambda_{k,l} = lambda_k^x + lambda_l^y (separability).
  lam_inv_.resize(Nx, Ny);
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      lam_inv_(i, j) = 1.0 / (eps0 * (lam_x(i) + lam_y(j)));
    }
  }

  // Eigen matrices are column-major by default; FFTW layout for
  // fftw_plan_r2r_2d(n0, n1) is row-major with first index varying slowest.
  // We supply dimensions in Eigen's order (cols = outer) using
  // fftw_plan_r2r with explicit rank-2 descriptor.
  int ns[2] = { Ny, Nx };
  ::fftw_r2r_kind kinds_fwd[2] = { FFTW_RODFT00, FFTW_RODFT00 };
  ::fftw_r2r_kind kinds_inv[2] = { FFTW_RODFT00, FFTW_RODFT00 };

  plan_fwd_ = FFTWPlan(::fftw_plan_r2r(2, ns, in_.data(), out_.data(),
                                        kinds_fwd, FFTW_MEASURE));
  plan_inv_ = FFTWPlan(::fftw_plan_r2r(2, ns, in_.data(), out_.data(),
                                        kinds_inv, FFTW_MEASURE));
  in_.setZero();
  out_.setZero();
}

Eigen::MatrixXd DSTSolver2D::solve(Eigen::Ref<const Eigen::MatrixXd> rho) const {
  if (rho.rows() != Nx_ || rho.cols() != Ny_) {
    throw std::invalid_argument("DSTSolver2D::solve: rho shape must be (Nx, Ny)");
  }

  // Orthonormal normalization: factor 1/sqrt(2(Nx+1) * 2(Ny+1)).
  const double norm = 1.0 / std::sqrt(4.0 * (Nx_ + 1) * (Ny_ + 1));

  in_ = rho;
  plan_fwd_.execute();
  Eigen::MatrixXd rho_hat = out_ * norm;

  Eigen::MatrixXd V_hat = rho_hat.cwiseProduct(lam_inv_);

  in_ = V_hat;
  plan_inv_.execute();
  return out_ * norm;
}

}  // namespace poisson::spectral
