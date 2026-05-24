#include "poisson/spectral/dft2d.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace poisson::spectral {

namespace {

// Eigenvalue of the discrete periodic Laplacian, 5-point stencil, at
// wave number m on a grid of N cells over period L:
//   ∂²_x e^{i 2π m x / L} ≈ (2/h²) (cos(2π m / N) - 1) e^{...}
//                          = -(4/h²) sin²(π m / N) e^{...}
// Returns the negative eigenvalue (so that -λ ≥ 0 — convenient for
// avoiding sign juggling when forming 1/(eps0·λ) on non-zero modes).
double lambda_x(int m, int N, double L) {
  const double h = L / static_cast<double>(N);
  const double s = std::sin(std::numbers::pi * static_cast<double>(m) /
                              static_cast<double>(N));
  return 4.0 / (h * h) * s * s;
}

}  // namespace

DFTSolver2D::DFTSolver2D(int Nx, int Ny, double Lx, double Ly, double eps0)
    : Nx_(Nx), Ny_(Ny),
      real_buf_(Nx, Ny),
      complex_buf_(Nx / 2 + 1, Ny),
      lam_(Nx / 2 + 1, Ny) {
  if (Nx < 2 || Ny < 2) throw std::invalid_argument("DFTSolver2D: Nx,Ny >= 2");
  if (Nx % 2 != 0)
    throw std::invalid_argument("DFTSolver2D: Nx must be even (FFTW r2c)");
  if (!(Lx > 0.0 && Ly > 0.0))
    throw std::invalid_argument("DFTSolver2D: Lx, Ly must be > 0");
  if (!(eps0 > 0.0))
    throw std::invalid_argument("DFTSolver2D: eps0 must be > 0");

  // Pre-compute λ at the (Nx/2 + 1) × Ny half-spectrum.  We index the
  // first dim by m = 0..Nx/2 (FFTW r2c output convention) and the
  // second dim by n = 0..Ny-1 (real-valued FFT in x, full DFT in y so
  // n wraps as 0..Ny/2,-Ny/2+1..-1 ; the eigenvalue formula is
  // symmetric so we just use min(n, Ny-n) implicitly via sin²(πn/Ny)).
  for (int j = 0; j < Ny; ++j) {
    const double ly = lambda_x(j, Ny, Ly);
    for (int i = 0; i <= Nx / 2; ++i) {
      const double lx = lambda_x(i, Nx, Lx);
      lam_(i, j) = eps0 * (lx + ly);
    }
  }

  // FFTW plans: real Nx*Ny -> complex (Nx/2 + 1)*Ny via dft_r2c, and
  // the inverse. Eigen column-major (Nx, Ny) memory is identical to
  // FFTW row-major (Ny, Nx) — pass dims as {Ny, Nx}.
  int ns[2] = { Ny, Nx };
  plan_fwd_ = FFTWPlan(::fftw_plan_dft_r2c(
      2, ns, real_buf_.data(),
      reinterpret_cast<::fftw_complex*>(complex_buf_.data()),
      FFTW_MEASURE));
  plan_inv_ = FFTWPlan(::fftw_plan_dft_c2r(
      2, ns,
      reinterpret_cast<::fftw_complex*>(complex_buf_.data()),
      real_buf_.data(),
      FFTW_MEASURE));
  real_buf_.setZero();
  complex_buf_.setZero();
}

Eigen::MatrixXd DFTSolver2D::solve(Eigen::Ref<const Eigen::MatrixXd> rho) const {
  if (rho.rows() != Nx_ || rho.cols() != Ny_) {
    throw std::invalid_argument(
        "DFTSolver2D::solve: rho shape must be (Nx, Ny)");
  }

  // Subtract the mean so the periodic Laplacian has a solution. The mean
  // of V (the constant null mode) is conventionally pinned to 0.
  const double mean = rho.mean();
  real_buf_ = rho.array() - mean;

  plan_fwd_.execute();
  // complex_buf_ now holds ρ̂(m, n) (unnormalised, FFTW convention).

  // V̂(m, n) = ρ̂(m, n) / λ(m, n)  for (m, n) ≠ (0, 0) ; V̂(0,0) = 0.
  // (Mean removal above guarantees ρ̂(0,0) ≈ 0 to round-off, so the
  // division is well defined under any safeguard.)
  complex_buf_(0, 0) = std::complex<double>(0.0, 0.0);
  for (int j = 0; j < Ny_; ++j) {
    for (int i = 0; i <= Nx_ / 2; ++i) {
      if (i == 0 && j == 0) continue;
      complex_buf_(i, j) /= lam_(i, j);
    }
  }

  plan_inv_.execute();
  // FFTW inverse is unnormalised — divide by Nx * Ny.
  return real_buf_ / static_cast<double>(Nx_ * Ny_);
}

}  // namespace poisson::spectral
