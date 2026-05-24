#pragma once

#include <Eigen/Core>

#include "poisson/spectral/fftw_wrap.hpp"

namespace poisson::spectral {

/// Spectral Poisson solver on a 2D uniform PERIODIC grid using the
/// discrete Fourier transform (FFTW r2c / c2r). Sister to `DSTSolver2D`,
/// which gives Dirichlet BC; this one gives periodic BC.
///
/// Grid: Nx × Ny cells, period (Lx, Ly), spacings hx = Lx/Nx, hy = Ly/Ny
/// (no half-cell shift — fully periodic grid). Cells are indexed
/// i = 0..Nx-1, j = 0..Ny-1.
///
/// Solves `eps0 ∇²V = -rho` for V where ∇² is the standard 5-point
/// periodic Laplacian. Because the periodic Laplacian has a constant
/// null mode (V = const), a solution exists only when ∫ ρ = 0 (mean
/// zero). The solver subtracts the mean of ρ automatically and pins
/// the mean of V to 0 — typical convention for self-gravity codes.
class DFTSolver2D {
 public:
  DFTSolver2D(int Nx, int Ny, double Lx, double Ly, double eps0 = 1.0);
  DFTSolver2D(const DFTSolver2D&) = delete;
  DFTSolver2D& operator=(const DFTSolver2D&) = delete;
  DFTSolver2D(DFTSolver2D&&) noexcept = default;
  DFTSolver2D& operator=(DFTSolver2D&&) noexcept = default;

  /// Solve eps0 ∇²V = -rho. Returns V at the (Nx, Ny) cell centres.
  [[nodiscard]] Eigen::MatrixXd solve(
      Eigen::Ref<const Eigen::MatrixXd> rho) const;

  [[nodiscard]] int Nx() const noexcept { return Nx_; }
  [[nodiscard]] int Ny() const noexcept { return Ny_; }

 private:
  int Nx_, Ny_;
  // Real workspace: column-major Eigen (Nx, Ny) corresponds to FFTW's
  // row-major (Ny, Nx) interpretation (slowest = Ny outer, fastest = Nx
  // inner) when we pass dims as {Ny, Nx} to FFTW.
  mutable Eigen::MatrixXd real_buf_;          // (Nx, Ny)
  mutable Eigen::MatrixXcd complex_buf_;       // (Nx/2 + 1, Ny)
  Eigen::MatrixXd lam_;                        // (Nx/2 + 1, Ny) — Laplacian eigenvalues
  FFTWPlan plan_fwd_;
  FFTWPlan plan_inv_;
};

}  // namespace poisson::spectral
