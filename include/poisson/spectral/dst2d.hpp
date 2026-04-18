#pragma once

#include <Eigen/Core>

#include "poisson/spectral/fftw_wrap.hpp"

namespace poisson::spectral {

/// Spectral Poisson solver on a 2D uniform grid with Dirichlet homogeneous
/// BC on the whole boundary, using the type-I Discrete Sine Transform.
///
/// The grid has Nx x Ny interior points with
///   hx = Lx / (Nx + 1), hy = Ly / (Ny + 1),
///   x_i = i * hx for i = 1..Nx, y_j = j * hy for j = 1..Ny.
/// Convention matches
/// `CourseOnPoisson/notebooks/TP4bis_Poisson_2D_spectral.ipynb`.
class DSTSolver2D {
 public:
  DSTSolver2D(int Nx, int Ny, double Lx, double Ly, double eps0 = 1.0);
  DSTSolver2D(const DSTSolver2D&) = delete;
  DSTSolver2D& operator=(const DSTSolver2D&) = delete;
  DSTSolver2D(DSTSolver2D&&) noexcept = default;
  DSTSolver2D& operator=(DSTSolver2D&&) noexcept = default;

  /// Solve eps0 (V_xx + V_yy) = -rho on the (Nx, Ny) interior points.
  Eigen::MatrixXd solve(Eigen::Ref<const Eigen::MatrixXd> rho) const;

  int Nx() const noexcept { return Nx_; }
  int Ny() const noexcept { return Ny_; }

 private:
  int Nx_, Ny_;
  mutable Eigen::MatrixXd in_, out_;   // column-major Eigen => FFTW ordering
  Eigen::MatrixXd lam_inv_;
  FFTWPlan plan_fwd_;
  FFTWPlan plan_inv_;
};

}  // namespace poisson::spectral
