#pragma once

#include <Eigen/Core>

#include "poisson/spectral/fftw_wrap.hpp"

namespace poisson::spectral {

/// Spectral Poisson solver on a 1D uniform grid with Dirichlet homogeneous
/// BC (V = 0 at both endpoints), using the type-I Discrete Sine Transform.
///
/// The grid has N interior points at x_i = i * h for i = 1..N with
/// h = L / (N + 1). The endpoints x_0 = 0 and x_{N+1} = L are excluded
/// from the unknowns. Convention matches
/// `CourseOnPoisson/notebooks/TP4_Poisson_1D_spectral.ipynb`.
///
/// The DST-I plan is built once at construction and reused across solves.
class DSTSolver1D {
 public:
  DSTSolver1D(int N, double L, double eps0 = 1.0);
  DSTSolver1D(const DSTSolver1D&) = delete;
  DSTSolver1D& operator=(const DSTSolver1D&) = delete;
  DSTSolver1D(DSTSolver1D&&) noexcept = default;
  DSTSolver1D& operator=(DSTSolver1D&&) noexcept = default;

  /// Solve eps0 V'' = -rho on the N interior points.
  /// \param rho right-hand side, shape (N,).
  /// \returns potential V at the N interior points.
  Eigen::VectorXd solve(Eigen::Ref<const Eigen::VectorXd> rho) const;

  int N() const noexcept { return N_; }
  double L() const noexcept { return L_; }

 private:
  int N_;
  double L_;
  double eps0_;
  // Scratch buffers (fftw_plan requires owning arrays that outlive the plan).
  mutable Eigen::VectorXd in_, out_;
  Eigen::VectorXd lam_inv_;   // 1 / (eps0 * lambda_k), precomputed
  FFTWPlan plan_fwd_;
  FFTWPlan plan_inv_;
};

}  // namespace poisson::spectral
