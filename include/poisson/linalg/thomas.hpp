#pragma once

#include <Eigen/Core>

namespace poisson::linalg {

/// Solve a tridiagonal system A x = d by the Thomas algorithm in O(N).
///
/// The matrix A has `a` on the sub-diagonal, `b` on the diagonal, `c` on the
/// super-diagonal. The entries `a(0)` and `c(N-1)` are unused (they lie
/// outside the matrix). All vectors must have the same length N >= 2.
///
/// Numerically stable for diagonally dominant or symmetric positive definite
/// matrices (no pivoting). Port of the Python `thomas()` in
/// `CourseOnPoisson/notebooks/TP1_Poisson_1D.ipynb`.
///
/// \param a sub-diagonal, shape (N,). `a(0)` ignored.
/// \param b diagonal, shape (N,).
/// \param c super-diagonal, shape (N,). `c(N-1)` ignored.
/// \param d right-hand side, shape (N,).
/// \returns solution x, shape (N,).
/// \throws std::invalid_argument if lengths mismatch or N < 2.
Eigen::VectorXd thomas(Eigen::Ref<const Eigen::VectorXd> a,
                       Eigen::Ref<const Eigen::VectorXd> b,
                       Eigen::Ref<const Eigen::VectorXd> c,
                       Eigen::Ref<const Eigen::VectorXd> d);

}  // namespace poisson::linalg
