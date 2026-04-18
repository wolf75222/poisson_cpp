#pragma once

#include <cmath>
#include <concepts>
#include <functional>
#include <vector>

#include <Eigen/Core>

/// Matrix-free Conjugate Gradient / Preconditioned CG solvers.
///
/// Reference: Saad, *Iterative Methods for Sparse Linear Systems* (2nd ed.),
/// Alg. 6.18 (CG) and 9.1 (PCG). Chosen over SOR for symmetric
/// positive-definite systems (our discrete 2D Laplacian with Dirichlet /
/// Neumann BCs on a cell-centered FV mesh): CG converges in `O(sqrt(κ))`
/// iterations vs `O(κ)` for SOR, where κ is the condition number of the
/// operator.
///
/// The solver is *matrix-free*: the caller provides two lambdas,
///   - `apply`: y = A x, the linear operator action on x
///   - `preconditioner` (optional): y = M⁻¹ x, an approximate inverse of A
/// Both accept and return `Eigen::MatrixXd` of the same shape. This keeps
/// the code reusable across FV / spectral / AMR operators.

namespace poisson::iter {

/// Concept for a linear operator: takes a matrix, returns a matrix of the
/// same shape.
template <typename F>
concept MatrixOp = std::invocable<F, const Eigen::MatrixXd&> &&
                   std::convertible_to<std::invoke_result_t<F, const Eigen::MatrixXd&>,
                                        Eigen::MatrixXd>;

struct CGParams {
  double tol      = 1e-8;   ///< relative residual tolerance: ||r|| / ||b||
  int    max_iter = 5'000;
};

struct CGReport {
  int    iterations;
  double residual;          ///< final ||r|| / ||b||
};

/// Unpreconditioned Conjugate Gradient. Requires A symmetric-positive-
/// definite.  Solves A x = b starting from x.
///
/// \param apply       callable y = A x
/// \param x           initial guess, overwritten with the solution
/// \param b           right-hand side
/// \param p           tolerance and iteration cap
/// \param history     optional: if non-null, appends `||r||_2 / ||b||_2`
///                    after every iteration (for convergence plots).
template <MatrixOp Apply>
[[nodiscard]] CGReport cg(Apply&& apply,
                          Eigen::Ref<Eigen::MatrixXd> x,
                          Eigen::Ref<const Eigen::MatrixXd> b,
                          CGParams p = {},
                          std::vector<double>* history = nullptr) {
  const double b_norm = b.norm();
  if (b_norm == 0.0) {
    x.setZero();
    return {0, 0.0};
  }
  Eigen::MatrixXd r = b - apply(Eigen::MatrixXd(x));
  Eigen::MatrixXd d = r;
  double rr = (r.array() * r.array()).sum();
  double res = std::sqrt(rr) / b_norm;
  if (history) history->push_back(res);
  int iter = 0;
  for (; iter < p.max_iter && res > p.tol; ++iter) {
    Eigen::MatrixXd Ad = apply(d);
    const double alpha = rr / (d.array() * Ad.array()).sum();
    x.noalias() += alpha * d;
    r.noalias() -= alpha * Ad;
    const double rr_new = (r.array() * r.array()).sum();
    const double beta = rr_new / rr;
    d = r + beta * d;
    rr = rr_new;
    res = std::sqrt(rr) / b_norm;
    if (history) history->push_back(res);
  }
  return {iter, res};
}

/// Preconditioned Conjugate Gradient. `precond(r)` returns M⁻¹ r. A good
/// choice for a symmetric FV operator is diagonal Jacobi:
/// `precond(r) = r ./ diag(A)`.
template <MatrixOp Apply, MatrixOp Precond>
[[nodiscard]] CGReport pcg(Apply&& apply, Precond&& precond,
                           Eigen::Ref<Eigen::MatrixXd> x,
                           Eigen::Ref<const Eigen::MatrixXd> b,
                           CGParams p = {},
                           std::vector<double>* history = nullptr) {
  const double b_norm = b.norm();
  if (b_norm == 0.0) {
    x.setZero();
    return {0, 0.0};
  }
  Eigen::MatrixXd r = b - apply(Eigen::MatrixXd(x));
  Eigen::MatrixXd z = precond(r);
  Eigen::MatrixXd d = z;
  double rz = (r.array() * z.array()).sum();
  double res = r.norm() / b_norm;
  if (history) history->push_back(res);
  int iter = 0;
  for (; iter < p.max_iter && res > p.tol; ++iter) {
    Eigen::MatrixXd Ad = apply(d);
    const double alpha = rz / (d.array() * Ad.array()).sum();
    x.noalias() += alpha * d;
    r.noalias() -= alpha * Ad;
    res = r.norm() / b_norm;
    if (history) history->push_back(res);
    if (res <= p.tol) { ++iter; break; }
    z = precond(r);
    const double rz_new = (r.array() * z.array()).sum();
    const double beta = rz_new / rz;
    d = z + beta * d;
    rz = rz_new;
  }
  return {iter, res};
}

}  // namespace poisson::iter
