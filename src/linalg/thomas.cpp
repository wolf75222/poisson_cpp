#include "poisson/linalg/thomas.hpp"

#include <stdexcept>

namespace poisson::linalg {

Eigen::VectorXd thomas(Eigen::Ref<const Eigen::VectorXd> a,
                       Eigen::Ref<const Eigen::VectorXd> b,
                       Eigen::Ref<const Eigen::VectorXd> c,
                       Eigen::Ref<const Eigen::VectorXd> d) {
  const Eigen::Index N = d.size();
  if (N < 2) throw std::invalid_argument("thomas: N must be >= 2");
  if (a.size() != N || b.size() != N || c.size() != N) {
    throw std::invalid_argument("thomas: a, b, c, d must have the same length");
  }

  Eigen::VectorXd cp(N);   // modified super-diagonal
  Eigen::VectorXd dp(N);   // modified right-hand side
  Eigen::VectorXd x(N);

  // Forward sweep.
  cp(0) = c(0) / b(0);
  dp(0) = d(0) / b(0);
  for (Eigen::Index i = 1; i < N; ++i) {
    const double denom = b(i) - a(i) * cp(i - 1);
    cp(i) = (i < N - 1) ? c(i) / denom : 0.0;
    dp(i) = (d(i) - a(i) * dp(i - 1)) / denom;
  }

  // Backward substitution.
  x(N - 1) = dp(N - 1);
  for (Eigen::Index i = N - 2; i >= 0; --i) {
    x(i) = dp(i) - cp(i) * x(i + 1);
  }
  return x;
}

}  // namespace poisson::linalg
