#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "poisson/linalg/thomas.hpp"

using poisson::linalg::thomas;

namespace {

// Reconstruct the dense tridiagonal matrix from (a, b, c).
// `a(0)` and `c(N-1)` are ignored (they lie outside the band).
Eigen::MatrixXd tridiag(const Eigen::VectorXd& a,
                        const Eigen::VectorXd& b,
                        const Eigen::VectorXd& c) {
  const Eigen::Index N = b.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
  for (Eigen::Index i = 0; i < N; ++i) {
    A(i, i) = b(i);
    if (i > 0)     A(i, i - 1) = a(i);
    if (i < N - 1) A(i, i + 1) = c(i);
  }
  return A;
}

}  // namespace

TEST_CASE("thomas solves a 3x3 dominant tridiagonal system", "[thomas]") {
  Eigen::VectorXd a(3); a << 0.0, 1.0, 1.0;
  Eigen::VectorXd b(3); b << 4.0, 4.0, 4.0;
  Eigen::VectorXd c(3); c << 1.0, 1.0, 0.0;
  Eigen::VectorXd d(3); d << 5.0, 6.0, 5.0;

  const Eigen::VectorXd x = thomas(a, b, c, d);
  const Eigen::VectorXd r = tridiag(a, b, c) * x - d;
  REQUIRE(r.cwiseAbs().maxCoeff() < 1e-14);
}

TEST_CASE("thomas matches dense solve on random diagonally dominant systems",
          "[thomas]") {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> udist(0.0, 1.0);

  for (int N : {10, 50, 200, 1000}) {
    Eigen::VectorXd a(N), b(N), c(N), x_theo(N);
    for (Eigen::Index i = 0; i < N; ++i) {
      a(i) = udist(rng);
      b(i) = udist(rng) + 2.0;   // ensures diagonal dominance
      c(i) = udist(rng);
      x_theo(i) = udist(rng);
    }
    Eigen::MatrixXd A = tridiag(a, b, c);
    Eigen::VectorXd d = A * x_theo;

    const Eigen::VectorXd x = thomas(a, b, c, d);
    REQUIRE((x - x_theo).cwiseAbs().maxCoeff() < 1e-10);
  }
}

TEST_CASE("thomas rejects mismatched sizes", "[thomas]") {
  Eigen::VectorXd a(3), b(3), c(3), d(2);
  a.setZero(); b.setOnes(); c.setZero(); d.setOnes();
  REQUIRE_THROWS_AS(thomas(a, b, c, d), std::invalid_argument);
}

TEST_CASE("thomas detects a zero first pivot", "[thomas]") {
  Eigen::VectorXd a(3); a << 0.0, 1.0, 1.0;
  Eigen::VectorXd b(3); b << 0.0, 4.0, 4.0;   // b(0) = 0 → singular
  Eigen::VectorXd c(3); c << 1.0, 1.0, 0.0;
  Eigen::VectorXd d(3); d << 1.0, 1.0, 1.0;
  REQUIRE_THROWS_AS(thomas(a, b, c, d), std::runtime_error);
}

TEST_CASE("thomas detects a zero interior pivot", "[thomas]") {
  // b(1) - a(1)*cp(0) = 1 - 1*(1/1) = 0
  Eigen::VectorXd a(3); a << 0.0, 1.0, 1.0;
  Eigen::VectorXd b(3); b << 1.0, 1.0, 4.0;
  Eigen::VectorXd c(3); c << 1.0, 1.0, 0.0;
  Eigen::VectorXd d(3); d << 1.0, 1.0, 1.0;
  REQUIRE_THROWS_AS(thomas(a, b, c, d), std::runtime_error);
}
