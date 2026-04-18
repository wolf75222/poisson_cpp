// Minimal chrono-based benchmarks for the poisson_cpp solvers.
// Usage:   ./bench_solvers [N]
//
// Reports wall time (median of K runs) for:
//   - Thomas (tridiagonal, size N)
//   - Solver2D SOR red-black (N x N grid, Dirichlet)
//   - DSTSolver2D spectral (N x N grid)  [only if built with FFTW3]

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/linalg/thomas.hpp"

#if defined(POISSON_HAVE_FFTW3)
#  include "poisson/spectral/dst2d.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double median_ms(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

template <class F>
double bench(F fn, int repeats) {
  std::vector<double> samples;
  samples.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return median_ms(std::move(samples));
}

}  // namespace

int main(int argc, char** argv) {
  const int N = (argc > 1) ? std::atoi(argv[1]) : 128;
  const int repeats = 5;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> udist(0.0, 1.0);

  // Thomas.
  {
    Eigen::VectorXd a(N), b(N), c(N), d(N);
    for (int i = 0; i < N; ++i) {
      a(i) = udist(rng); b(i) = udist(rng) + 2.0;
      c(i) = udist(rng); d(i) = udist(rng);
    }
    const double ms = bench(
        [&] { volatile auto x = poisson::linalg::thomas(a, b, c, d); (void)x; },
        100);
    std::cout << "thomas      N=" << N << "  " << ms << " ms\n";
  }

  // Solver2D SOR.
  {
    poisson::Grid2D grid(1.0, 1.0, N, N);
    poisson::fv::Solver2D solver(grid, 1.0, 0.0, 1.0);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
    const double ms = bench(
        [&] {
          V.setZero();
          solver.solve(V, rho, {.tol = 1e-8, .max_iter = 50'000});
        },
        repeats);
    std::cout << "sor2d       N=" << N << "x" << N << "  " << ms << " ms\n";
  }

#if defined(POISSON_HAVE_FFTW3)
  // Spectral 2D.
  {
    poisson::spectral::DSTSolver2D solver(N, N, 1.0, 1.0);
    Eigen::MatrixXd rho = Eigen::MatrixXd::Random(N, N);
    const double ms = bench([&] { volatile auto V = solver.solve(rho); (void)V; },
                             50);
    std::cout << "spectral2d  N=" << N << "x" << N << "  " << ms << " ms\n";
  }
#endif

  return 0;
}
