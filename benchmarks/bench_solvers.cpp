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

#include "poisson/amr/quadtree.hpp"
#include "poisson/amr/solver.hpp"
#include "poisson/core/grid.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/linalg/thomas.hpp"
#include "poisson/mg/vcycle.hpp"

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
    poisson::fv::Solver2D::Report last{};
    const double ms = bench(
        [&] {
          V.setZero();
          last = solver.solve(V, rho, {.tol = 1e-8, .max_iter = 50'000});
        },
        repeats);
    std::cout << "sor2d       N=" << N << "x" << N << "  " << ms << " ms"
              << "  (" << last.iterations << " iter)\n";
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

  // mg::gs_smooth on a uniform grid. Isolates the smoother kernel from the
  // full V-cycle so we see the effect of hot-loop micro-optimizations.
  {
    Eigen::MatrixXd V   = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd rho = Eigen::MatrixXd::Random(N, N);
    const double h = 1.0 / N;
    const double ms = bench(
        [&] { poisson::mg::gs_smooth(V, rho, h, 50); },
        repeats);
    std::cout << "gs_smooth   N=" << N << "x" << N << "  " << ms << " ms"
              << "  (50 sweeps)\n";
  }

  // AMR SOR on a moderately refined quadtree. The leaf count drives the
  // hot-loop cost, so we report both the wall time and leaf count.
  {
    poisson::amr::Quadtree tree(1.0, 4);   // base 16x16
    auto pred = [](poisson::amr::CellKey k) {
      const uint8_t lv = poisson::amr::level_of(k);
      if (lv >= 6) return false;
      const uint32_t i = poisson::amr::i_of(k), j = poisson::amr::j_of(k);
      const double hh = 1.0 / (1u << lv);
      const double x = (i + 0.5) * hh, y = (j + 0.5) * hh;
      const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
      return r2 < 0.03;
    };
    auto rho_fn = [](double x, double y) {
      const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
      return std::exp(-r2 / 0.01);
    };
    tree.build(pred, /*level_max=*/6, rho_fn);
    auto arr = poisson::amr::extract_arrays(tree);
    const auto n_leaves = arr.keys.size();

    auto arr0 = arr;   // snapshot so each run starts from the same V = 0
    const double ms = bench(
        [&] {
          arr = arr0;
          poisson::amr::sor(arr, {.omega = 1.85, .tol = 0.0, .max_iter = 200,
                                   .eps0 = 1.0});
        },
        repeats);
    std::cout << "amr_sor     leaves=" << n_leaves
              << "  " << ms << " ms  (200 sweeps)\n";
  }

  return 0;
}
