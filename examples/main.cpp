// Demo binary for the poisson_cpp library.
//
// Dispatches to individual solvers via a `--problem` flag. This is a
// companion harness for the library defined in `include/poisson/*`; the
// value is in the library, not in this CLI.
//
//   ./poisson_demo --problem poisson1d --N 50
//   ./poisson_demo --problem sor2d --N 64
//   ./poisson_demo --problem spectral2d --N 64
//   ./poisson_demo --problem amr --Nmin 4 --Nmax 7

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "poisson/amr/quadtree.hpp"
#include "poisson/amr/solver.hpp"
#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"
#include "poisson/fv/solver2d.hpp"

#if defined(POISSON_HAVE_FFTW3)
#  include "poisson/spectral/dst2d.hpp"
#endif

namespace {

void print_usage() {
  std::cout
      << "Usage: poisson_demo --problem <name> [options]\n\n"
         "Problems:\n"
         "  poisson1d   Finite-volume 1D Poisson with Dirichlet BC\n"
         "  sor2d       Finite-volume 2D Poisson with SOR red-black\n"
         "  spectral2d  Spectral 2D Poisson via DST (requires FFTW3)\n"
         "  amr         Quadtree AMR 2D with Gaussian charge\n\n"
         "Options (1D):\n"
         "  --N <int>      Number of grid nodes (default 50)\n"
         "  --L <double>   Domain length (default 1.0)\n"
         "  --uL <double>  V(0)  (default 10.0)\n"
         "  --uR <double>  V(L)  (default  0.0)\n\n"
         "Options (2D/AMR):\n"
         "  --N <int>      Grid size per direction\n"
         "  --Nmin <int>   AMR min level (default 4)\n"
         "  --Nmax <int>   AMR max level (default 7)\n"
         "  --sigma <dbl>  Gaussian charge width (default 0.04)\n";
}

int run_poisson1d(int N, double L, double uL, double uR) {
  poisson::Grid1D grid(L, N);
  Eigen::VectorXd rho = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd V = poisson::fv::solve_poisson_1d(rho, uL, uR, grid);

  std::cout << "# poisson1d  N=" << N << "  L=" << L
            << "  uL=" << uL << "  uR=" << uR << '\n'
            << "# i\tx\tV\n";
  for (int i = 0; i < N; ++i) {
    std::cout << i << '\t' << grid.x(i) << '\t' << V(i) << '\n';
  }
  return 0;
}

int run_sor2d(int N, double uL, double uR) {
  poisson::Grid2D grid(1.0, 1.0, N, N);
  poisson::fv::Solver2D solver(grid, 1.0, uL, uR);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);

  const auto r = solver.solve(V, rho);
  std::cout << "# sor2d  N=" << N << "x" << N << "  uL=" << uL << "  uR=" << uR
            << "  iter=" << r.iterations << "  residual=" << r.residual << '\n'
            << "# Coupe y = Ly/2, i\tx\tV\n";
  const int jmid = N / 2;
  for (int i = 0; i < N; ++i) {
    std::cout << i << '\t' << (i + 0.5) * grid.dx() << '\t' << V(i, jmid) << '\n';
  }
  return 0;
}

int run_spectral2d(int N) {
#if defined(POISSON_HAVE_FFTW3)
  poisson::spectral::DSTSolver2D solver(N, N, 1.0, 1.0);
  // rho = sin(pi x / L) * sin(pi y / L) discretised -> V = rho / (2 (pi/L)^2).
  Eigen::MatrixXd rho(N, N);
  const double h = 1.0 / (N + 1);
  for (int j = 1; j <= N; ++j) {
    for (int i = 1; i <= N; ++i) {
      rho(i - 1, j - 1) = std::sin(3.14159265358979323846 * i * h) *
                          std::sin(3.14159265358979323846 * j * h);
    }
  }
  const Eigen::MatrixXd V = solver.solve(rho);
  std::cout << "# spectral2d  N=" << N << "x" << N << "  V_max=" << V.maxCoeff()
            << "  V_min=" << V.minCoeff() << '\n';
  return 0;
#else
  (void)N;
  std::cerr << "Built without FFTW3; spectral2d is unavailable.\n";
  return 3;
#endif
}

int run_amr(int level_min, int level_max, double sigma) {
  poisson::amr::Quadtree tree(1.0, level_min);
  const double xc = 0.5, yc = 0.5;
  auto predicate = [=](poisson::amr::CellKey k) {
    const uint8_t lv = poisson::amr::level_of(k);
    if (lv >= level_max) return false;
    const uint32_t i = poisson::amr::i_of(k);
    const uint32_t j = poisson::amr::j_of(k);
    const double h = 1.0 / (1u << lv);
    const double x = (i + 0.5) * h, y = (j + 0.5) * h;
    const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    return r2 < 16.0 * sigma * sigma && h > 0.25 * sigma;
  };
  auto rho_fn = [=](double x, double y) {
    const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    return std::exp(-r2 / (sigma * sigma));
  };
  tree.build(predicate, static_cast<uint8_t>(level_max), rho_fn);

  auto arr = poisson::amr::extract_arrays(tree);
  const auto rep = poisson::amr::sor(
      arr, {.omega = 1.85, .tol = 1e-7, .max_iter = 10'000, .eps0 = 1.0});

  std::cout << "# amr  level_min=" << level_min << "  level_max=" << level_max
            << "  sigma=" << sigma
            << "  N_leaves=" << tree.num_leaves()
            << "  iter=" << rep.iterations
            << "  residual=" << rep.residual << '\n'
            << "# V min/max = " << arr.V.minCoeff() << " / " << arr.V.maxCoeff()
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string problem;
  int N = 50, level_min = 4, level_max = 7;
  double L = 1.0, uL = 10.0, uR = 0.0, sigma = 0.04;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
    else if (arg == "--problem") problem = next("--problem");
    else if (arg == "--N")       N     = std::atoi(next("--N"));
    else if (arg == "--L")       L     = std::atof(next("--L"));
    else if (arg == "--uL")      uL    = std::atof(next("--uL"));
    else if (arg == "--uR")      uR    = std::atof(next("--uR"));
    else if (arg == "--Nmin")    level_min = std::atoi(next("--Nmin"));
    else if (arg == "--Nmax")    level_max = std::atoi(next("--Nmax"));
    else if (arg == "--sigma")   sigma = std::atof(next("--sigma"));
    else {
      std::cerr << "Unknown argument: " << arg << '\n';
      print_usage();
      return 2;
    }
  }

  if (problem.empty())         { print_usage(); return 1; }
  if (problem == "poisson1d")  return run_poisson1d(N, L, uL, uR);
  if (problem == "sor2d")      return run_sor2d(N, uL, uR);
  if (problem == "spectral2d") return run_spectral2d(N);
  if (problem == "amr")        return run_amr(level_min, level_max, sigma);
  std::cerr << "Unknown problem: " << problem << '\n';
  print_usage();
  return 2;
}
