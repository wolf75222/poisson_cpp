// Demo binary for the poisson_cpp library.
//
// Dispatches to individual solvers via a `--problem` flag. This is a
// companion harness for the library defined in `include/poisson/*`; the
// value is in the library, not in this CLI.
//
//   ./poisson_demo --problem poisson1d --N 50
//   ./poisson_demo --help

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"

namespace {

void print_usage() {
  std::cout << "Usage: poisson_demo --problem <name> [options]\n"
               "\n"
               "Problems:\n"
               "  poisson1d   Finite-volume 1D Poisson with Dirichlet BC\n"
               "\n"
               "Options:\n"
               "  --N <int>       Number of grid nodes (default 50)\n"
               "  --L <double>    Domain length (default 1.0)\n"
               "  --uL <double>   V(0)  (default 10.0)\n"
               "  --uR <double>   V(L)  (default  0.0)\n"
               "  --help          Show this help\n";
}

int run_poisson1d(int N, double L, double uL, double uR) {
  poisson::Grid1D grid(L, N);
  Eigen::VectorXd rho = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd V = poisson::fv::solve_poisson_1d(rho, uL, uR, grid);

  std::cout << "# poisson1d  N=" << N << "  L=" << L
            << "  uL=" << uL << "  uR=" << uR << "\n";
  std::cout << "# i\tx\tV\n";
  for (int i = 0; i < N; ++i) {
    std::cout << i << '\t' << grid.x(i) << '\t' << V(i) << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string problem;
  int N = 50;
  double L = 1.0;
  double uL = 10.0;
  double uR = 0.0;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--problem") {
      problem = next("--problem");
    } else if (arg == "--N") {
      N = std::atoi(next("--N"));
    } else if (arg == "--L") {
      L = std::atof(next("--L"));
    } else if (arg == "--uL") {
      uL = std::atof(next("--uL"));
    } else if (arg == "--uR") {
      uR = std::atof(next("--uR"));
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      print_usage();
      return 2;
    }
  }

  if (problem.empty()) {
    print_usage();
    return 1;
  }
  if (problem == "poisson1d") {
    return run_poisson1d(N, L, uL, uR);
  }
  std::cerr << "Unknown problem: " << problem << '\n';
  print_usage();
  return 2;
}
