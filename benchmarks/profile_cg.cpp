// Profile harness: run CG in a hot loop long enough to be sampled by
// macOS `sample` or xctrace. Outputs nothing — purely for profiling.

#include <Eigen/Core>
#include "poisson/core/grid.hpp"
#include "poisson/iter/poisson_cg.hpp"

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 512;
  const int runs = argc > 2 ? std::atoi(argv[2]) : 10;
  poisson::Grid2D grid(1.0, 1.0, N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd V(N, N);
  for (int r = 0; r < runs; ++r) {
    V.setZero();
    poisson::iter::solve_poisson_cg(
        V, rho, grid, 1.0, 0.0, 10.0,
        {.tol = 1e-10, .max_iter = 5000},
        /*use_preconditioner=*/false);
  }
  return 0;
}
