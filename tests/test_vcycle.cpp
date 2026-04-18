#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "poisson/amr/quadtree.hpp"
#include "poisson/amr/solver.hpp"
#include "poisson/mg/vcycle.hpp"

using namespace poisson;

TEST_CASE("Uniform V-cycle converges on a Gaussian source", "[mg][uniform]") {
  const int N = 32;
  const double L = 1.0, h = L / N;
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho(N, N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const double x = (i + 0.5) * h - 0.5;
      const double y = (j + 0.5) * h - 0.5;
      rho(i, j) = std::exp(-(x * x + y * y) / 0.01);
    }
  }

  double r_prev = (rho - mg::laplacian_fv(V, h)).cwiseAbs().maxCoeff();
  for (int cycle = 0; cycle < 6; ++cycle) {
    V = mg::vcycle_uniform(std::move(V), rho, h);
    const double r_now = (rho - mg::laplacian_fv(V, h)).cwiseAbs().maxCoeff();
    REQUIRE(r_now < r_prev);   // monotone decrease per V-cycle
    r_prev = r_now;
  }
  REQUIRE(r_prev < 1e-5);
}

TEST_CASE("Composite V-cycle on an unrefined AMR tree (1-level edge case)",
          "[mg][amr]") {
  // When no cell is refined, the AMR mesh is just a uniform grid at level_min.
  // The composite V-cycle must still converge, reducing to essentially a
  // uniform solve. This exercises the shift = 0 branch in the restriction.
  amr::Quadtree tree(1.0, 3);   // uniform 8x8, no refinement
  const double hcoarse = tree.cell_size(3);
  for (auto& [key, cell] : tree.leaves()) {
    const auto [x, y] = tree.cell_center(key);
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    cell.rho = std::exp(-r2 / 0.02);
    (void)hcoarse;
  }
  auto arr = amr::extract_arrays(tree);
  const double r0 = amr::residual(arr).cwiseAbs().maxCoeff();
  mg::CompositeParams p{.n_pre = 3, .n_post = 3, .n_coarse_cycles = 3,
                         .omega = 1.85, .eps0 = 1.0};
  for (int k = 0; k < 5; ++k) {
    mg::vcycle_amr_composite(arr, tree, p);
  }
  const double r1 = amr::residual(arr).cwiseAbs().maxCoeff();
  REQUIRE(r1 < r0);
  REQUIRE(r1 < 0.1 * r0);
}

TEST_CASE("Composite V-cycle reduces the AMR residual", "[mg][amr]") {
  // Build a refined AMR tree around a Gaussian.
  amr::Quadtree tree(1.0, 3);
  auto pred = [](amr::CellKey k) {
    const uint8_t lv = amr::level_of(k);
    if (lv >= 5) return false;
    const uint32_t i = amr::i_of(k), j = amr::j_of(k);
    const double h = 1.0 / (1u << lv);
    const double x = (i + 0.5) * h, y = (j + 0.5) * h;
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return r2 < 0.04;   // radius ~0.2 around center
  };
  auto rho_fn = [](double x, double y) {
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return std::exp(-r2 / 0.01);
  };
  tree.build(pred, /*level_max=*/5, rho_fn);

  auto arr = amr::extract_arrays(tree);
  double r0 = amr::residual(arr).cwiseAbs().maxCoeff();
  mg::CompositeParams p{.n_pre = 3, .n_post = 3, .n_coarse_cycles = 4,
                         .omega = 1.85, .eps0 = 1.0};
  for (int k = 0; k < 6; ++k) {
    mg::vcycle_amr_composite(arr, tree, p);
  }
  const double r1 = amr::residual(arr).cwiseAbs().maxCoeff();
  REQUIRE(r1 < r0);
  // With a non-Galerkin coarse operator the per-cycle reduction factor is
  // about 0.7 (see TP5 analysis). After 6 cycles we expect ~0.1 * r0.
  REQUIRE(r1 < 0.2 * r0);
}
