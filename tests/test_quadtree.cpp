#include <catch2/catch_test_macros.hpp>

#include <set>

#include "poisson/amr/quadtree.hpp"

using namespace poisson::amr;

TEST_CASE("Quadtree initial uniform grid has 2^(2 level_min) leaves",
          "[amr][quadtree]") {
  for (int lv : {0, 1, 3, 5}) {
    Quadtree tree(1.0, lv);
    REQUIRE(tree.num_leaves() == static_cast<std::size_t>(1u << (2 * lv)));
  }
}

TEST_CASE("Quadtree::refine turns one leaf into four", "[amr][quadtree]") {
  Quadtree tree(1.0, 2);
  const CellKey k = make_key(2, 1, 1);
  REQUIRE(tree.is_leaf(k));
  const std::size_t before = tree.num_leaves();
  tree.refine(k);
  REQUIRE_FALSE(tree.is_leaf(k));
  REQUIRE(tree.num_leaves() == before + 3);
  for (auto child : children_of(k)) REQUIRE(tree.is_leaf(child));
}

TEST_CASE("Quadtree: build + balance_2to1 maintains 2:1 invariant",
          "[amr][quadtree]") {
  // Refine a single deep "hot spot" to force a balance cascade.
  Quadtree tree(1.0, 3);
  auto predicate = [](CellKey k) {
    // Refine cells near (0.1, 0.1), requires many levels there.
    const uint32_t i = i_of(k), j = j_of(k);
    const uint8_t lv = level_of(k);
    const double h = 1.0 / (1u << lv);
    const double x = (i + 0.5) * h, y = (j + 0.5) * h;
    return (x < 0.2 && y < 0.2) && lv < 6;
  };
  auto rho = [](double, double) { return 0.0; };
  tree.build(predicate, /*level_max=*/6, rho);

  // Verify the 2:1 invariant: for every leaf, every face-adjacent leaf is
  // at most one level apart.
  for (const auto& [key, _] : tree.leaves()) {
    const uint8_t lv = level_of(key);
    for (auto dir : {Direction::N, Direction::S, Direction::E, Direction::W}) {
      auto neighs = tree.neighbour_leaves(key, dir);
      for (auto nk : neighs) {
        const uint8_t nlv = level_of(nk);
        const int diff = std::abs(static_cast<int>(lv) - static_cast<int>(nlv));
        REQUIRE(diff <= 1);
      }
    }
  }
}

TEST_CASE("Quadtree::neighbour_leaves covers all 3 cases", "[amr][quadtree]") {
  Quadtree tree(1.0, 2);
  // Refine cell (2, 0, 0) only. Its east neighbour (2, 1, 0) stays same-level.
  tree.refine(make_key(2, 0, 0));
  // Now (2, 0, 0) has 4 children. Its east face has 2 finer leaves.
  auto e = tree.neighbour_leaves(make_key(2, 1, 0), Direction::W);
  REQUIRE(e.size() == 2);
  // Their levels are 3.
  for (auto k : e) REQUIRE(level_of(k) == 3);

  // The same-level case: from (2, 1, 0) looking east.
  auto same = tree.neighbour_leaves(make_key(2, 1, 0), Direction::E);
  REQUIRE(same.size() == 1);
  REQUIRE(level_of(same[0]) == 2);

  // Coarser neighbour: from a deep child at (3, 0, 0) looking east we hit
  // (3, 1, 0), which IS a leaf of the refined block (same level).
  auto coarse = tree.neighbour_leaves(make_key(3, 0, 0), Direction::N);
  REQUIRE(coarse.size() == 1);
  REQUIRE(level_of(coarse[0]) == 3);
}
