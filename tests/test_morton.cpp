#include <catch2/catch_test_macros.hpp>

#include "poisson/amr/morton.hpp"

using namespace poisson::amr;

TEST_CASE("Morton encode/decode round-trip", "[amr][morton]") {
  for (int lv : {0, 1, 5, 10, 20}) {
    const uint32_t N = (lv > 0) ? (1u << lv) : 1u;
    for (uint32_t i : {0u, N / 2, N - 1}) {
      for (uint32_t j : {0u, N / 3, N - 1}) {
        const CellKey k = make_key(static_cast<uint8_t>(lv), i, j);
        REQUIRE(level_of(k) == lv);
        REQUIRE(i_of(k) == i);
        REQUIRE(j_of(k) == j);
      }
    }
  }
}

TEST_CASE("Morton parent / children / neighbours", "[amr][morton]") {
  const CellKey root = make_key(2, 1, 2);  // at level 2, index (1, 2)
  const auto kids = children_of(root);
  REQUIRE(kids.size() == 4);
  for (auto& kid : kids) {
    REQUIRE(level_of(kid) == 3);
    REQUIRE(parent_of(kid) == root);
  }

  // SW child is (2*1, 2*2) at level 3 = (2, 4).
  REQUIRE(i_of(kids[0]) == 2);
  REQUIRE(j_of(kids[0]) == 4);

  const CellKey c = make_key(3, 4, 4);
  REQUIRE(neighbour_same_level(c, Direction::E).value() == make_key(3, 5, 4));
  REQUIRE(neighbour_same_level(c, Direction::N).value() == make_key(3, 4, 5));
  REQUIRE(neighbour_same_level(c, Direction::W).value() == make_key(3, 3, 4));
  REQUIRE(neighbour_same_level(c, Direction::S).value() == make_key(3, 4, 3));
}

TEST_CASE("Morton neighbour returns nullopt at domain boundary",
          "[amr][morton]") {
  const CellKey left_edge = make_key(3, 0, 4);
  REQUIRE_FALSE(neighbour_same_level(left_edge, Direction::W).has_value());
  const CellKey right_edge = make_key(3, 7, 4);  // N = 2^3 = 8
  REQUIRE_FALSE(neighbour_same_level(right_edge, Direction::E).has_value());
}
