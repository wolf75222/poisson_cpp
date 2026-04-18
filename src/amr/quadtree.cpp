#include "poisson/amr/quadtree.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace poisson::amr {

namespace {

// Two children of `tkey` that touch the face opposite to `dir`
// (i.e. the face shared with a coarser neighbour on our side).
std::array<CellKey, 2> face_children(CellKey tkey, Direction dir) noexcept {
  const uint8_t lv = level_of(tkey);
  const uint32_t ti = i_of(tkey), tj = j_of(tkey);
  const uint8_t lvp = static_cast<uint8_t>(lv + 1);
  switch (dir) {
    case Direction::N:
      return {make_key(lvp, 2 * ti,     2 * tj),
              make_key(lvp, 2 * ti + 1, 2 * tj)};
    case Direction::S:
      return {make_key(lvp, 2 * ti,     2 * tj + 1),
              make_key(lvp, 2 * ti + 1, 2 * tj + 1)};
    case Direction::E:
      return {make_key(lvp, 2 * ti,     2 * tj),
              make_key(lvp, 2 * ti,     2 * tj + 1)};
    case Direction::W:
      return {make_key(lvp, 2 * ti + 1, 2 * tj),
              make_key(lvp, 2 * ti + 1, 2 * tj + 1)};
  }
  return {};  // unreachable
}

}  // namespace

Quadtree::Quadtree(double L, int level_min)
    : L_(L), level_min_(static_cast<uint8_t>(level_min)) {
  if (!(L > 0.0)) throw std::invalid_argument("Quadtree: L must be > 0");
  if (level_min < 0 || level_min > MAX_LEVEL - 1)
    throw std::invalid_argument("Quadtree: level_min out of range");

  const uint32_t N = 1u << level_min;
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      leaves_.emplace(make_key(static_cast<uint8_t>(level_min), i, j), Cell{});
    }
  }
}

double Quadtree::cell_size(uint8_t level) const noexcept {
  return L_ / static_cast<double>(1u << level);
}

std::pair<double, double> Quadtree::cell_center(CellKey key) const noexcept {
  const double h = cell_size(level_of(key));
  return {(i_of(key) + 0.5) * h, (j_of(key) + 0.5) * h};
}

void Quadtree::refine(CellKey key) {
  auto it = leaves_.find(key);
  if (it == leaves_.end()) {
    throw std::invalid_argument("refine: key is not a leaf");
  }
  leaves_.erase(it);
  for (const auto& child : children_of(key)) {
    leaves_.emplace(child, Cell{});
  }
}

void Quadtree::build(const std::function<bool(CellKey)>& predicate,
                     uint8_t level_max,
                     const std::function<double(double, double)>& rho_func) {
  // Iterative refinement until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<CellKey> to_refine;
    to_refine.reserve(leaves_.size());
    for (const auto& [k, _] : leaves_) {
      if (level_of(k) < level_max && predicate(k)) to_refine.push_back(k);
    }
    for (const CellKey k : to_refine) {
      if (leaves_.count(k)) {  // may have been removed meanwhile
        refine(k);
        changed = true;
      }
    }
  }

  balance_2to1();

  // Evaluate rho at each leaf center.
  for (auto& [key, cell] : leaves_) {
    const auto [x, y] = cell_center(key);
    cell.rho = rho_func(x, y);
  }
}

void Quadtree::balance_2to1() {
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<CellKey> leaves_snapshot;
    leaves_snapshot.reserve(leaves_.size());
    for (const auto& [k, _] : leaves_) leaves_snapshot.push_back(k);

    for (const CellKey key : leaves_snapshot) {
      if (!leaves_.count(key)) continue;  // already subdivided
      const uint8_t lv = level_of(key);
      if (lv == 0) continue;

      for (auto dir : {Direction::N, Direction::S, Direction::E, Direction::W}) {
        auto tkey = neighbour_same_level(key, dir);
        if (!tkey) continue;
        // Walk up parents of tkey until we find a leaf.
        CellKey anc = parent_of(*tkey);
        while (anc != 0 && !leaves_.count(anc)) {
          anc = parent_of(anc);
        }
        if (anc != 0 && leaves_.count(anc) && level_of(anc) < lv - 1) {
          refine(anc);
          changed = true;
        }
      }
    }
  }
}

std::vector<CellKey>
Quadtree::neighbour_leaves(CellKey key, Direction dir) const {
  const auto tkey_opt = neighbour_same_level(key, dir);
  if (!tkey_opt) return {};          // domain boundary

  const CellKey tkey = *tkey_opt;
  if (leaves_.count(tkey)) return {tkey};   // same-level leaf

  // Coarser: walk up parents of tkey until a leaf is found.
  CellKey anc = parent_of(tkey);
  while (anc != 0) {
    if (leaves_.count(anc)) return {anc};
    anc = parent_of(anc);
  }

  // Finer: in a 2:1-balanced tree, the two children of tkey touching our
  // face are leaves themselves.
  std::vector<CellKey> out;
  for (const CellKey c : face_children(tkey, dir)) {
    if (leaves_.count(c)) out.push_back(c);
  }
  return out;
}

}  // namespace poisson::amr
