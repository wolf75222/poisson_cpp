#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "poisson/amr/morton.hpp"

namespace poisson::amr {

/// Per-leaf scientific data stored in the quadtree.
struct Cell {
  double V = 0.0;
  double rho = 0.0;
};

/// Cell-centered quadtree on the square domain [0, L] x [0, L], with
/// Morton-encoded cell keys. Only leaves are stored.
class Quadtree {
 public:
  /// Build a uniform grid of size 2^level_min at level `level_min`, with
  /// all cells initialised to `Cell{}` (V = rho = 0).
  Quadtree(double L, int level_min);

  /// Side length of a cell at the given level.
  [[nodiscard]] double cell_size(uint8_t level) const noexcept;

  /// Geometric center of the given cell key (regardless of it being a leaf).
  [[nodiscard]] std::pair<double, double> cell_center(
      CellKey key) const noexcept;

  /// Subdivide the leaf `key` into its 4 children. Precondition: `key` must
  /// be a leaf (present in the internal map).
  void refine(CellKey key);

  /// Refine cells satisfying `predicate(key)` until stable, then enforce
  /// the 2:1 balance constraint. Finally, evaluate `rho_func(x, y)` at each
  /// leaf center and store it in `Cell::rho`.
  ///
  /// \param predicate returns true if the leaf should be subdivided.
  /// \param level_max stop refining beyond this level.
  void build(const std::function<bool(CellKey)>& predicate,
             uint8_t level_max,
             const std::function<double(double, double)>& rho_func);

  /// Enforce the 2:1 balance: any leaf whose direct-neighbour side touches a
  /// leaf more than one level coarser triggers subdivision of that coarser
  /// leaf, iterated until stable.
  void balance_2to1();

  /// List of leaf neighbours of `key` on face `dir`.
  /// Returns 0 (boundary), 1 (same level or coarser), or 2 (two finer) keys.
  [[nodiscard]] std::vector<CellKey> neighbour_leaves(
      CellKey key, Direction dir) const;

  /// Read-only access to a leaf's data (throws if not a leaf).
  [[nodiscard]] const Cell& at(CellKey key) const { return leaves_.at(key); }
  [[nodiscard]] Cell&       at(CellKey key)       { return leaves_.at(key); }

  /// Convenience: is this cell currently a leaf ?
  [[nodiscard]] bool is_leaf(CellKey key) const noexcept {
    return leaves_.count(key) > 0;
  }

  [[nodiscard]] std::size_t num_leaves() const noexcept {
    return leaves_.size();
  }

  /// Direct access to the internal map for iteration, for example:
  ///   `for (const auto& [key, cell] : tree.leaves()) { ... }`.
  [[nodiscard]] const std::unordered_map<CellKey, Cell>& leaves()
      const noexcept { return leaves_; }
  [[nodiscard]] std::unordered_map<CellKey, Cell>& leaves() noexcept {
    return leaves_;
  }

  [[nodiscard]] double  L()         const noexcept { return L_; }
  [[nodiscard]] uint8_t level_min() const noexcept { return level_min_; }

 private:
  double L_;
  uint8_t level_min_;
  std::unordered_map<CellKey, Cell> leaves_;
};

}  // namespace poisson::amr
