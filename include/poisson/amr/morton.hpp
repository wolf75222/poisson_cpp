#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace poisson::amr {

/// Maximum per-coordinate level supported by the 64-bit Morton encoding.
/// With 8 bits of level and 28 bits each for `i` and `j` (interleaved), we
/// can represent grids up to 2^28 x 2^28, far beyond any practical need.
inline constexpr int MAX_LEVEL = 28;

/// Morton-encoded cell key: `(level, i, j)` packed in a 64-bit integer.
///
/// Layout:
///   - bits 56..63: `level` (0..255)
///   - bits  0..55: interleaved `(i, j)` with `i` on even bits, `j` on odd
///
/// The interleaving means that the Morton code preserves spatial locality
/// (Z-order), which is useful for cache-friendly traversals, but the
/// AMR code here does not exploit that directly and simply uses the key as
/// a hash map identifier.
using CellKey = uint64_t;

/// Interleave the lower 28 bits of `x` onto even bit positions: bit k -> 2k.
constexpr uint64_t interleave28(uint32_t x) noexcept {
  uint64_t r = 0;
  for (int k = 0; k < MAX_LEVEL; ++k) {
    r |= (static_cast<uint64_t>((x >> k) & 1u)) << (2 * k);
  }
  return r;
}

/// Inverse of `interleave28` applied to even bits of a 56-bit Morton payload.
constexpr uint32_t deinterleave28_even(uint64_t bits) noexcept {
  uint32_t r = 0;
  for (int k = 0; k < MAX_LEVEL; ++k) {
    r |= static_cast<uint32_t>((bits >> (2 * k)) & 1u) << k;
  }
  return r;
}

/// Encode (level, i, j) into a Morton-based CellKey.
constexpr CellKey make_key(uint8_t level, uint32_t i, uint32_t j) noexcept {
  const uint64_t bits = interleave28(i) | (interleave28(j) << 1);
  return (static_cast<uint64_t>(level) << 56) | (bits & 0x00FF'FFFF'FFFF'FFFFULL);
}

constexpr uint8_t level_of(CellKey k) noexcept {
  return static_cast<uint8_t>(k >> 56);
}

constexpr uint32_t i_of(CellKey k) noexcept {
  return deinterleave28_even(k & 0x00FF'FFFF'FFFF'FFFFULL);
}

constexpr uint32_t j_of(CellKey k) noexcept {
  return deinterleave28_even((k & 0x00FF'FFFF'FFFF'FFFFULL) >> 1);
}

/// Parent of `k` (one level coarser), or 0 if `k` is at level 0.
constexpr CellKey parent_of(CellKey k) noexcept {
  const uint8_t lv = level_of(k);
  if (lv == 0) return 0;
  return make_key(lv - 1, i_of(k) >> 1, j_of(k) >> 1);
}

/// Four children of `k` at level+1, in fixed order (SW, SE, NW, NE).
constexpr std::array<CellKey, 4> children_of(CellKey k) noexcept {
  const uint8_t lv = level_of(k);
  const uint32_t i = i_of(k), j = j_of(k);
  return {
      make_key(static_cast<uint8_t>(lv + 1), 2 * i,     2 * j),
      make_key(static_cast<uint8_t>(lv + 1), 2 * i + 1, 2 * j),
      make_key(static_cast<uint8_t>(lv + 1), 2 * i,     2 * j + 1),
      make_key(static_cast<uint8_t>(lv + 1), 2 * i + 1, 2 * j + 1),
  };
}

/// Geometric direction across a face of a quadtree cell.
enum class Direction : uint8_t { N, S, E, W };

/// Same-level neighbour in the given direction, or nullopt if outside the
/// 2^level x 2^level grid of the root domain.
constexpr std::optional<CellKey>
neighbour_same_level(CellKey k, Direction d) noexcept {
  const uint8_t lv = level_of(k);
  const uint32_t N = (lv >= MAX_LEVEL) ? 0 : (1u << lv);
  uint32_t i = i_of(k), j = j_of(k);
  switch (d) {
    case Direction::N: if (j + 1 >= N) return std::nullopt; ++j; break;
    case Direction::S: if (j == 0)     return std::nullopt; --j; break;
    case Direction::E: if (i + 1 >= N) return std::nullopt; ++i; break;
    case Direction::W: if (i == 0)     return std::nullopt; --i; break;
  }
  return make_key(lv, i, j);
}

}  // namespace poisson::amr
