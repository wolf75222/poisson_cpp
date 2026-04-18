#include "poisson/amr/solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace poisson::amr {

namespace {

constexpr std::array<Direction, 4> kDirections = {
    Direction::N, Direction::S, Direction::E, Direction::W};

}  // namespace

AMRArrays extract_arrays(const Quadtree& tree) {
  const auto& leaves_map = tree.leaves();
  const Eigen::Index N = static_cast<Eigen::Index>(leaves_map.size());

  AMRArrays a;
  a.keys.reserve(N);
  a.V.resize(N);
  a.rho.resize(N);
  a.h.resize(N);
  a.Vc = Eigen::VectorXd::Zero(N);
  a.nb0.resize(N, 4);   a.nb0.setConstant(-1);
  a.nb1.resize(N, 4);   a.nb1.setConstant(-1);
  a.w0 = Eigen::Matrix<double, Eigen::Dynamic, 4>::Zero(N, 4);
  a.w1 = Eigen::Matrix<double, Eigen::Dynamic, 4>::Zero(N, 4);

  std::unordered_map<CellKey, int64_t> key_to_idx;
  key_to_idx.reserve(static_cast<std::size_t>(N));
  Eigen::Index n = 0;
  for (const auto& [key, cell] : leaves_map) {
    a.keys.push_back(key);
    a.V(n) = cell.V;
    a.rho(n) = cell.rho;
    a.h(n) = tree.cell_size(level_of(key));
    key_to_idx[key] = static_cast<int64_t>(n);
    ++n;
  }

  for (Eigen::Index i = 0; i < N; ++i) {
    const CellKey key = a.keys[static_cast<std::size_t>(i)];
    const uint8_t lv = level_of(key);
    for (int d = 0; d < 4; ++d) {
      const auto neighs = tree.neighbour_leaves(key, kDirections[d]);
      double wdiag = 0.0;
      if (neighs.empty()) {
        // Dirichlet boundary V = 0.
        wdiag = 2.0;
      } else if (neighs.size() == 1) {
        const CellKey nk = neighs[0];
        a.nb0(i, d) = key_to_idx.at(nk);
        if (level_of(nk) == lv) {
          wdiag = 1.0;   a.w0(i, d) = 1.0;
        } else {
          // Coarser neighbour: diag += 2/3, off = 2/3.
          wdiag = 2.0 / 3.0;   a.w0(i, d) = 2.0 / 3.0;
        }
      } else {
        // Two finer neighbours: diag += 4/3, each off = 2/3.
        a.nb0(i, d) = key_to_idx.at(neighs[0]);
        a.nb1(i, d) = key_to_idx.at(neighs[1]);
        wdiag = 4.0 / 3.0;
        a.w0(i, d) = 2.0 / 3.0;
        a.w1(i, d) = 2.0 / 3.0;
      }
      a.Vc(i) += wdiag;
    }
  }
  return a;
}

void writeback(Quadtree& tree,
               const std::vector<CellKey>& keys,
               Eigen::Ref<const Eigen::VectorXd> V) {
  if (V.size() != static_cast<Eigen::Index>(keys.size())) {
    throw std::invalid_argument("writeback: V size must match keys");
  }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    tree.at(keys[i]).V = V(static_cast<Eigen::Index>(i));
  }
}

SORReport sor(AMRArrays& a, SORParams p) {
  const Eigen::Index N = static_cast<Eigen::Index>(a.keys.size());
  if (!(p.eps0 > 0.0)) throw std::invalid_argument("sor: eps0 must be > 0");
  if (!(p.omega > 0.0 && p.omega < 2.0))
    throw std::invalid_argument("sor: omega must be in (0, 2)");

  // Precompute per-cell scalars that are invariant across iterations:
  //   rhs(i)    = h(i)^2 rho(i) / eps0       (source term in V units)
  //   Vc_inv(i) = 1 / Vc(i)                  (avoid division in hot loop)
  // This turns the inner loop's (2 div + 1 mul) into (2 mul + 0 div),
  // which is ~3x cheaper on modern CPUs.
  const Eigen::VectorXd rhs    = a.h.array().square() * a.rho.array() / p.eps0;
  const Eigen::VectorXd Vc_inv = a.Vc.array().inverse();
  const double omega = p.omega;
  const double one_minus_omega = 1.0 - omega;

  double max_diff = 0.0;
  int iter = 0;
  for (iter = 0; iter < p.max_iter; ++iter) {
    max_diff = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
      double s = 0.0;
      for (int d = 0; d < 4; ++d) {
        const int64_t n0 = a.nb0(i, d);
        if (n0 >= 0) s += a.w0(i, d) * a.V(n0);
        const int64_t n1 = a.nb1(i, d);
        if (n1 >= 0) s += a.w1(i, d) * a.V(n1);
      }
      const double V_gs = (s + rhs(i)) * Vc_inv(i);
      const double V_i  = a.V(i);
      const double V_new = one_minus_omega * V_i + omega * V_gs;
      const double diff = std::abs(V_new - V_i);
      if (diff > max_diff) max_diff = diff;
      a.V(i) = V_new;
    }
    if (max_diff < p.tol) {
      ++iter;
      break;
    }
  }
  return {iter, max_diff};
}

Eigen::VectorXd residual(const AMRArrays& a, double eps0) {
  const Eigen::Index N = static_cast<Eigen::Index>(a.keys.size());
  Eigen::VectorXd r(N);
  for (Eigen::Index i = 0; i < N; ++i) {
    double s = 0.0;
    for (int d = 0; d < 4; ++d) {
      const int64_t n0 = a.nb0(i, d);
      if (n0 >= 0) s += a.w0(i, d) * a.V(n0);
      const int64_t n1 = a.nb1(i, d);
      if (n1 >= 0) s += a.w1(i, d) * a.V(n1);
    }
    r(i) = s + a.h(i) * a.h(i) * a.rho(i) / eps0 - a.Vc(i) * a.V(i);
  }
  return r;
}

}  // namespace poisson::amr
