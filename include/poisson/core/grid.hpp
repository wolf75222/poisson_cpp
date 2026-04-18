#pragma once

#include <cstddef>
#include <stdexcept>

namespace poisson {

/// Uniform 1D mesh with node-centered convention.
///
/// The grid has `N` nodes at positions `x_i = i * dx` for i = 0..N-1, with
/// `x[0] = 0` and `x[N-1] = L`. The first and last nodes carry the Dirichlet
/// boundary values directly (FD-style). This matches the Python notebooks
/// `TP1_Poisson_1D.ipynb` and `TP2_Poisson_1D.ipynb`.
struct Grid1D {
  double L;      ///< Domain length.
  int N;         ///< Number of nodes (>= 2).

  constexpr Grid1D(double length, int n) : L(length), N(n) {
    if (n < 2) throw std::invalid_argument("Grid1D requires N >= 2");
    if (!(length > 0.0)) throw std::invalid_argument("Grid1D requires L > 0");
  }

  /// Spacing between two consecutive nodes.
  [[nodiscard]] constexpr double dx() const noexcept {
    return L / static_cast<double>(N - 1);
  }

  /// Position of node `i` (0 <= i < N).
  [[nodiscard]] constexpr double x(int i) const noexcept {
    return static_cast<double>(i) * dx();
  }
};

/// Uniform 2D mesh, cell-centered convention (used by `Solver2D` and `AMR`).
///
/// Cells span the domain `[0, Lx] x [0, Ly]` with `Nx * Ny` cells of size
/// `dx = Lx/Nx, dy = Ly/Ny`. Cell `(i, j)` has its center at
/// `((i + 0.5) dx, (j + 0.5) dy)` and Dirichlet values are imposed on the
/// outer faces. This matches the notebooks `TP3_Poisson_2D.ipynb` and
/// `TP5_AMR_Poisson_2D.ipynb`.
struct Grid2D {
  double Lx, Ly;
  int Nx, Ny;

  constexpr Grid2D(double lx, double ly, int nx, int ny)
      : Lx(lx), Ly(ly), Nx(nx), Ny(ny) {
    if (nx < 1 || ny < 1) throw std::invalid_argument("Grid2D requires Nx,Ny >= 1");
  }

  [[nodiscard]] constexpr double dx() const noexcept {
    return Lx / static_cast<double>(Nx);
  }
  [[nodiscard]] constexpr double dy() const noexcept {
    return Ly / static_cast<double>(Ny);
  }
};

}  // namespace poisson
