# poisson_cpp

![banner](docs/figures/banner.png)

*Dipôle gaussien (charges ±1 à σ = 0.03) résolu par le solveur AMR :
1 756 feuilles quadtree, SOR convergé à 10⁻⁸ en 204 itérations, V ≈ ±1.1×10⁻³.
Script : [`python/make_banner.py`](python/make_banner.py).*

---

Modern C++20 library of Poisson solvers: Thomas tridiagonal, FV red-black
SOR, spectral (DST-I via FFTW), and quadtree AMR with 2-grid composite
multigrid.

Port of the Python notebooks in
[`CourseOnPoisson/notebooks/`](https://github.com/alvarezlaguna/AMR_Poisson)
with a library-oriented API.

## Status

- **57/57 Catch2 tests pass** (unit + invariants + conservation + benchmarks).
- Optional OpenMP acceleration for large grids (`POISSON_USE_OPENMP=ON`).
- Reference-quality physical invariants enforced at machine precision
  (Green's reciprocity, energy identity, D-continuity across dielectric
  interfaces).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module layout
and the stencil / BC conventions,
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for benchmark methodology
and A/B results, and [`docs/RESULTS.md`](docs/RESULTS.md) for
TP-style figures and interpretation of the solver outputs.

| Module | Scope |
|---|---|
| `poisson::linalg::thomas` | Tridiagonal solver, O(N) |
| `poisson::fv::Solver1D` / `solve_poisson_1d` | 1D FV with variable ε |
| `poisson::fv::Solver2D` | 2D FV, SOR red-black, Dirichlet×Neumann |
| `poisson::spectral::DSTSolver1D` / `DSTSolver2D` | Spectral, FFTW DST-I |
| `poisson::amr::Quadtree` / `AMRArrays` | Quadtree on Morton keys |
| `poisson::amr::sor` | SOR on unstructured AMR leaves |
| `poisson::mg::vcycle_uniform` | Uniform-grid V-cycle (Dirichlet) |
| `poisson::mg::vcycle_amr_composite` | 2-grid composite MG on AMR |

## Requirements

- C++20 compiler (GCC 12+, Clang 15+, AppleClang 16+, MSVC 19.36+)
- CMake ≥ 3.20
- Eigen ≥ 3.4 (fetched via FetchContent if not found on the system)
- FFTW3 (optional, enables the spectral solvers)
- OpenMP 4.5+ (optional, enables parallel sweeps on large grids)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable OpenMP for the 2D SOR and `gs_smooth` kernels
(recommended for N ≥ 384):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOISSON_USE_OPENMP=ON
```

Build the pybind11 module for interactive Python use
(requires Python 3.9+, pybind11 fetched via FetchContent if missing):

```bash
cmake -B build -DPOISSON_BUILD_PYTHON=ON
cmake --build build --target poisson_py -j
PYTHONPATH=build/python python -c "import poisson_cpp; print(poisson_cpp.__version__)"
```

Observed single-node speedups on Apple M-series (8 threads):
| Kernel | N | Serial | OpenMP | Speedup |
|---|---|---|---|---|
| `Solver2D::solve` | 512² | 2720 ms | 1416 ms | **1.92×** |
| `mg::gs_smooth`   | 512² | 13.2 ms | 8.7 ms  | **1.52×** |

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the full methodology.

## Usage

```cpp
#include <Eigen/Core>
#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"

poisson::Grid1D grid(1.0, 100);                        // L = 1, 100 nodes
Eigen::VectorXd rho = Eigen::VectorXd::Ones(100);
Eigen::VectorXd V   = poisson::fv::solve_poisson_1d(
    rho, /*uL=*/10.0, /*uR=*/0.0, grid);
```

```cpp
#include "poisson/fv/solver2d.hpp"

poisson::Grid2D grid(1.0, 1.0, 128, 128);
poisson::fv::Solver2D solver(grid, /*eps=*/1.0, /*uL=*/0.0, /*uR=*/10.0);

Eigen::MatrixXd V   = Eigen::MatrixXd::Zero(128, 128);
Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(128, 128);
auto report = solver.solve(V, rho, {.tol = 1e-10});    // converged SOR
```

The `poisson_demo` binary exposes the same solvers from the command line:

```bash
./build/examples/poisson_demo --problem poisson1d --N 50 --uL 10 --uR 0
./build/examples/poisson_demo --problem spectral2d --N 128 --output V.json
```

## Verification

The library enforces physical consistency through **three independent
layers** of tests:

1. **Mathematical invariants** (`tests/test_invariants.cpp`) — polynomial
   exactness at machine precision, Green's-function reciprocity
   G(A,B) = G(B,A), linearity, reflection symmetry.
2. **Physical conservation** (`tests/test_conservation.cpp`) — Gauss's
   law, energy identity ½∫ρV = ½ε₀∫|∇V|², D-continuity across
   dielectric interfaces.
3. **Benchmark vs reference** (`tests/test_benchmark.cpp`) — Gaussian
   charge in a grounded box vs the analytical Fourier series (Jackson,
   *Classical Electrodynamics* §2.10), with observed O(h²) convergence
   ratio = 4.00.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
