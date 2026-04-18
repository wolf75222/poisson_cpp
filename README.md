# poisson_cpp

Modern C++20 library of Poisson solvers: Thomas tridiagonal, SOR red-black,
spectral (DST-I via FFTW), and quadtree AMR with 2-grid composite
multigrid.

Port of the Python notebooks in
[`CourseOnPoisson/notebooks/`](https://github.com/alvarezlaguna/AMR_Poisson)
with library-oriented API.

## Status

Work in progress. See the 6-phase roadmap in
[`docs/plan.md`](docs/plan.md) (if present) or the issues tracker.

| Phase | Scope | Status |
|---|---|---|
| 1 | Thomas + 1D finite-volume Dirichlet + Catch2 harness | in progress |
| 2 | Dielectric (harmonic mean) + 2D SOR red-black | planned |
| 3 | Spectral DST 1D/2D via FFTW | planned |
| 4 | Quadtree AMR (Morton indexing) + SOR on AMR | planned |
| 5 | V-cycle multigrid, composite 2-grid on AMR | planned |
| 6 | Benchmarks + pybind11 + JSON snapshots | planned |

## Requirements

- C++20 compiler (GCC 12+, Clang 15+, or MSVC 19.36+).
- CMake ≥ 3.20.
- Eigen ≥ 3.4 (header-only, fetched via FetchContent if not found).
- FFTW3 (spectral phase 3 and later).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Usage

```cpp
#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"

poisson::Grid1D grid(1.0, 100);                  // L = 1, 100 nodes
Eigen::VectorXd rho = Eigen::VectorXd::Ones(100);
Eigen::VectorXd V   = poisson::fv::solve_poisson_1d(rho, 10.0, 0.0, grid);
```

The companion binary `poisson_demo` exposes the same solvers from the
command line:

```bash
./build/examples/poisson_demo --problem poisson1d --N 50 --uL 10 --uR 0
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
