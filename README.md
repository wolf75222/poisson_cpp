# poisson_cpp

Modern C++20 library of Poisson solvers: Thomas tridiagonal, SOR red-black,
spectral (DST-I via FFTW), and quadtree AMR with 2-grid composite
multigrid.

Port of the Python notebooks in
[`CourseOnPoisson/notebooks/`](https://github.com/alvarezlaguna/AMR_Poisson)
with library-oriented API.

## Status

All 6 phases of the initial port are done. 30 Catch2 tests green on
macOS/clang (CI matrix: Ubuntu + macOS × Debug + Release).

| Phase | Scope | Status |
|---|---|---|
| 1 | Thomas + 1D finite-volume Dirichlet + Catch2 harness | done |
| 2 | Dielectric (harmonic mean) + 2D SOR red-black | done |
| 3 | Spectral DST 1D/2D via FFTW | done |
| 4 | Quadtree AMR (Morton indexing) + SOR on AMR | done |
| 5 | V-cycle multigrid, composite 2-grid on AMR | done |
| 6 | Benchmarks (std::chrono) + JSON snapshots + reference tests | done |

pybind11 bindings are intentionally not provided; Python validation is
done via JSON snapshot files written by `python/dump_reference.py` and
loaded in the C++ test suite.

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
