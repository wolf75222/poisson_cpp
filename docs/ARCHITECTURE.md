# Architecture

## Module layout

```
poisson_cpp/
  include/poisson/
    core/            Grid1D, Grid2D, shared types
    linalg/          thomas tridiagonal solver
    fv/              Finite-volume solvers (1D/2D, uniform / dielectric)
    spectral/        DST-I spectral solvers (FFTW wrapper)
    amr/             Quadtree + Morton keys + heterogeneous FV
    mg/              Uniform V-cycle + 2-grid composite MG on AMR
    io/              JSON snapshot I/O (header-only, test use only)
  src/               .cpp mirror of the headers above
  tests/             Catch2 unit, invariant, conservation, benchmark tests
  benchmarks/        std::chrono perf harness (bench_solvers)
  examples/          CLI binary (poisson_demo)
  python/            reference snapshot generator (pytest-compatible)
  data/reference/    frozen JSON snapshots for regression testing
```

## Discretisation conventions

The library mixes two grid conventions depending on the module. Each is
documented in the corresponding header; do not mix them across solvers.

| Module | Grid | BCs supported |
|---|---|---|
| `fv::Solver1D` / `solve_poisson_1d` | Cell-centered, N cells, Dirichlet at both faces | Dirichlet (uL, uR), variable ε |
| `fv::Solver2D` | Cell-centered, Nx × Ny cells | Dirichlet in x (uL, uR), Neumann in y |
| `spectral::DSTSolver1D` / `DSTSolver2D` | Node-centered, N interior nodes at i·h, h = L/(N+1) | Homogeneous Dirichlet on whole boundary |
| `amr::Quadtree` | Cell-centered, leaf-based, Morton-encoded keys | Dirichlet (V=0) at domain boundary |
| `mg::vcycle_uniform` | Cell-centered (shares `gs_smooth` convention) | Homogeneous Dirichlet on 4 faces |

### FV stencil weights at AMR coarse-fine interfaces

Derived in `CourseOnPoisson/notebooks/TP5_AMR_Poisson_2D.ipynb` and
mirrored in `AMRArrays::extract_arrays`. For a leaf cell facing a
neighbour:

| Neighbour configuration | `diag +=` | `off=` (per neighbour) |
|---|---|---|
| Domain boundary (Dirichlet V=0) | 2 | 0 |
| Same-level neighbour | 1 | 1 |
| Coarser neighbour (1) | 2/3 | 2/3 |
| Two finer neighbours (2) | 4/3 | 2/3 each |

These weights are locally conservative and preserve the discrete
divergence-theorem identity `Σ F_face = h² ρ` at every cell.

### Morton encoding

Cell keys are `uint64_t` packing (level, i, j) so that children are
adjacent in numerical order. See `include/poisson/amr/morton.hpp`:

```
bits 0..55   : interleaved i, j (28-bit each, supports up to level 28)
bits 56..63  : level
```

The encoding uses bit-interleaving with `_pdep_u64` when available,
with a portable fallback otherwise.

## Stencil / operator properties

- The uniform 5-point Laplacian is **self-adjoint** under homogeneous
  Dirichlet BCs (verified by `test_invariants.cpp` via Green's
  reciprocity to 1e-13).
- The AMR FV stencil with the weights above is **locally conservative**
  (`test_conservation.cpp` verifies Σ F_face = h² ρ at every cell after
  converged SOR).
- The 5-point stencil is **exact on polynomials of degree ≤ 2** in each
  variable (`test_invariants.cpp` verifies V = x(L-x)y(L-y) is
  reproduced at machine precision).

## Build system

- Eigen: prefers a system install, falls back to a FetchContent clone of
  the 3.4 headers (without executing Eigen's own CMake, which registers
  hundreds of internal tests).
- FFTW3: optional. The library builds without it, but the spectral
  solvers are disabled.
- nlohmann_json: used only by `poisson/io/json_io.hpp` (excluded from
  the installed headers). Tests link it directly.
- OpenMP: optional (`POISSON_USE_OPENMP=ON`). Graceful fallback: without
  OpenMP all kernels run serially with the same semantics.
- Catch2: fetched via FetchContent at v3.5.3.

## Installation contract

`install(EXPORT poissonTargets ...)` publishes `poisson::poisson`. A
downstream consumer needs only:

```cmake
find_package(poisson CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE poisson::poisson)
```

Eigen3 is propagated as a public dependency. FFTW3 is a public link
dependency too when the library was built with spectral support.

## Thread safety

All solvers are **reentrant on separate instances**. They are **not
thread-safe on a shared instance** because the FFTW scratch buffers
(`in_`, `out_` in `DSTSolver*`) are `mutable` and shared. Spawn one
solver per thread if doing concurrent solves; the SOR/Gauss-Seidel
solvers have no shared state beyond `V` and can be called from multiple
threads on disjoint data.

## Known limitations

- Only square domains at the root of the quadtree (Lx = Ly).
- Morton encoding limits refinement to 28 levels (= 256M × 256M
  equivalent uniform grid).
- No Neumann BCs for the spectral solvers (DST-I implies homogeneous
  Dirichlet). Use `Solver2D` for mixed Dirichlet/Neumann.
- The 2-grid composite MG uses rediscretisation, not a Galerkin coarse
  operator. The observed per-cycle reduction factor is ~0.7 on refined
  AMR grids (vs ~0.1 for a true multigrid).
