# Performance

All numbers below were obtained on an Apple M-series laptop (8 performance
cores) with AppleClang 17, `-O3 -march=native`. Each datapoint is the
median of 5 runs of `bench_solvers`.

Reproduce with:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release [-DPOISSON_USE_OPENMP=ON]
cmake --build build -j
./build/benchmarks/bench_solvers <N>
```

## Hot-loop micro-optimisations

Three independent A/B experiments validated against `tests/`
(57/57 pass at each step):

### 1. AMR SOR — precompute `rhs = h²ρ/ε₀` and `Vc_inv = 1/Vc`

`src/amr/solver.cpp`. Moves two divisions per cell out of the hot loop.

| Metric | Before | After | Δ |
|---|---|---|---|
| amr_sor (640 leaves, 200 sweeps) | 0.665 ms | 0.557 ms | **−16 %** |

### 2. `mg::gs_smooth` — true in-place red-black + `Vc_inv`

`src/mg/vcycle.cpp`. Replaces "compute full V_gs then use half" with a
true in-place red-black Gauss-Seidel sweep. Eliminates the N×N scratch
buffer.

| Metric | Before | After | Δ |
|---|---|---|---|
| gs_smooth 128² (50 sweeps) | 2.51 ms | 0.80 ms | **−68 % (3.1×)** |
| gs_smooth 256² (50 sweeps) | ~10 ms | 3.22 ms | **−68 % (3.1×)** |

### 3. `fv::Solver2D::solve` — in-place red-black + member `Vc_inv_`

`src/fv/solver2d.cpp`. Same pattern. Boundary Dirichlet / Neumann
branches stay inside the stride-2 inner loop; the pattern is
branch-predictor friendly.

| Metric | Before | After | Δ |
|---|---|---|---|
| sor2d 128² (965 iter) | 67.8 ms | 36.7 ms | **−46 % (1.85×)** |
| sor2d 256² (1837 iter) | 511.8 ms | 321.3 ms | **−37 % (1.6×)** |

Iteration counts are unchanged — these are pure per-iteration wins.

## OpenMP parallel sweeps

Enabled with `-DPOISSON_USE_OPENMP=ON` (default OFF). Parallelizes the
red-black sweeps of `Solver2D::solve` and `mg::gs_smooth`. Uses a
`#pragma omp parallel for reduction(max:color_max) schedule(static)`
inside each color sweep, with a C++ `if (N >= 384)` gate bypassing the
pragma entirely at small grids.

### Speedup vs serial (N ≥ 384)

8 threads on Apple M-series:

| Kernel | N | Serial | OpenMP | Speedup |
|---|---|---|---|---|
| `Solver2D::solve` | 512² | 2720 ms | 1416 ms | **1.92×** |
| `Solver2D::solve` | 768² | 6740 ms* | 4263 ms | **1.58×** |
| `mg::gs_smooth`   | 512² | 13.2 ms | 8.7  ms | **1.52×** |
| `mg::gs_smooth`   | 768² | 19.5 ms* | 12.7 ms | **1.54×** |

<small>*Serial 768² estimated by extrapolation from 256² / 512² pairs.</small>

### Why a runtime threshold?

OpenMP adds `~50 µs` of fork-join overhead per `#pragma omp parallel for`.
Below about N ≈ 384 that overhead exceeds the per-sweep work, so
parallelism slows things down. The C++ `if (N >= kOmpThreshold)` branch
routes small grids through a fully-OpenMP-free code path.

With `POISSON_USE_OPENMP=ON` there is still a ~10 % residual cost at
small N from the presence of libomp in the linked binary (memory
allocator, TLS setup). This is why the default is OFF: users running
small-N interactive problems should not pay for a feature they cannot
benefit from.

### Correctness under parallelism

Red-black Gauss-Seidel is **data-parallel within a color** — no two
cells of the same color share a neighbour being updated concurrently.
The reduction on `max|V_new - V_old|` is exact (OpenMP `reduction(max:)`).
All 57 tests, including the invariant suite (Green's reciprocity,
energy identity, polynomial exactness), pass under both
`POISSON_USE_OPENMP=OFF` and `=ON` with identical output values.

## Benchmarks exposed

`benchmarks/bench_solvers.cpp` reports:

- `thomas`      — tridiagonal solve, N coefficients
- `sor2d`       — `fv::Solver2D::solve` on N×N, tol 1e-8
- `spectral2d`  — `DSTSolver2D::solve`, N×N (if FFTW available)
- `gs_smooth`   — `mg::gs_smooth`, 50 sweeps on N×N
- `amr_sor`     — `amr::sor`, 200 sweeps on a Gaussian-refined tree
                   (base 16×16, up to level 6, ≈ 640 leaves)

## Non-applied optimisations

The following were evaluated but rejected (either net-negative or
marginal on this workload):

- **Morton-ordered SOR for `Solver2D`**: column-major iteration already
  exploits Eigen's column-major layout optimally. No cache win.
- **Higher-order (bi-cubic) prolongation in V-cycle**: multigrid rate
  is dominated by the smoother, not by the prolongation. No measurable
  improvement on Gaussian sources.
- **Galerkin coarse operator for composite MG**: non-trivial to
  implement correctly across coarse-fine interfaces; the
  rediscretisation-based coarse operator converges fast enough (~0.7
  per cycle) for the tested problems.

## Bottlenecks and what to try next

| Rank | Kernel | Dominant cost | Next idea |
|---|---|---|---|
| 1 | `Solver2D::solve` | Memory bandwidth (5-point stencil) | Tile blocking for L1 |
| 2 | `mg::gs_smooth` | Same | Same |
| 3 | `DSTSolver2D` | FFTW work | `FFTW_PATIENT` + re-plan once |
| 4 | `amr::sor` | Random-access neighbour fetch | Leaf reordering by space-filling curve |
