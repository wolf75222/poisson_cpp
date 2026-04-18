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

### 4. `Solver2D::solve` — fold Dirichlet BC into effective rhs to unblock auto-vectorization

Discovered via `otool -tV` on the .o file: `mg::gs_smooth` emitted
307 NEON instructions (75 × `fadd.2d`), while `Solver2D::solve`
emitted **zero**. Root cause: the ternary expressions
`(i > 0) ? Vw_(i,j) * V(i-1,j) : Vw_(0,j) * uL_` have distinct
operands on the two branches, which the loop vectorizer rejects.
`gs_smooth` uses `if (...) s += ...` (predicated add with same-shape
operands), which vectorizes cleanly.

The transform precomputes the Dirichlet contribution into an effective
right-hand side before the iteration loop:
```
rhs_bc = rho + Vw(0,:) * uL * e_0 + Ve(Nx-1,:) * uR * e_{Nx-1}
```
and replaces the ternaries with plain `if (i > 0) s += Vw_(i,j) * V(i-1,j)`.
SIMD instruction count in `solve()` goes from **2 to 40** (partial
vectorization: the remaining branches on j-boundaries still block
full coverage).

| Metric | Before | After | Δ |
|---|---|---|---|
| sor2d 128² (965 iter) | 35.5 ms | 33.8 ms | **−5 %** |
| sor2d 256² (1837 iter) | 321 ms | 310 ms | **−3 %** |
| sor2d 512² (3483 iter) | 2720 ms | 2586 ms | **−5 %** |

The gain is modest because vectorization is only partial — reaching
`gs_smooth`'s ~307-instruction SIMD density would require also peeling
the j-boundary cases. That refactor was attempted (triplicate the j
loop, inline `update_cell`): SIMD count stayed at 40, perf stayed
within 1 % of the current code. Reverted to keep the code simple.

### 5. `Solver2D::solve` — `#pragma unroll 4` + `std::max` reduction

Applied after re-reading CS:APP loop-optimizations.md §2 (k×1 unrolling
reduces loop overhead) and §5 (branchless `fmax` plays better with
SIMD than `if (diff > color_max)`). Net:

| Metric | Before | After | Δ |
|---|---|---|---|
| sor2d 128² | 33.8 ms | 33.3 ms | **−1.5 %** |
| sor2d 256² | 310 ms | 302 ms | **−2.7 %** |
| sor2d 512² | 2586 ms | 2494 ms | **−3.6 %** |

SIMD instruction count stays at 40 (the unroll reduces branches without
exposing new vectorization opportunities). Software-prefetch variant
was tried and reverted: added run-to-run variance at N ≤ 256 with no
stable benefit at larger N.

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

## Conjugate Gradient: new iterative method

Added after an algorithm review of `CourseOnPoisson/Notes/poisson_equation.pdf`
§4.7 (iterative methods). The discrete 2D Poisson operator is symmetric
positive-definite for our Dirichlet/Neumann BCs, so Conjugate Gradient
is applicable and converges in `O(sqrt(kappa))` ≈ `O(N)` iterations
vs `O(N²)` for SOR with optimal omega.

Implementation (`include/poisson/iter/cg.hpp`, matrix-free, templated):
  - `cg(apply, x, b, params)`: classical unpreconditioned CG
  - `pcg(apply, precond, x, b, params)`: preconditioned CG with a
    user-supplied `M⁻¹` action
  - `solve_poisson_cg(...)`: high-level wrapper for the FV Dirichlet/
    Neumann Poisson problem, with optional Jacobi preconditioner
    (`diag(A)⁻¹`)

Measured on Apple M-series, tol = 1e-8, linear ramp problem:

| N    | SOR (ms / iter)    | CG (ms / iter)   | CG speedup |
|------|--------------------|------------------|------------|
| 128² | 33.1 ms / 965 iter | 7.8 ms / 183 iter | **4.2×** |
| 256² | 304 ms / 1837 iter | 63 ms / 368 iter  | **4.8×** |
| 512² | 2515 ms / 3483 iter | 639 ms / 734 iter | **3.9×** |

Iteration counts scale as ~N for CG (183, 368, 734 — factor 2 per
doubling of N) vs ~N² for SOR (965, 1837, 3483 — factor 2 also but
from a much higher baseline because each SOR iteration only
propagates information one cell).

### When to choose which solver

| Problem | Best choice | Why |
|---|---|---|
| Uniform coefficients + Dirichlet homogeneous | **DSTSolver2D** | O(N² log N) total, no iterations |
| Mixed Dirichlet/Neumann, uniform ε | **CG** | O(sqrt(κ)·N²), SPD system |
| Variable ε, needs cheap iteration | **PCG (Jacobi)** | Preconditioner helps when diag varies |
| Complex BCs, need SOR features (ω auto) | **Solver2D SOR** | Matches existing Python workflows |

### Jacobi preconditioner caveat

On the uniform-ε problems the library currently benchmarks, Jacobi
preconditioning *slightly hurts* CG (PCG ~ 442 iter vs CG ~ 187 iter
at N=128, tol=1e-10): the FV stencil diagonal is near-constant in the
interior, so diagonal scaling mostly commutes with the operator and
redirects the CG search away from the optimal Krylov subspace. PCG
becomes useful for problems with **strongly varying permittivity** or
**highly anisotropic grids**, where `diag(A)⁻¹` captures the
leading-order ill-conditioning.

## Non-applied optimisations

The following were evaluated via the CS:APP measure → transform →
re-measure workflow and rejected (either net-negative or within
measurement noise on this hardware).

- **Morton-ordered sorting of AMR leaves** (`amr::extract_arrays`).
  Hypothesis: sorting the flat `AMRArrays` by `CellKey` value (level
  then interleaved i,j) would place spatial neighbours close in memory
  and improve cache behaviour. **Result**: 0.55 ms → 1.00 ms (regression
  ×1.8). On Apple M-series, the entire 640-leaf `AMRArrays` (~100 KB)
  already fits in the 128 KB L1D regardless of ordering. Sorting by
  key groups cells *first by level*, which places level-6 neighbours
  of level-5 cells far apart (guaranteed cache line jump at every
  coarse-fine interface). The hash-map order gave better average
  locality by accident. A true space-filling-curve traversal across
  levels (normalise all cells to the finest resolution before
  Morton-interleaving) might help at larger leaf counts, but was not
  pursued since the current bench shows no cache pressure.

- **Reassociation of the 4-neighbour sum in `Solver2D::solve`**.
  Hypothesis: replacing the 4-deep serial `s += ...` chain with a
  balanced tree `(sw + se) + (ss + sn)` would reduce the dependency
  chain from depth-4 to depth-2 and expose more ILP (CS:APP §5.6).
  **Result**: N=128 35.5 ms → 36.1 ms (noise-level); N=256 321 ms →
  324 ms (noise-level). On Apple M-series FP-add latency is 2–3 cycles
  and the compiler at `-O3` already schedules independent multiplies
  in parallel. The hot loop is memory-latency-bound, not dependency-
  chain-bound, so the reassociation has no measurable effect.

- **Loop peeling of the i-boundary branches** in `Solver2D`.
  Hypothesis: splitting the inner i loop into `[i=0] [interior] [i=Nx-1]`
  would eliminate the two ternaries per cell. **Rejected without
  implementation**: on modern CPUs the border branches are taken
  consistently (always false for interior i), so branch prediction
  handles them at zero cost. Code duplication would be unjustified.

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
