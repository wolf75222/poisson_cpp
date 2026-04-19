# Quickstart

```python
import numpy as np
import poisson_cpp as pc
```

## Thomas tridiagonal (1D)

```python
N = 100
a = np.zeros(N); b = 2.0 * np.ones(N); c = np.zeros(N); d = np.ones(N)
a[1:] = -1.0; c[:-1] = -1.0
x = pc.thomas(a, b, c, d)
```

## SOR red-black 2D

```python
grid = pc.Grid2D(1.0, 1.0, 64, 64)
sor  = pc.Solver2D(grid, eps=1.0, uL=0.0, uR=10.0)
V, report = sor.solve(np.zeros((64, 64)), tol=1e-10)
print(report)   # SORReport(iterations=..., residual=...)
```

## Gradient conjugué

```python
grid = pc.Grid2D(1.0, 1.0, 128, 128)
V    = np.zeros((128, 128), order="F")
rho  = np.zeros((128, 128), order="F")
report, history = pc.solve_poisson_cg(
    V, rho, grid, uL=0.0, uR=10.0, tol=1e-10, record_history=True)
```

L'ordre Fortran (`order="F"`) est requis pour que pybind11 / Eigen
travaillent en place sans copie.

## Spectral DST-I

```python
if pc.has_fftw3:
    dst = pc.DSTSolver2D(127, 127, 1.0, 1.0, eps0=1.0)
    V_dst = dst.solve(np.random.randn(127, 127))
```

## Choix du solveur

| Problème | Solveur recommandé |
|---|---|
| 1D, ε quelconque | `solve_poisson_1d` (Thomas direct) |
| 2D Dirichlet homogène, ε uniforme | `DSTSolver2D` (O(N² log N)) |
| 2D Dirichlet × Neumann mixte | `solve_poisson_cg` (~5× plus rapide que SOR) |
| Source localisée, gradient fort | `Quadtree` AMR + `vcycle_amr_composite` |
| Reproduction TPs, ω_opt auto | `Solver2D` |

Détail des trade-offs et complexité :
[ALGORITHMS.md](https://github.com/wolf75222/poisson_cpp/blob/main/docs/ALGORITHMS.md).
