# Quickstart

```python
import numpy as np
import poisson_cpp as pc
```

## Thomas tridiagonal (1D)

Résout `A x = d` quand `A` est tridiagonale, en `O(N)`. Ici on construit
le Laplacien discret 1D pour `-V''(x) = 1` avec `V(0) = V(L) = 0`. La
solution exacte est une parabole.

```python
N = 100                              # nombre de points
b = 2.0 * np.ones(N)                 # diagonale principale
a = np.zeros(N); a[1:]  = -1.0       # sous-diagonale ; a[0] inutilisé
c = np.zeros(N); c[:-1] = -1.0       # sur-diagonale ; c[N-1] inutilisé
d = np.ones(N)                       # second membre constant
x = pc.thomas(a, b, c, d)            # vecteur solution, longueur N
```

## SOR red-black 2D

Poisson 2D sans charge interne avec Dirichlet en x (V=0 à gauche,
V=10 à droite) et Neumann en y. La solution est une rampe linéaire en x.

```python
grid = pc.Grid2D(1.0, 1.0, 64, 64)              # domaine [0,1]^2, 64x64 cellules
sor  = pc.Solver2D(grid, eps=1.0, uL=0.0, uR=10.0)
rho  = np.zeros((64, 64))                       # pas de source
V, report = sor.solve(rho, tol=1e-10)           # omega_opt auto si non fourni
print(report)                                   # SORReport(iterations=..., residual=...)
```

## Gradient conjugué

Même problème que SOR, autre algorithme. CG converge en `O(sqrt(kappa))`
itérations, ~5× plus rapide que SOR à tolérance égale. `order='F'` aligne
les tableaux numpy sur le layout column-major d'Eigen et évite la copie
en entrée/sortie.

```python
grid = pc.Grid2D(1.0, 1.0, 128, 128)
V    = np.zeros((128, 128), order="F")          # initial guess + sortie in-place
rho  = np.zeros((128, 128), order="F")
report, history = pc.solve_poisson_cg(
    V, rho, grid,
    uL=0.0, uR=10.0,                            # Dirichlet en x
    tol=1e-10,
    record_history=True,                        # garde ||r||/||b|| par itération
)
print(f"{report.iterations} iter, résidu {report.residual:.2e}")
```

## Spectral DST-I

Solveur direct via DST-I (FFTW), `O(N² log N)`, uniquement pour
Dirichlet homogène sur les quatre faces.

```python
if pc.has_fftw3:
    dst   = pc.DSTSolver2D(127, 127, 1.0, 1.0, eps0=1.0)   # 127 nœuds intérieurs par axe
    rho   = np.random.randn(127, 127)                       # source aléatoire
    V_dst = dst.solve(rho)
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
