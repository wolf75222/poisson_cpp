# Quickstart

```python
import numpy as np
import poisson_cpp as pc
```

Chaque solveur retourne du `numpy.ndarray` standard plus, pour les
itératifs, un objet *Report* avec le nombre d'itérations et le résidu
final. Tout est directement exploitable depuis numpy / matplotlib.

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
x = pc.thomas(a, b, c, d)            # ndarray (N,) float64, solution
```

**Retour :** `numpy.ndarray` de shape `(N,)`, dtype `float64`. Lève
`RuntimeError` si la matrice n'est pas inversible (pivot nul).

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

**Retour :** tuple `(V, report)`.
- `V` : `numpy.ndarray (Nx, Ny)`, potentiel par cellule.
- `report` : `SORReport` avec `report.iterations` (nombre de sweeps
  red+black effectués) et `report.residual` (`||V^{k+1} - V^k||_inf` au
  dernier sweep).

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

**Retour :** tuple `(report, history)`.
- `V` est mis à jour **en place**, donc disponible directement après l'appel.
- `report` : `CGReport(iterations, residual)` ; `residual` est
  `||r|| / ||b||` final.
- `history` : `list[float]` du résidu relatif à chaque itération si
  `record_history=True`, sinon liste vide. Plot direct :
  `plt.semilogy(history)`.

## Spectral DST-I

Solveur direct via DST-I (FFTW), `O(N² log N)`, uniquement pour
Dirichlet homogène sur les quatre faces.

```python
if pc.has_fftw3:
    dst   = pc.DSTSolver2D(127, 127, 1.0, 1.0, eps0=1.0)   # 127 nœuds intérieurs par axe
    rho   = np.random.randn(127, 127)                       # source aléatoire
    V_dst = dst.solve(rho)                                  # ndarray (127, 127)
```

**Retour :** `numpy.ndarray (Nx, Ny)`, potentiel aux nœuds intérieurs.
Les valeurs aux quatre bords sont implicitement nulles (Dirichlet
homogène, pas stockées).

## Exploiter le résultat

Une fois `V` obtenu, c'est un `ndarray` standard — tout numpy/matplotlib
fonctionne. Recettes courantes pour `Solver2D` (cell-centered) :

```python
import matplotlib.pyplot as plt

# Coordonnées des centres de cellules
x = (np.arange(grid.Nx) + 0.5) * grid.dx
y = (np.arange(grid.Ny) + 0.5) * grid.dy

# Heatmap + équipotentielles
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.pcolormesh(x, y, V.T, cmap="viridis", shading="auto")    # V.T car V[i,j] = (x_i, y_j)
ax.contour(x, y, V.T, levels=10, colors="white", linewidths=0.5)
fig.colorbar(im, ax=ax, label="V")

# Champ électrique E = -grad V (différences finies centrées)
gx, gy = np.gradient(V, grid.dx, grid.dy)
Ex, Ey = -gx, -gy

# Coupe 1D au milieu en y
plt.figure(); plt.plot(x, V[:, grid.Ny // 2])

# Sauvegarde
np.save("V.npy", V)                              # binaire numpy
np.savetxt("V.csv", V, delimiter=",")            # CSV
```

## Choix du solveur

| Problème | Solveur recommandé |
|---|---|
| 1D, ε quelconque | `solve_poisson_1d` (Thomas direct) |
| 2D Dirichlet homogène, ε uniforme | `DSTSolver2D` (O(N² log N)) |
| 2D Dirichlet × Neumann mixte | `solve_poisson_cg` (~5× plus rapide que SOR) |
| Source localisée, gradient fort | `Quadtree` AMR + `vcycle_amr_composite` |
| Reproduction TPs, ω_opt auto | `Solver2D` |

Workflows complets reproduisant TP1/TP3/TP4 du cours : voir
[TPs reproductibles](examples.md). Détail des trade-offs et complexité :
[ALGORITHMS.md](https://github.com/wolf75222/poisson_cpp/blob/main/docs/ALGORITHMS.md).
