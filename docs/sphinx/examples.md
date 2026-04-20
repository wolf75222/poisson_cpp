# Exemples

Workflows complets utilisant les bindings poisson_cpp depuis Python.
Chaque section donne le code minimal et explique ce que retourne le
solveur.

## TP1 : Poisson 1D, comparaison à la solution analytique

Poisson 1D sans charge entre deux électrodes : on fixe `V(0) = uL` et
`V(L) = uR`, et on résout `-V''(x) = 0`. Sans source, la solution est
la rampe linéaire `V(x) = uL + (uR - uL) * x / L`, qu'on connaît
analytiquement. C'est le test minimal : tout écart à la rampe vient des
erreurs d'arrondi du solveur direct, pas du schéma de discrétisation.

```python
import numpy as np
import matplotlib.pyplot as plt
import poisson_cpp as pc

N, L, uL, uR = 100, 1.0, 10.0, 0.0

# 1. Maillage et résolution
grid = pc.Grid1D(L, N)                          # N noeuds equispacés sur [0, L]
rho  = np.zeros(N)                              # source nulle
V    = pc.solve_poisson_1d(rho, uL, uR, grid)   # ndarray (N,)

# 2. Coordonnées des noeuds (Grid1D.x(i) = i * dx)
x = np.array([grid.x(i) for i in range(N)])

# 3. Comparaison à l'analytique
V_th    = uL + (uR - uL) * x / L
err_inf = np.max(np.abs(V - V_th))
print(f"erreur L_inf = {err_inf:.2e}")          # ~1e-14, précision machine

# 4. Tracé
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(x, V, "o", ms=3, label="numérique")
axes[0].plot(x, V_th, "k--", label="analytique")
axes[0].set(xlabel="x", ylabel="V(x)"); axes[0].legend()
axes[1].semilogy(x, np.abs(V - V_th) + 1e-30, "o", ms=3)
axes[1].set(xlabel="x", ylabel="|erreur|", title=f"L_inf = {err_inf:.2e}")
plt.show()
```

L'erreur reste sous `~1e-14` partout, soit la borne de Thomas en double
précision (`O(N) * eps_machine * ||V||_inf`).

![TP1 : Poisson 1D + erreur](../figures/tp1_poisson_1d.png)

## TP2 : couches diélectriques 1D et continuité de D

Empilement de trois couches diélectriques sur `[0, L]` avec
`ε_r = 5` pour `x < 0.3`, `ε_r = 1` pour `0.3 ≤ x < 0.7`, `ε_r = 2`
ensuite. Électrodes à `V(0) = 15 V` et `V(L) = 0`, pas de charge
volumique. Comme `∇·D = 0` sans charge surfacique, le déplacement
`D = ε₀ ε_r dV/dx` doit être *constant* dans tout le domaine ; le
potentiel `V` reste continu mais sa pente change à chaque interface
(plus pentue dans les couches de faible `ε_r`).

```python
import numpy as np
import matplotlib.pyplot as plt
import poisson_cpp as pc

N, L, uL, uR, eps0 = 200, 1.0, 15.0, 0.0, 1.0

grid  = pc.Grid1D(L, N)
x     = np.array([grid.x(i) for i in range(N)])
eps_r = np.where(x < 0.3, 5.0, np.where(x < 0.7, 1.0, 2.0))
rho   = np.zeros(N)

# Solveur direct, ε(x) variable, moyenne harmonique aux faces
V = pc.solve_poisson_1d_dielectric(rho, eps_r, uL, uR, grid, eps0)

# Champ E et déplacement D aux faces
dx       = L / (N - 1)
eps_face = 2 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])
E        = -(V[1:] - V[:-1]) / dx
D        = eps0 * eps_face * E

# Valeur théorique de D (constante par conservation)
D_theo = eps0 * (uL - uR) / (dx * np.sum(1.0 / eps_face))
print(f"D varie de {(D.max()-D.min())/abs(D.mean()):.1e} relatif")
print(f"D_num ≈ {D.mean():.4f},  D_theo = {D_theo:.4f}")
```

`D` reste constant à `~1e-13` près (précision machine) : la moyenne
harmonique aux faces préserve la composante normale de `D` à travers
les interfaces. Le code complet avec tracé V(x)/E(x)/D(x) est dans
[`python/plot_tp_style.py:tp2`](https://github.com/wolf75222/poisson_cpp/blob/main/python/plot_tp_style.py).

![TP2 : couches diélectriques](../figures/tp2_dielectric.png)

## TP3 : SOR 2D + courbe de convergence

Poisson 2D sans charge dans un carré `[0, L]²`. Bords gauche et droit
fixés à `V = uL` et `V = uR` (Dirichlet), bords haut et bas en Neumann
homogène (`∂V/∂n = 0`). Sans source et avec Neumann en y, la solution
doit être invariante en y et former la même rampe linéaire qu'au TP1.

On en profite pour tracer la décroissance du résidu : on appelle
`solve_inplace` par paquets de quelques itérations et on lit
`report.residual` après chaque paquet.

```python
N, uL, uR = 64, 0.0, 10.0
grid = pc.Grid2D(1.0, 1.0, N, N)
sor  = pc.Solver2D(grid, eps=1.0, uL=uL, uR=uR)

V    = np.zeros((N, N), order="F")              # initial guess + sortie in-place
rho  = np.zeros((N, N), order="F")

history, iters, total = [], [], 0
for _ in range(500):
    rep = sor.solve_inplace(V, rho, omega=-1.0, tol=1e-10, max_iter=20)
    total += rep.iterations
    history.append(rep.residual)
    iters.append(total)
    if rep.residual < 1e-10 or rep.iterations < 20:
        break

# Centres des cellules
xc = (np.arange(N) + 0.5) * grid.dx
yc = (np.arange(N) + 0.5) * grid.dy

# 3 panneaux : heatmap, coupe y=L/2, convergence
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im = axes[0].pcolormesh(xc, yc, V.T, cmap="viridis", shading="auto")
axes[0].set(title="V(x, y)", xlabel="x", ylabel="y", aspect="equal")
fig.colorbar(im, ax=axes[0])

axes[1].plot(xc, V[:, N // 2], "o", ms=3, label="SOR")
axes[1].plot(xc, uL + (uR - uL) * xc, "k--", label="analytique")
axes[1].set(title="Coupe y = L/2", xlabel="x", ylabel="V"); axes[1].legend()

axes[2].semilogy(iters, history, "o-", ms=3)
axes[2].set(title="Convergence SOR", xlabel="itération",
            ylabel=r"$\max |V^{k+1} - V^k|$")
plt.show()
```

Avec `omega=-1` et `N=64`, ω_opt ≈ 1.91 et SOR converge en ~600 sweeps
red+black à `tol=1e-10`. L'écart à la rampe analytique reste sous `1e-9`.

![TP3 : SOR 2D](../figures/tp3_sor2d.png)

## TP4 : Étude de convergence DST spectrale

On prend une solution analytique connue `V(x, y) = sin(πx/L) sin(πy/L)`,
on en déduit la source `-Δ V = 2 (π/L)² V`, puis on vérifie que le
solveur DST retrouve `V` à partir de cette source. En répétant pour
plusieurs `N` et en traçant l'erreur `||V_num - V_exact||_∞` en fonction
de `h = L/(N+1)` en log-log, on mesure l'ordre du schéma. On attend
une pente `+2` car la discrétisation FV 5-points est O(h²).

```python
import math
Ns = [15, 31, 63, 127, 255]
L  = 1.0
a  = math.pi / L
errs, hs = [], []

for N in Ns:
    h = L / (N + 1)
    # Source pour la version continue: rho = 2 a^2 V
    i = np.arange(1, N + 1) * h
    j = np.arange(1, N + 1) * h
    X, Y    = np.meshgrid(i, j, indexing="ij")
    V_exact = np.sin(a * X) * np.sin(a * Y)
    rho     = 2 * a * a * V_exact

    dst   = pc.DSTSolver2D(N, N, L, L, eps0=1.0)
    V_num = dst.solve(rho)
    errs.append(np.max(np.abs(V_num - V_exact)))
    hs.append(h)

hs, errs = np.array(hs), np.array(errs)
slope, intercept = np.polyfit(np.log(hs), np.log(errs), 1)
print(f"pente empirique = {slope:.3f} (attendu +2.0)")

plt.loglog(hs, errs, "o-", label=f"err DST (pente {slope:.2f})")
plt.loglog(hs, np.exp(intercept) * hs ** 2, "k--", label="référence h²")
plt.xlabel("h"); plt.ylabel("erreur L_inf"); plt.legend(); plt.grid(which="both")
plt.show()
```

La pente empirique tombe sur `+2.000` à mieux que 1 % pour `N >= 31`.
Pour la version *discrète* (mode propre exact du Laplacien 5-points),
voir `python/plot_tp_style.py:tp4` : l'erreur descend alors à
`~eps_machine` (DST inverse exactement le Laplacien discret).

![TP4 : convergence DST O(h²)](../figures/tp4_spectral_convergence.png)

## Multigrille uniforme : SOR vs V-cycle

Sur une grille uniforme, SOR coûte `O(N³)` itérations alors qu'un V-cycle
multigrille avec un bon smoother converge en `O(N²)` au total. On le
vérifie sur une source manufacturée avec solution exacte connue.

`gs_smooth` (lisseur Gauss-Seidel red-black), `laplacian_fv` (calcul du
résidu) et `vcycle_uniform` sont les briques exposées. Pour écrire son
propre V-cycle, utiliser aussi `restrict_avg`, `prolongate_const` et
`prolongate_bilinear`.

```python
import math
import numpy as np
import poisson_cpp as pc

N = 64
h = 1.0 / N

# Source manufacturée : V_exact(x, y) = sin(pi x) sin(pi y), donc
# rho = 2 pi^2 V_exact (avec eps0 = 1).
xc = (np.arange(N) + 0.5) * h
X, Y = np.meshgrid(xc, xc, indexing="ij")
V_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
rho     = 2 * (math.pi ** 2) * V_exact

# Méthode 1 : SOR pur via gs_smooth (omega = 1, pas red-black SOR)
V_gs = np.zeros((N, N), order="F")
res_gs = []
for k in range(200):
    pc.gs_smooth(V_gs, np.asfortranarray(rho), h, n_iter=5)
    r = float(np.max(np.abs(rho - pc.laplacian_fv(V_gs, h))))
    res_gs.append(r)

# Méthode 2 : V-cycle multigrille uniforme
V_mg = np.zeros((N, N))
res_mg = []
for k in range(20):
    V_mg = pc.vcycle_uniform(V_mg, rho, h, n_pre=2, n_post=2, n_min=4)
    r = float(np.max(np.abs(rho - pc.laplacian_fv(V_mg, h))))
    res_mg.append(r)

print(f"GS  ({len(res_gs)*5} sweeps): résidu final {res_gs[-1]:.2e}")
print(f"MG  ({len(res_mg)} V-cycles): résidu final {res_mg[-1]:.2e}")
print(f"erreur MG vs exacte : {np.max(np.abs(V_mg - V_exact)):.2e}")
```

20 V-cycles MG amènent le résidu sous `1e-10` ; il faudrait des
milliers de sweeps Gauss-Seidel pour atteindre la même précision.

## TP5 : AMR quadtree sur source localisée

Pour une source concentrée (ici une gaussienne au centre du domaine), un
maillage uniforme gaspille des cellules loin du pic. Le quadtree adapte
la résolution localement : on raffine récursivement les feuilles dont le
centre tombe à moins de `4 σ` du pic. Le solveur est ensuite SOR sur les
arrays plats extraits de l'arbre, puis on accélère avec un V-cycle
composite (smoother SOR sur AMR + V-cycle uniforme sur la grille de
fond).

```python
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import poisson_cpp as pc

L, sigma = 1.0, 0.04

# 1. Critère de raffinement : on subdivise tant qu'une feuille est dans
#    un disque de rayon 4*sigma autour de la source.
def predicate(key):
    lvl = pc.level_of(key)
    if lvl >= 8:
        return False
    h  = L / (1 << lvl)
    cx = (pc.i_of(key) + 0.5) * h
    cy = (pc.j_of(key) + 0.5) * h
    return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < (4 * sigma) ** 2

def rho_func(x, y):
    return math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))

# 2. Construction de l'arbre : refine + balance 2:1 + évaluation de rho
tree = pc.Quadtree(L, level_min=4)
tree.build(predicate, level_max=8, rho_func=rho_func)
print(f"{tree.num_leaves()} feuilles (vs {(1 << 8) ** 2} en uniforme)")

# 3. Vue plate + résolution SOR sur AMR
arr = pc.extract_arrays(tree)
rep = pc.amr_sor(arr, omega=1.85, tol=1e-7, max_iter=5000)
print(rep)
print(f"|résidu|_∞ = {np.abs(pc.amr_residual(arr)).max():.2e}")

# 4. Accélération multigrille : 5 V-cycles composites
arr_mg = pc.extract_arrays(tree)
for _ in range(5):
    pc.vcycle_amr_composite(arr_mg, tree)
print(f"après 5 V-cycles : |résidu|_∞ = "
      f"{np.abs(pc.amr_residual(arr_mg)).max():.2e}")

# 5. Renvoyer V dans l'arbre (utile si on veut itérer sur cell.V depuis Python)
pc.writeback(tree, arr_mg.keys, arr_mg.V)
center_key = pc.make_key(8, 1 << 7, 1 << 7)
if tree.is_leaf(center_key):
    print("V au centre :", tree.at(center_key).V)

# 6. Visualisation : V color-coded + bord des feuilles
fig, ax = plt.subplots(figsize=(6, 6))
patches, colors = [], []
for key, V in zip(arr.keys, arr.V):
    lvl = pc.level_of(key)
    h   = L / (1 << lvl)
    x   = pc.i_of(key) * h
    y   = pc.j_of(key) * h
    patches.append(Rectangle((x, y), h, h))
    colors.append(V)
pc_coll = PatchCollection(patches, edgecolor="black", linewidth=0.1,
                          cmap="viridis")
pc_coll.set_array(np.array(colors))
ax.add_collection(pc_coll)
ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", title="V sur le quadtree")
fig.colorbar(pc_coll, ax=ax, label="V")
plt.show()
```

Le V-cycle composite divise le résidu par ~0.7 par cycle (re-discrétisation
sur la grille grossière, pas Galerkin). Pour une réduction plus agressive,
voir [`python/make_banner.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/make_banner.py)
qui résout une scène à 10 charges sur 5000 feuilles.

![TP5 : maillage AMR + V](../figures/tp5_amr.png)

## CG vs SOR : convergence comparée

Sur le même problème (Dirichlet en x, Neumann en y, sans source), CG
converge en `O(sqrt(kappa))` itérations, SOR en `O(N)` itérations avec
ω_opt. La constante en pratique : CG est ~5× plus rapide. On compare
en enregistrant l'historique du résidu côté CG et le résidu par paquet
côté SOR.

```python
import numpy as np
import matplotlib.pyplot as plt
import poisson_cpp as pc

N, uL, uR = 64, 0.0, 10.0
g = pc.Grid2D(1.0, 1.0, N, N)

# CG : solve_poisson_cg met V à jour en place et renvoie l'historique
V_cg = np.zeros((N, N), order="F")
rep_cg, hist_cg = pc.solve_poisson_cg(
    V_cg, np.zeros((N, N), order="F"), g,
    uL=uL, uR=uR, tol=1e-10, max_iter=2000, record_history=True,
)

# SOR : appel par paquets pour récupérer le résidu intermédiaire
sor = pc.Solver2D(g, eps=1.0, uL=uL, uR=uR)
V_sor = np.zeros((N, N), order="F")
rho   = np.zeros((N, N), order="F")
hist_sor, iters_sor, total = [], [], 0
for _ in range(500):
    rep = sor.solve_inplace(V_sor, rho, omega=-1.0, tol=1e-10, max_iter=20)
    total += rep.iterations
    hist_sor.append(rep.residual); iters_sor.append(total)
    if rep.residual < 1e-10 or rep.iterations < 20:
        break

print(f"CG  : {rep_cg.iterations} iter, résidu {rep_cg.residual:.2e}")
print(f"SOR : {total} iter, résidu {hist_sor[-1]:.2e}")

plt.semilogy(range(len(hist_cg)), hist_cg, "o-", ms=3, label="CG")
plt.semilogy(iters_sor, hist_sor, "s-", ms=3, label="SOR ω_opt")
plt.xlabel("itération"); plt.ylabel("résidu"); plt.legend(); plt.grid(which="both")
plt.show()
```

Pour activer le préconditionneur Jacobi sur CG, passer
`use_preconditioner=True` à `solve_poisson_cg`. À ε uniforme, le gain
est marginal voire négatif (la diagonale du Laplacien FV est presque
constante en intérieur) ; à ε fortement variable, Jacobi capture la
variation locale et accélère.

## Plus loin

- Scène AMR multi-charges (10 gaussiennes ±q, 5000 feuilles) :
  [`python/make_banner.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/make_banner.py).
- Reproductibilité figures `docs/figures/*.png` :
  [`python/plot_tp_style.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/plot_tp_style.py).
