# TPs reproductibles

Trois workflows complets, transposés depuis
[`python/plot_tp_style.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/plot_tp_style.py).
Chaque section donne le code minimal, ce que retourne le solveur, et
comment exploiter le résultat.

## TP1 — Poisson 1D, comparaison à la solution analytique

Énoncé : `-V''(x) = 0` sur `[0, L]` avec `V(0) = uL`, `V(L) = uR`. La
solution exacte est la rampe linéaire `V(x) = uL + (uR - uL) * x / L`.

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

## TP3 — SOR 2D + courbe de convergence

Énoncé : Poisson 2D sans charge, Dirichlet en x (`V = uL` à gauche,
`V = uR` à droite), Neumann en y. La solution est une rampe linéaire
en x indépendante de y.

Pour récupérer l'historique du résidu, appeler `solve_inplace` par
paquets de quelques itérations et lire `report.residual` après chaque
paquet.

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

## TP4 — Étude de convergence DST spectrale

Énoncé : solution manufacturée `V(x, y) = sin(πx/L) sin(πy/L)`, donc
`-Δ V = 2 (π/L)² V`. On résout par DST pour plusieurs N et on trace
l'erreur `||V_num - V_exact||_inf` en fonction de `h = L/(N+1)`. La
pente attendue en log-log est `+2` (schéma O(h²)).

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

## Plus loin

- AMR multi-charges (TP5) : voir [`python/make_banner.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/make_banner.py).
- Diélectrique multicouche (TP2) : voir [`python/plot_tp_style.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/plot_tp_style.py)
  fonction `tp2()`.
- CG vs SOR head-to-head : voir [`python/plot_cg.py`](https://github.com/wolf75222/poisson_cpp/blob/main/python/plot_cg.py).
