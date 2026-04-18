# Algorithmes

Pseudocode des algorithmes implémentés dans la librairie, avec les
liens vers le code C++ et les concepts physiques sous-jacents.

## Équation modèle

On résout l'équation de Poisson en 1D/2D :

```
−∇·(ε ∇V) = ρ        sur un domaine Ω
V = g                sur ∂Ω_Dirichlet
∂V/∂n = 0            sur ∂Ω_Neumann
```

- V : potentiel électrostatique (volts)
- ρ : densité de charge (C/m³ en 3D, C/m² en 2D)
- ε = ε₀·ε_r : permittivité (F/m)
- `D = ε·E = −ε·∇V` : déplacement électrique. `∇·D = ρ` (loi de Gauss)

## Discrétisation volumes finis (FV)

Le domaine est découpé en cellules. L'intégration de `−∇·(ε∇V) = ρ`
sur chaque cellule donne, via Green–Ostrogradski :

```
−Σ F_face = h^d · ρ_cell           (d = dim, h = taille de cellule)
F_face = ε_face · (V_out − V_in) / dx
```

où `F_face` est le flux à travers chaque face. C'est le schéma à
5 points en 2D uniforme :

```
(−V_{i−1,j} − V_{i+1,j} − V_{i,j−1} − V_{i,j+1} + 4·V_{i,j}) / h² = ρ_{i,j}/ε
```

Aux faces entre deux cellules de permittivités différentes, on utilise
la **moyenne harmonique** `ε_face = 2·ε_a·ε_b / (ε_a + ε_b)` : c'est la
seule qui préserve la continuité du déplacement normal
`D·n = ε·∂V/∂n` à travers l'interface, cf. test
[`test_conservation.cpp`](../tests/test_conservation.cpp).

---

## 1. Thomas tridiagonal

Solveur direct `O(N)` pour un système `A x = d` avec `A` tridiagonale.
Utilisé par `fv::Solver1D` (Poisson 1D). Code :
[`src/linalg/thomas.cpp`](../src/linalg/thomas.cpp).

```
function thomas(a, b, c, d):      # a: sub-diag, b: diag, c: super-diag
  # Forward sweep — élimination gaussienne
  for i = 1 .. N−1:
    m     = a[i] / b[i−1]
    b[i] -= m · c[i−1]
    d[i] -= m · d[i−1]
  # Back substitution
  x[N−1] = d[N−1] / b[N−1]
  for i = N−2 .. 0:
    x[i] = (d[i] − c[i] · x[i+1]) / b[i]
  return x
```

Stable pour les matrices à diagonale dominante ou SPD (notre cas) ;
pas de pivot partiel nécessaire.

---

## 2. SOR red-black 2D

Solveur itératif pour `A V = b` où `A` est le Laplacien discret 2D.
Utilisé par `fv::Solver2D`. Code :
[`src/fv/solver2d.cpp`](../src/fv/solver2d.cpp).

```
function sor_redblack(V, rho, Vw, Ve, Vs, Vn, Vc, omega, tol):
  repeat:
    max_diff = 0
    for color in (red, black):
      # red = (i+j) pair, black = (i+j) impair
      for each cell (i, j) of current color:
        # Moyenne pondérée des 4 voisins (stencil FV)
        s = Vw(i,j)·V(i−1,j) + Ve(i,j)·V(i+1,j)
          + Vs(i,j)·V(i,j−1) + Vn(i,j)·V(i,j+1)
        V_gs  = (s + rho(i,j)) / Vc(i,j)           # Gauss-Seidel step
        V_new = (1 − omega)·V(i,j) + omega · V_gs  # SOR relaxation
        max_diff = max(max_diff, |V_new − V(i,j)|)
        V(i,j) = V_new
  until max_diff < tol
```

Choix `ω_opt = 2 / (1 + sin(π/N))` pour une grille N×N : converge en
`O(N)` itérations vs `O(N²)` pour ω=1 (Gauss-Seidel pur). Le
red-black split rend chaque demi-sweep **data-parallel** (aucune
cellule rouge ne partage un voisin avec une autre cellule rouge).
L'implémentation fait les deux demi-sweeps en place pour économiser
un buffer scratch.

---

## 3. DST-I spectrale 2D

Inversion **exacte** du Laplacien discret en `O(N² log N)` via deux
transformées sinus discrètes. Utilisé par `spectral::DSTSolver2D`.
Code : [`src/spectral/dst2d.cpp`](../src/spectral/dst2d.cpp).

Sur une grille node-centered N×N avec Dirichlet homogène, les vecteurs
propres du Laplacien discret sont `sin(k π i/(N+1)) · sin(l π j/(N+1))`
pour `k,l = 1..N`, de valeurs propres :

```
λ_{k,l} = 4 · sin²(k π / (2(N+1))) / hx²
        + 4 · sin²(l π / (2(N+1))) / hy²
```

```
function dst_solve(rho):                      # rho(N, N)
  # Forward DST sur chaque direction (FFTW RODFT00)
  rho_hat = DST2D(rho)
  # Division spectrale par les valeurs propres, précomputées
  V_hat   = rho_hat * lambda_inv              # cwiseProduct
  # Backward DST
  V = DST2D(V_hat)
  # Normalisation orthonormale
  V *= 1 / (2·(N+1))
  return V
```

Le solveur construit les plans FFTW une seule fois à la construction
(`FFTW_MEASURE`) et les réutilise pour chaque solve.

---

## 4. Quadtree AMR + stencil FV hétérogène

Maillage adaptatif où chaque cellule est identifiée par une clé Morton
`uint64_t` empaquetant `(level, i, j)`. Code :
[`src/amr/quadtree.cpp`](../src/amr/quadtree.cpp),
[`src/amr/solver.cpp`](../src/amr/solver.cpp).

### Construction

```
function build(tree, predicate, level_max, rho_func):
  repeat:
    to_refine = [key in tree.leaves if predicate(key) and level(key) < level_max]
    for key in to_refine:
      tree.refine(key)              # remplace la feuille par ses 4 enfants
  until to_refine is empty
  balance_2to1(tree)                # force |level(voisin) − level(k)| ≤ 1
  for each leaf:
    leaf.rho = rho_func(leaf.x_center, leaf.y_center)
```

### Stencil FV hétérogène à l'interface coarse–fine

Pour une cellule feuille face à un voisin sur une face donnée :

| Configuration du voisin | `diag +=` | `off =` |
|---|---|---|
| Bord du domaine (Dirichlet V=0) | 2 | 0 |
| Même niveau | 1 | 1 (×1 voisin) |
| Plus grossier | 2/3 | 2/3 (×1 voisin) |
| Plus fin (2 voisins) | 4/3 | 2/3 (×2 voisins) |

Ces poids sont **localement conservatifs** : l'identité `Σ F_face = h²·ρ`
est vérifiée à chaque cellule par
[`test_conservation.cpp`](../tests/test_conservation.cpp). La
contrainte 2:1 (pas plus d'un niveau d'écart entre voisins) garantit
que ce stencil suffit ; sans elle, il faudrait des règles pour 4:1, 8:1,
etc.

### SOR AMR

Après `extract_arrays` qui aplatit l'arbre en `AMRArrays` (vecteurs
`V`, `rho`, `h`, `Vc` + matrices de voisins `nb0`, `nb1` et poids `w0`,
`w1`) :

```
function amr_sor(arr, omega, eps0, tol):
  # Pré-calcul hors hot loop
  rhs    = arr.h² · arr.rho / eps0
  Vc_inv = 1 / arr.Vc

  repeat:
    max_diff = 0
    for i = 0 .. N−1:
      s = 0
      for d in (N, S, E, W):
        if arr.nb0(i, d) >= 0:  s += arr.w0(i, d) · V(nb0(i, d))
        if arr.nb1(i, d) >= 0:  s += arr.w1(i, d) · V(nb1(i, d))
      V_new = (1 − omega) · V(i) + omega · (s + rhs(i)) · Vc_inv(i)
      max_diff = max(max_diff, |V_new − V(i)|)
      V(i) = V_new
  until max_diff < tol
```

---

## 5. V-cycle multigrille uniforme

Accélère la convergence d'un smoother GS en travaillant sur plusieurs
échelles de grille. Complexité `O(N²)` totale avec un bon smoother.
Code : [`src/mg/vcycle.cpp`](../src/mg/vcycle.cpp).

```
function vcycle(V, rho, h, n_pre, n_post, n_min):
  gs_smooth(V, rho, h, n_pre)              # pré-lissage sur grille fine
  if V.size ≤ n_min:
    gs_smooth(V, rho, h, 50)               # "exact solve" à la plus grossière
    return V
  r       = rho − laplacian_fv(V, h)       # résidu sur grille fine
  r_c     = restrict_avg(r)                # restriction : moyenne 4-cellules
  delta_c = zeros(r_c.shape)
  delta_c = vcycle(delta_c, r_c, 2·h, ...) # récursion sur grille 2× plus grosse
  V += prolongate_bilinear(delta_c)        # interpolation bilinéaire
  gs_smooth(V, rho, h, n_post)             # post-lissage
  return V
```

Restriction par moyenne simple (4 cellules → 1). Prolongation
bilinéaire exacte sur les polynômes affines (testée dans
[`test_prolongate.cpp`](../tests/test_prolongate.cpp)).

---

## 6. V-cycle composite 2-grid sur AMR

Accélère la convergence du SOR sur AMR en couplant avec un V-cycle
uniforme sur une grille grossière équivalente. Code : même fichier,
fonction `vcycle_amr_composite`.

```
function vcycle_amr_composite(arr, tree, n_pre, n_post, n_coarse_cycles,
                              omega, eps0):
  # 1. Pré-lissage SOR sur les feuilles AMR
  amr_sor(arr, omega, eps0, max_iter = n_pre)

  # 2. Résidu AMR
  r = amr_residual(arr)

  # 3. Restriction volume-pondérée vers grille uniforme au niveau level_min
  N_c = 2^level_min
  r_c = zeros(N_c, N_c)
  for each leaf (x, y, h, r_i):
    (I, J) = coord_at_level_min(x, y)
    weight = 4^(level_min − level(leaf))    # conservation de la charge
    r_c(I, J) += weight · r_i

  # 4. Résolution approchée A · delta = r_c sur grille uniforme fine
  delta_c = zeros(N_c, N_c)
  for k = 1 .. n_coarse_cycles:
    delta_c = vcycle(delta_c, r_c / h_c², h_c, ...)

  # 5. Prolongation bilinéaire du correctif δ vers les feuilles AMR
  for each leaf (x, y):
    arr.V(leaf) += bilinear_interpolate(delta_c, x, y)

  # 6. Post-lissage SOR
  amr_sor(arr, omega, eps0, max_iter = n_post)
```

Facteur de réduction par cycle observé ~0.7 (non-Galerkin coarse
operator). Un vrai multigrille Galerkin atteindrait ~0.1.

---

## 7. Conjugate Gradient (Krylov)

Solveur itératif pour `A x = b` avec `A` SPD. Converge en `O(√κ)`
itérations où κ = λ_max/λ_min est le conditionnement de A. Pour le
Laplacien 2D Dirichlet/Neumann, κ ~ N² donc `O(N)` itérations.

Code : [`include/poisson/iter/cg.hpp`](../include/poisson/iter/cg.hpp)
(templated, matrix-free).

```
function cg(apply, x, b, tol, max_iter):      # apply(x) = A·x
  r  = b − apply(x)
  d  = r
  rr = r·r
  for k = 1 .. max_iter:
    Ad    = apply(d)
    alpha = rr / (d · Ad)
    x    += alpha · d
    r    -= alpha · Ad
    rr_new = r·r
    if sqrt(rr_new) / ||b|| < tol: break
    beta = rr_new / rr
    d    = r + beta · d
    rr   = rr_new
  return x
```

Version préconditionnée (PCG) avec `M⁻¹`, ici Jacobi `M = diag(A)` :

```
function pcg(apply, precond, x, b, tol, max_iter):   # precond(r) = M⁻¹·r
  r  = b − apply(x)
  z  = precond(r)
  d  = z
  rz = r·z
  for k = 1 .. max_iter:
    Ad    = apply(d)
    alpha = rz / (d · Ad)
    x    += alpha · d
    r    -= alpha · Ad
    if ||r|| / ||b|| < tol: break
    z     = precond(r)
    rz_new = r·z
    beta  = rz_new / rz
    d     = z + beta · d
    rz    = rz_new
  return x
```

Pour notre opérateur FV 2D (Dirichlet x, Neumann y), on replie d'abord
les valeurs de bord dans un rhs effectif :

```
rhs_bc = rho / eps + 2·uL/dx²·e_{i=0} + 2·uR/dx²·e_{i=Nx−1}
```

puis CG s'applique sur l'opérateur zero-BC (strictement SPD).

---

## Qui utiliser quand

| Problème | Solveur | Pourquoi |
|---|---|---|
| Poisson 1D (quelconque BC, ε variable) | `fv::Solver1D` | Thomas direct O(N) |
| Dirichlet homogène 2D, ε uniforme | `spectral::DSTSolver2D` | O(N² log N), pas d'itérations |
| Dirichlet + Neumann mixte, ε uniforme | `iter::solve_poisson_cg` | O(√κ) iter, ~5× plus rapide que SOR |
| ε variable forte ou grille anisotrope | `iter::solve_poisson_cg` avec Jacobi | Le preconditioner capture la variation |
| Compat TPs (ω_opt auto) | `fv::Solver2D` | Identique aux notebooks |
| Source localisée, besoin de raffinement | `amr::Quadtree` + `sor` | Réduit le nb d'inconnues ×10 |

## Références

- Volumes finis : LeVeque, *Finite Volume Methods for Hyperbolic
  Problems*, Cambridge, 2002.
- SOR ω_opt : Varga, *Matrix Iterative Analysis*, Springer, 2000.
- DST spectral : Press et al., *Numerical Recipes*, §20.4.
- CG / Krylov : Saad, *Iterative Methods for Sparse Linear Systems*,
  SIAM, 2003, §6.
- Multigrille : Briggs, Henson, McCormick, *A Multigrid Tutorial*,
  SIAM, 2000.
- AMR : Berger & Colella, *Local adaptive mesh refinement for shock
  hydrodynamics*, J. Comput. Phys. 82 (1989).
- Morton encoding : Samet, *The Design and Analysis of Spatial Data
  Structures*, Addison-Wesley, 1990.
