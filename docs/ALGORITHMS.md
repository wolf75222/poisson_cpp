# Algorithmes

Pseudocode de chaque solveur, avec lien vers le fichier C++
correspondant et l'intuition derrière le choix.

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

Traduire cette EDP en système linéaire discret est le rôle de la
discrétisation FV ci-dessous. Ensuite, résoudre ce système efficacement
est le rôle des 7 algorithmes.

## Discrétisation volumes finis (FV)

Le domaine est découpé en cellules. L'intégration de `−∇·(ε∇V) = ρ`
sur chaque cellule et l'application de Green–Ostrogradski transforme
la divergence en flux à travers les faces :

```
−Σ F_face = h^d · ρ_cell           (d = dim, h = taille de cellule)
F_face    = ε_face · (V_out − V_in) / dx
```

Sur grille uniforme 2D ça donne le schéma classique 5-points :

```
(−V_{i−1,j} − V_{i+1,j} − V_{i,j−1} − V_{i,j+1} + 4·V_{i,j}) / h² = ρ_{i,j}/ε
```

Deux choses à noter :

1. **Pourquoi FV et pas différences finies ?** FV est naturellement
   conservatif : la somme des flux entrants/sortants d'une cellule
   égale la source intégrée dessus, à la précision machine. C'est ce
   qui rend `Σ F_face = h²·ρ` vrai par construction.
2. **Pourquoi la moyenne harmonique `ε_face = 2ε_a ε_b / (ε_a + ε_b)` ?**
   À l'interface entre deux diélectriques, le déplacement normal
   `D·n = ε·∂V/∂n` doit être continu (c'est la forme physique de
   `∇·D = 0`). La moyenne harmonique est la seule formule qui préserve
   cette continuité au niveau discret, vérifié par
   [`test_conservation.cpp`](../tests/test_conservation.cpp) à 10⁻¹².

Le schéma FV est **exact sur les polynômes de degré ≤ 2** (la
troncature est `h²·V''''(x)/12` qui s'annule). L'ordre de convergence
vers la solution continue est donc O(h²).

---

## 1. Thomas tridiagonal

Solveur direct O(N) pour un système `A x = d` avec `A` tridiagonale.
Utilisé par `fv::Solver1D` (Poisson 1D). Code :
[`src/linalg/thomas.cpp`](../src/linalg/thomas.cpp).

En 1D, le stencil 3-points donne une matrice tridiagonale. Pour ce
cas particulier on peut résoudre en O(N) via élimination gaussienne
sans pivotage. Gauss général est O(N³) sur une matrice dense, O(N²)
sur une matrice bande ; pour une tridiagonale on descend à O(N).

```
function thomas(a, b, c, d):      # a: sub-diag, b: diag, c: super-diag
  # Forward sweep (élimination gaussienne : met A en forme triangulaire sup.)
  for i = 1 .. N−1:
    m     = a[i] / b[i−1]
    b[i] -= m · c[i−1]
    d[i] -= m · d[i−1]
  # Back substitution (résout le système triangulaire)
  x[N−1] = d[N−1] / b[N−1]
  for i = N−2 .. 0:
    x[i] = (d[i] − c[i] · x[i+1]) / b[i]
  return x
```

Stable pour les matrices à diagonale dominante ou SPD (notre cas :
la diagonale du Laplacien discret est positive et domine les
off-diagonales), donc pas de pivot partiel nécessaire. Le solveur
détecte néanmoins les pivots nuls et jette une exception plutôt que
de propager des NaN.

---

## 2. SOR red-black 2D

Solveur itératif pour `A V = b` où `A` est le Laplacien discret 2D.
Utilisé par `fv::Solver2D`. Code :
[`src/fv/solver2d.cpp`](../src/fv/solver2d.cpp).

**Intuition.** Gauss-Seidel (GS) met à jour chaque cellule à partir
de ses voisins déjà calculés. L'erreur se comporte comme un
multiplicateur `ρ_GS < 1` par itération ; pour le Laplacien sur grille
N×N on a `ρ_GS = 1 − O(1/N²)`, donc il faut O(N²) itérations pour
converger. SOR introduit un facteur de sur-relaxation ω > 1 qui, bien
choisi (`ω_opt`), accélère à O(N) itérations (même ordre que
multigrille pour un coût par itération moindre).

**Pourquoi red-black ?** Séparer les cellules en deux couleurs
(i+j pair = rouge, impair = noir) rend chaque demi-sweep *indépendant*
en intra-couleur : on peut paralléliser (OpenMP, SIMD) sans casser
la convergence sérielle de GS, puisque les mises à jour rouges
n'utilisent que des valeurs noires et vice-versa.

```
function sor_redblack(V, rho, Vw, Ve, Vs, Vn, Vc, omega, tol):
  repeat:
    max_diff = 0
    for color in (red, black):
      # red = (i+j) pair, black = (i+j) impair
      for each cell (i, j) of current color:
        # Moyenne pondérée des 4 voisins (stencil FV hétérogène ε)
        s = Vw(i,j)·V(i−1,j) + Ve(i,j)·V(i+1,j)
          + Vs(i,j)·V(i,j−1) + Vn(i,j)·V(i,j+1)
        V_gs  = (s + rho(i,j)) / Vc(i,j)           # Gauss-Seidel step
        V_new = (1 − omega)·V(i,j) + omega · V_gs  # SOR relaxation
        max_diff = max(max_diff, |V_new − V(i,j)|)
        V(i,j) = V_new
  until max_diff < tol
```

Avec `ω_opt = 2 / (1 + sin(π/N))` sur une grille N×N la convergence
est O(N) itérations (vs O(N²) pour ω=1, Gauss-Seidel pur). La formule
de ω_opt vient de l'analyse spectrale du Laplacien discret : elle
minimise le rayon spectral de l'opérateur d'itération, cf. Varga,
*Matrix Iterative Analysis*. Les deux demi-sweeps se font en place.

---

## 3. DST-I spectrale 2D

Inversion exacte du Laplacien discret en O(N² log N) via deux
transformées sinus discrètes. Utilisé par `spectral::DSTSolver2D`.
Code : [`src/spectral/dst2d.cpp`](../src/spectral/dst2d.cpp).

**Intuition.** La famille `{sin(k π x/L)}_{k≥1}` est une base
orthogonale de L²([0, L]) qui s'annule en 0 et L, donc qui respecte
automatiquement des BC Dirichlet homogènes. Sur la version discrète
(sinus aux N points intérieurs), ces vecteurs sont les vecteurs
propres du Laplacien discret à 3-points : appliquer le Laplacien dans
cette base revient à multiplier coefficient par coefficient par les
valeurs propres λ_k. L'équation `A V = b` devient `λ_k · V_k = b_k`,
qu'on résout trivialement par division. En 2D c'est le produit
tensoriel des bases 1D.

Les vecteurs propres 2D sont `sin(k π i/(N+1)) · sin(l π j/(N+1))`
pour `k,l = 1..N`, de valeurs propres :

```
λ_{k,l} = 4 · sin²(k π / (2(N+1))) / hx²
        + 4 · sin²(l π / (2(N+1))) / hy²
```

```
function dst_solve(rho):                      # rho(N, N)
  # Forward DST sur chaque direction (FFTW RODFT00), O(N² log N)
  rho_hat = DST2D(rho)
  # Division spectrale par les valeurs propres, précomputées
  V_hat   = rho_hat * lambda_inv              # cwiseProduct
  # Backward DST, O(N² log N)
  V = DST2D(V_hat)
  # Normalisation orthonormale
  V *= 1 / (2·(N+1))
  return V
```

Le solveur construit les plans FFTW une seule fois à la construction
(`FFTW_MEASURE`) et les réutilise pour chaque solve. À N=512, un
solve prend ~8 ms contre ~2.5 s pour SOR à même précision, soit ~×300.
Limitation : n'accepte que Dirichlet homogène partout (pas mixte).

---

## 4. Quadtree AMR + stencil FV hétérogène

Maillage adaptatif où chaque cellule est identifiée par une clé Morton
`uint64_t` empaquetant `(level, i, j)`. Code :
[`src/amr/quadtree.cpp`](../src/amr/quadtree.cpp),
[`src/amr/solver.cpp`](../src/amr/solver.cpp).

**Motivation.** Une gaussienne de largeur σ dans une boîte 1×1
demande h ~ σ/10 pour être bien résolue près du centre, mais nulle
part ailleurs. Un maillage uniforme à cette résolution compte
(10/σ)² cellules. Un quadtree n'en met autant qu'autour de la
gaussienne et laisse le reste grossier : 10× moins d'inconnues pour
la même précision locale (mesuré sur TP5).

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

**Problème.** Aux interfaces entre cellules de tailles différentes,
le stencil 5-points standard ne s'applique pas : une cellule grossière
de côté `2h` est adjacente à deux cellules fines de côté `h`, et la
distance entre centres n'est plus uniforme. Il faut intégrer Gauss
sur chaque face avec les vraies aires et distances.

Le résultat, pour une cellule feuille face à un voisin sur une face
donnée :

| Configuration du voisin | `diag +=` | `off =` |
|---|---|---|
| Bord du domaine (Dirichlet V=0) | 2 | 0 |
| Même niveau | 1 | 1 (×1 voisin) |
| Plus grossier | 2/3 | 2/3 (×1 voisin) |
| Plus fin (2 voisins) | 4/3 | 2/3 (×2 voisins) |

Poids localement conservatifs : l'identité `Σ F_face = h²·ρ` est
vérifiée à chaque cellule par
[`test_conservation.cpp`](../tests/test_conservation.cpp). La
contrainte 2:1 (pas plus d'un niveau d'écart entre voisins) garantit
que ce stencil suffit ; sans elle, il faudrait des règles pour 4:1,
8:1, etc., ce qui explose vite.

**Pourquoi Morton ?** Les clés Morton empilent (level, i, j) de
façon que (a) les enfants d'une cellule soient adjacents en ordre
numérique (bon pour les courbes SFC et le load-balancing) et (b) la
recherche de voisins se fasse par manipulation de bits sans structure
arborescente explicite.

### SOR AMR

Après `extract_arrays` qui aplatit l'arbre en `AMRArrays` (vecteurs
`V`, `rho`, `h`, `Vc` + matrices de voisins `nb0`, `nb1` et poids `w0`,
`w1`), on applique SOR sur le tableau plat. Le stencil est plus
complexe qu'en grille uniforme (chaque cellule peut avoir jusqu'à 6
voisins au lieu de 4 quand elle est plus grossière que deux de ses
voisins), mais l'itération reste la même :

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

Limitation : le coloriage red-black ne marche plus (les voisins ne
sont plus sur une grille régulière), donc la version AMR de SOR est
séquentielle dans sa boucle intérieure. D'où l'intérêt du V-cycle
composite ci-dessous pour accélérer.

---

## 5. V-cycle multigrille uniforme

**Motivation.** Un smoother GS tue efficacement les erreurs à *haute*
fréquence (celles qui oscillent cellule par cellule) mais avance très
lentement sur les erreurs *basse* fréquence (gradient doux à l'échelle
du domaine). Conséquence : GS seul donne O(N²) itérations pour
converger. L'idée multigrille : les erreurs basse fréquence sur une
grille fine deviennent des erreurs *haute* fréquence sur une grille
2× plus grossière, où un smoother GS les tue vite. En cascadant les
niveaux, chaque échelle gère ses propres modes.

Coût total O(N²) avec un bon smoother.
Code : [`src/mg/vcycle.cpp`](../src/mg/vcycle.cpp).

```
function vcycle(V, rho, h, n_pre, n_post, n_min):
  gs_smooth(V, rho, h, n_pre)              # pré-lissage sur grille fine
                                           #  (tue les hautes fréq)
  if V.size ≤ n_min:
    gs_smooth(V, rho, h, 50)               # "exact solve" à la plus grossière
    return V
  r       = rho − laplacian_fv(V, h)       # résidu sur grille fine
  r_c     = restrict_avg(r)                # restriction : moyenne 4-cellules
  delta_c = zeros(r_c.shape)
  delta_c = vcycle(delta_c, r_c, 2·h, ...) # récursion : correctif sur grille 2× plus grosse
  V += prolongate_bilinear(delta_c)        # interpolation bilinéaire
  gs_smooth(V, rho, h, n_post)             # post-lissage
                                           #  (tue les hautes fréq réintroduites
                                           #   par l'interpolation)
  return V
```

Restriction par moyenne simple (4 cellules → 1). Prolongation
bilinéaire (exacte sur les polynômes affines, testée dans
[`test_prolongate.cpp`](../tests/test_prolongate.cpp)). Le post-
smoothing est important : la prolongation introduit des petites
oscillations haute fréquence que GS nettoie vite.

---

## 6. V-cycle composite 2-grid sur AMR

**Motivation.** Le SOR AMR de la section 4 est séquentiel et
lentement convergent (pas de coloriage red-black possible sur les
feuilles irrégulières). Le V-cycle composite couple le SOR AMR (bon
sur les détails du maillage adaptatif) avec un V-cycle uniforme sur
une grille de fond (efficace sur les modes globaux). On paie un
petit coût de projection entre les deux, on y gagne plusieurs ordres
de grandeur sur le nombre d'itérations nécessaires.

Code : même fichier, fonction `vcycle_amr_composite`.

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

Facteur de réduction par cycle observé ~0.7 (re-discrétisation sur la
grille grossière, pas Galerkin). Un multigrille Galerkin (avec
`A_coarse = R · A_fine · P` construit explicitement) atteindrait ~0.1,
mais demande beaucoup plus de machinerie.

---

## 7. Conjugate Gradient (Krylov)

Solveur itératif pour `A x = b` avec `A` SPD (symétrique définie
positive). Code :
[`include/poisson/iter/cg.hpp`](../include/poisson/iter/cg.hpp)
(templated, matrix-free).

**Intuition.** CG construit à chaque itération une direction de
recherche `d_k` orthogonale à toutes les précédentes *au sens du
produit scalaire A* (`<d_i, A d_j> = 0` pour i ≠ j), ce qui minimise
l'erreur dans la norme énergie `||e||_A²`. Après k itérations,
l'approximation `x_k` est la meilleure possible dans le sous-espace
de Krylov `span(r₀, A r₀, A² r₀, ..., A^{k-1} r₀)`. En arithmétique
exacte, CG termine en ≤ N itérations ; en pratique on arrête bien
avant, quand le résidu passe sous la tolérance.

Convergence en O(√κ) itérations où κ = λ_max / λ_min est le
conditionnement de A. Pour le Laplacien 2D Dirichlet/Neumann, κ ~ N²
donc CG converge en O(N) itérations, même ordre que SOR ω_opt mais
avec une constante ~5× plus petite (mesurée).

**Exigence : A doit être SPD.** Pour notre opérateur FV 2D avec
Dirichlet en x, on replie les valeurs de bord dans un rhs effectif :

```
rhs_bc = rho / eps + 2·uL/dx²·e_{i=0} + 2·uR/dx²·e_{i=Nx−1}
```

CG s'applique alors sur l'opérateur zero-BC, qui est SPD.

```
function cg(apply, x, b, tol, max_iter):      # apply(x) = A·x
  r  = b − apply(x)
  d  = r
  rr = r·r
  for k = 1 .. max_iter:
    Ad    = apply(d)
    alpha = rr / (d · Ad)                     # taille du pas
    x    += alpha · d                         # nouveau x
    r    -= alpha · Ad                        # nouveau résidu
    rr_new = r·r
    if sqrt(rr_new) / ||b|| < tol: break
    beta = rr_new / rr                        # correction A-orthogonale
    d    = r + beta · d                       # nouvelle direction
    rr   = rr_new
  return x
```

### PCG (préconditionné)

**Motivation.** Si A est mal conditionnée, CG rame. On peut
pré-transformer le système en `M⁻¹ A x = M⁻¹ b` avec `M` une
approximation bon marché de A, pour que `M⁻¹ A` ait un meilleur
conditionnement. Le choix le plus simple est Jacobi : `M = diag(A)`.

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

Caveat mesuré : à ε uniforme, la diagonale du stencil FV est presque
constante à l'intérieur (4/h²) et Jacobi n'apporte rien, voire
ralentit CG (mesure : 442 iter pour PCG vs 187 pour CG à N=128).
Jacobi devient utile quand ε varie fortement dans l'espace, car
alors `diag(A)` capte la variation locale.

---

## Qui utiliser quand

| Problème | Solveur | Pourquoi |
|---|---|---|
| Poisson 1D (quelconque BC, ε variable) | `fv::Solver1D` | Thomas direct O(N) |
| Dirichlet homogène 2D, ε uniforme | `spectral::DSTSolver2D` | O(N² log N), pas d'itérations |
| Dirichlet + Neumann mixte, ε uniforme | `iter::solve_poisson_cg` | O(√κ) iter, ~5× plus rapide que SOR |
| ε variable forte ou grille anisotrope | `iter::solve_poisson_cg` avec Jacobi | Le préconditionneur capte la variation |
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
