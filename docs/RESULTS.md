# Résultats et interprétation (style TPs)

Ces figures reproduisent, à partir de la librairie C++, les visualisations
des notebooks `CourseOnPoisson/notebooks/TP{1..5}_*.ipynb`. Elles sont
générées par [`python/plot_tp_style.py`](../python/plot_tp_style.py) qui
appelle la librairie soit via le module pybind11 (`poisson_cpp`), soit via
les snapshots JSON écrits par `poisson_demo --output`.

## Reproduire

```bash
# build (avec pybind11 + FFTW3)
cmake -B build -DPOISSON_BUILD_PYTHON=ON
cmake --build build -j

# snapshot AMR (seul cas qui passe par le CLI + JSON)
./build/examples/poisson_demo --problem amr --Nmin 3 --Nmax 6 \
    --sigma 0.04 --output data/snapshots/amr.json

# génération des figures
PYTHONPATH=build/python python3 python/plot_tp_style.py all
```

Toutes les figures sont écrites dans [`docs/figures/`](figures/).

## Résumé des vérifications numériques

| TP | Quantité mesurée | Valeur | Référence / attendu | Verdict |
|---|---|---|---|---|
| TP1 | L∞ relative vs analytique | 3.85×10⁻¹⁵ | ε_mach ≈ 2.2×10⁻¹⁵ | ✅ précision machine |
| TP1 | ‖V_cpp − V_scipy‖∞ | 7.11×10⁻¹⁴ | ULP-level | ✅ notre Thomas ≡ LAPACK |
| TP2 | D continuity (max−min)/\|moy\| | 2.2×10⁻¹³ | eps_mach | ✅ conservation exacte |
| TP3 | y-indépendance `std(V, axis=1)` | 2.6×10⁻¹¹ | ≈ tol SOR | ✅ Neumann y parfait |
| TP3 | err vs rampe linéaire | 3.3×10⁻⁹ | ≈ tol SOR | ✅ |
| TP4 | pente log-log vs continu | +2.000 | +2.0 (O(h²)) | ✅ exact |
| TP4 | err vs mode propre discret | 4.4×10⁻¹⁶ | ε_mach | ✅ DST inverse exact |
| TP5 | Q_total = ∫∫ρ dA | 0.00503 | πσ² = 0.00503 | ✅ 0 % d'écart |
| TP5 | V_peak AMR vs DST 255² | 0.5 % | < 1 % (stencil ordre 1) | ✅ cohérent |

Tous les "tests de cohérence" passent à la précision attendue pour chaque
méthode (machine pour les solveurs directs, tolérance SOR pour les itératifs,
ordre de discrétisation pour les comparaisons avec une solution continue).

---

## TP1 — Poisson 1D avec BCs de Dirichlet

![TP1](figures/tp1_poisson_1d.png)

**Problème** : `V''(x) = 0` sur `(0, L)` avec `V(0) = uL`, `V(L) = uR`.
Solution analytique : rampe linéaire `V(x) = uL + (uR − uL) · x / L`.

**Solveur** : `poisson::fv::solve_poisson_1d` — volumes finis à 3 points +
Thomas tridiagonal (algorithme O(N) direct).

**Observations** :
- `‖V_num − V_analytique‖∞ = 2.13×10⁻¹⁴`, soit **3.85×10⁻¹⁵ en relatif** à N=100.
  Le plancher théorique est `ε_mach · ‖V‖∞ ≈ 2.22×10⁻¹⁵` (ligne en pointillés
  sur la figure) ; l'erreur mesurée est ~10× au-dessus, ce qui correspond à
  l'accumulation attendue des ~N opérations de Thomas en double précision.
- Le schéma VF est **exact sur les polynômes de degré ≤ 2** (la troncature
  `h² · V''''(x)/12` s'annule pour V affine ou parabolique). La solution
  linéaire tombe dans ce cadre → erreur = 0 à l'arithmétique flottante près.
- Cette assertion est figée par [`tests/test_invariants.cpp`](../tests/test_invariants.cpp)
  (invariant "polynomial exactness", tolérance 1e-12).

### Pourquoi la forme de l'erreur ponctuelle est asymétrique

Le panneau de droite montre une **bosse** entre x=0.1 et x=0.4 culminant à
~2×10⁻¹⁴, puis une décroissance vers ~4×10⁻¹⁶ vers x=1. Ce pattern
**n'est pas un bug** ; il traduit la façon dont l'algorithme de Thomas
accumule les erreurs d'arrondi :

1. **Forward sweep** (élimination gaussienne, i = 1 → N−1) : chaque étape
   combine les deux cellules voisines. Les arrondis s'accumulent
   proportionnellement à i. Le milieu gauche est le "pire" quand on
   pondère par |V| (qui y est ~8).
2. **Back substitution** (i = N−1 → 0) : repart de la condition uR = 0
   où V est petit, donc l'erreur est "re-pincée" à ~0 côté droit.

Le TP1 Python donne le même type de pattern (il ne l'affiche simplement pas
— il se contente d'un `np.max(err)`). Toute bibliothèque tridiagonale directe
en double (numpy.linalg.solve, scipy.linalg.solve_banded, LAPACK DGTSV)
produit la même signature.

**Conclusion** : le code est correct. La "bizarrerie" visuelle est juste un
artefact d'affichage log-scale d'une erreur à la précision machine —
masquer les zéros exacts et tracer la ligne `ε_mach · ‖V‖∞` clarifie
l'interprétation.

### Vérification externe contre `scipy.linalg.solve_banded`

`plot_tp_style.py` résout en parallèle le même système tridiagonal via
`scipy.linalg.solve_banded` (LAPACK `DGTSV`) et compare :

```
‖V_cpp − V_scipy‖∞ = 7.11×10⁻¹⁴
```

Cette différence entre deux implémentations indépendantes de la même
méthode est du **même ordre** que l'écart à l'analytique (`2×10⁻¹⁴`),
confirmant que tout le budget d'erreur est absorbé par les arrondis
d'ordre des opérations — il n'y a pas de biais numérique imputable à
notre Thomas.

---

## TP2 — Couches diélectriques 1D (continuité de D)

![TP2](figures/tp2_dielectric.png)

**Problème** : `∇·(ε ∇V) = 0` sur `[0, 1]` avec 3 couches de permittivité
relative `ε_r(x) = {5, 1, 2}` et Dirichlet `V(0) = 15, V(1) = 0`. Pas de
charge libre : `∇·D = 0` ⟹ `D = ε·E` doit être **constant** à travers
tout le domaine, y compris les interfaces.

**Solveur** : Thomas (via le module pybind) sur le système tridiagonal
issu du schéma VF avec moyenne harmonique de `ε` aux faces.

**Observations** :
- **Panneau de gauche** : `V(x)` piecewise-linéaire avec pentes
  inversement proportionnelles à `ε_r` (pente quintuple dans la couche
  centrale `ε_r = 1` par rapport à la première couche `ε_r = 5`).
- **Panneau central** : `E(x) = -dV/dx` présente deux sauts bien visibles
  aux interfaces `x = 0.3` et `x = 0.7`. Le saut de E à chaque interface
  vérifie `E_1 · ε_1 = E_2 · ε_2` (continuité du flux).
- **Panneau de droite** : `D(x) = ε · E(x)` est constant à
  **(max−min)/|moyenne| = 2.2×10⁻¹³** près — la précision machine.
  La ligne pointillée est la valeur théorique
  `D_theo = ε₀ · (uL − uR) / (dx · Σ 1/ε_face)` (circuit équivalent de
  3 condensateurs en série).

Ce test est équivalent à celui de [`tests/test_conservation.cpp`](../tests/test_conservation.cpp)
("D_n is continuous across interfaces"). Principe physique issu des
notes de cours §4.2 (*équation de Poisson avec diélectrique*).

---

## TP3 — SOR red-black 2D

![TP3](figures/tp3_sor2d.png)

**Problème** : `∇²V = 0` sur `[0, 1]² `, Dirichlet en x (`V(0,y) = uL`,
`V(1,y) = uR`), Neumann en y (`∂V/∂y = 0`). Pas de source. Attendu :
rampe linéaire indépendante de y.

**Solveur** : `poisson::fv::Solver2D` — volumes finis cell-centered,
SOR red-black avec `ω_opt = 2 / (1 + sin(π/N))` auto-sélectionné.

**Observations** (N = 64) :
- **Panneau de gauche** : heatmap 2D. V ne dépend que de x (comme attendu
  sous Neumann en y). Vérification numérique : `std(V, axis=1).max() =
  2.6×10⁻¹¹`, soit 40× le résidu d'arrêt — l'indépendance en y est
  parfaitement capturée.
- **Panneau central** : coupe `y = 0.5`. Le SOR coïncide avec la rampe
  analytique à **3.3×10⁻⁹** près. Note : les points SOR sont aux
  **centres de cellule** `x_i = (i+0.5)·dx`, pas aux faces — c'est
  pourquoi le premier point apparaît à `V ≈ 0.08` et le dernier à
  `V ≈ 9.92` (cellules décalées d'un demi-pas par rapport au bord où
  `V = uL`, `V = uR`).
- **Panneau de droite** : convergence en semi-log. Décroissance quasi
  géométrique du résidu `max|V^{k+1} − V^k|` jusqu'à 10⁻¹⁰ en ~743
  itérations. La pente est contrôlée par `ω_opt` ; un ω=1 (Gauss-Seidel)
  requerrait ~10× plus d'itérations, cf. TP3 du cours.

**Complexité** : `O(N²)` par balayage × `O(N)` itérations avec ω_opt ⇒
`O(N³)` total. Conforme à l'analyse du TP3 du cours.

---

## TP4 — Convergence spectrale O(h²)

![TP4](figures/tp4_spectral_convergence.png)

**Problème manufacturé** : `V(x,y) = sin(πx/L)·sin(πy/L)` avec
`−∇²V = 2(π/L)² V` sur la boîte unité avec Dirichlet homogène.

**Solveur** : `poisson::spectral::DSTSolver2D` — DST-I via FFTW,
diagonalisation exacte du Laplacien discret.

**Observations** :
- **Courbe bleue (cercles, pente ≈ +2.000)** : erreur vs la solution
  **continue** `sin(πx)·sin(πy)`. À N = 511 (h ≈ 2×10⁻³), L∞ ≈ 3×10⁻⁶.
  C'est purement de l'**erreur de discrétisation** (différence entre
  valeurs propres continues `(kπ/L)²` et discrètes `4 sin²(kπh/2)/h²`),
  pas de la méthode.
- **Courbe rouge (carrés)** : erreur vs le **mode propre discret** —
  on construit `ρ = λ_{1,1}^{disc} · V` avec les valeurs propres
  discrètes et on vérifie qu'on récupère exactement `V`. Résultat
  **constant à ~5×10⁻¹⁶** sur toutes les résolutions = **précision
  machine**, indépendamment de h. Le DST-I inverse le Laplacien
  discret *exactement*.
- Vérifié également par [`tests/test_invariants.cpp`](../tests/test_invariants.cpp)
  ("DSTSolver2D recovers a 2D discrete eigenmode to machine precision").

**Coût** : `O(N² log N)` par solve — dominé par les deux DSTs. À N = 512
un solve prend ~8 ms (vs ~2.5 s pour SOR à même précision), soit **×300**
en vitesse. Compromis : la DST exige Dirichlet homogène sur les 4 faces.

---

## TP5 — AMR quadtree avec Gaussienne centrée

![TP5](figures/tp5_amr.png)

**Problème** : `−ε₀ ∇²V = ρ(x, y)` avec `ρ` = Gaussienne centrée
(σ = 0.04) dans une boîte `[0, 1]²` à la terre. Raffinement adaptatif
autour de la zone `r² < (4σ)²`, 3 ≤ level ≤ 6.

**Solveur** : `poisson::amr::extract_arrays + sor` — discrétisation FV
hétérogène avec poids stencil `{2, 1, 2/3, 4/3}` aux interfaces
coarse-fine (localement conservative).

**Observations** (run : `--Nmin 3 --Nmax 6 --sigma 0.04`) :
- **400 feuilles** au total : level 3 → 48, level 4 → 32, level 5 → 64,
  level 6 → 256 (cœur de la Gaussienne, h ≈ 1/64).
- Un maillage uniforme à level 6 aurait 4⁶ = **4 096 cellules** ⇒ gain
  **×10.2** en nombre d'inconnues pour une précision équivalente dans
  la zone d'intérêt.

### Vérifications physiques

1. **Charge totale par loi de Gauss** :
   ```
   ∫∫ ρ dx dy (numérique) = 0.00503
   πσ² (théorique sur R²)  = 0.00503    → err rel = 0.0 %
   ```
   La somme discrète `Σ ρ_i · h_i²` sur toutes les feuilles reproduit
   exactement `πσ²` (intégrale analytique de `exp(-r²/σ²)` sur le plan).
   Confirme que (a) le schéma d'évaluation de `ρ` au centre de cellule
   est cohérent, et (b) la décroissance exponentielle de la Gaussienne
   fait que la contribution hors-boîte est négligeable à notre σ.

2. **Comparaison V_peak vs DST2D sur maillage uniforme fin** :
   ```
   V_peak AMR (400 cellules)  = 2.3006×10⁻³
   V_peak DST2D à N = 255²    = 2.3130×10⁻³
   écart relatif              = 0.5 %
   ```
   Le solveur AMR (avec stencil hétérogène aux interfaces coarse-fine,
   stopping tol = 10⁻⁷) reproduit le DST2D uniform-fin à **0.5%** près,
   avec ~10× moins de cellules. L'écart vient principalement de
   l'ordre-1 du stencil aux interfaces 2:1 — d'où la motivation de
   la prolongation bilinéaire dans le V-cycle composite.

3. **Contrainte 2:1** entre cellules voisines : enforcée par
   `Quadtree::balance_2to1` et vérifiée par
   [`tests/test_quadtree.cpp`](../tests/test_quadtree.cpp).

4. **Conservation locale des flux** : garantie par construction du
   stencil hétérogène, vérifiée par
   [`tests/test_conservation.cpp`](../tests/test_conservation.cpp)
   (test "per-cell flux balance to tolerance").

---

## CG — Gradient Conjugué (méthode de Krylov)

![CG convergence](figures/cg_convergence.png)

![CG scaling](figures/cg_scaling.png)

**Problème** : `−∇²V = ρ` sur `[0, 1]²` avec Dirichlet uL=0, uR=10 en x
et Neumann en y, ρ = 0 (rampe linéaire). Trois solveurs itératifs
comparés à la même tolérance (10⁻¹⁰).

**Solveur** : `poisson::iter::solve_poisson_cg` — matrix-free, templated,
fold des BCs Dirichlet dans le RHS puis CG pur sur l'opérateur SPD.

**Observations (N = 128, figure convergence)** :
- **CG** : **187 itérations**, 8.3 ms. Plateau caractéristique à
  ~10⁻² pendant les premières ~170 itérations (apprentissage des
  modes propres), puis "falaise super-linéaire" brutale jusqu'à
  10⁻¹⁰. Comportement typique : en arithmétique exacte, CG termine
  en ≤ N itérations ; en flottant, 1-2 iters supplémentaires.
- **PCG (Jacobi)** : **442 itérations**, 22.8 ms. *Plus lent* que
  CG sur ce problème à ε uniforme (diagonale quasi-constante →
  Jacobi distrait le subspace Krylov). PCG devient utile quand la
  diagonale varie fortement.
- **SOR ω_opt** : **1 443 itérations**, 52.8 ms. Décroissance
  géométrique régulière sans plateau ni falaise.

**Scaling (figure scaling)** :
- Itérations : CG et SOR *les deux* en O(N), mais avec CG ~5-6× moins
  d'itérations que SOR à chaque résolution.
- Wall time : les deux en O(N³) (N² travail par iter × O(N) iter).
  Constante : CG ~5× plus rapide que SOR.

**Interprétation** :
- Pour la rampe linéaire (Laplace homogène, V = rampe), CG trouve le
  bon sous-espace Krylov tout de suite et converge en ~180 itérations
  à N=128. SOR doit propager l'information cellule par cellule via
  ~7N/2 itérations.
- L'allure "plateau + falaise" de CG est la signature de l'orthogonalité
  des directions de recherche : tant que l'espace Krylov n'a pas capturé
  tous les modes lents, la norme L² du résidu reste élevée ; une fois
  capturés, la convergence devient super-linéaire.

**Usage depuis Python** (via pybind, `solve_poisson_cg`) :
```python
import numpy as np
import poisson_cpp as pc

N = 128
g   = pc.Grid2D(1.0, 1.0, N, N)
V   = np.zeros((N, N), order="F")
rho = np.zeros((N, N), order="F")
report, history = pc.solve_poisson_cg(
    V, rho, g, uL=0.0, uR=10.0, tol=1e-10, record_history=True)
# report.iterations = 187,  len(history) = 188
```

---

## Pourquoi utiliser pybind11 plutôt que juste le CLI JSON ?

Le **CLI** (`poisson_demo`) reste le chemin canonique pour les snapshots
figés et les tests de non-régression. Le **module pybind11**
(`poisson_cpp`) est plus pratique quand on veut :

- Scanner des paramètres (ω, σ, N) sans recompiler ni relancer le binaire.
- Combiner plusieurs solveurs dans une même session (ex. DST en référence
  + SOR en itératif, comme le font les TPs).
- Tracer directement les résultats sans passer par du sérialize/parse.

Exemple interactif :
```python
import numpy as np
import poisson_cpp as pc

g   = pc.Grid2D(1.0, 1.0, 128, 128)
sor = pc.Solver2D(g, 1.0, 0.0, 10.0)
rho = np.zeros((128, 128))
V, report = sor.solve(rho, tol=1e-10)
print(report)              # SORReport(iterations=965, residual=9.77e-11)
```

Les matrices Eigen sont auto-converties en `numpy.ndarray` (avec un copy
pour respecter l'ordre colonnes/lignes). La méthode `.solve_inplace(V, rho)`
accepte des arrays Fortran-ordered (`order='F'`) et évite la copie pour
les boucles très chaudes.

## Pybind11 build flags

- `-DPOISSON_BUILD_PYTHON=ON` active la cible `poisson_py`.
- Requiert Python 3.9+ et pybind11 ≥ 2.11 (compatible C++20). Si
  pybind11 n'est pas installé sur le système, CMake le télécharge via
  FetchContent (tag `v2.13.6`).
- Le module sort à `build/python/poisson_cpp*.so`. Pour l'importer :
  `PYTHONPATH=build/python python`.
