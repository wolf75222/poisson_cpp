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

## TP3 — SOR red-black 2D

![TP3](figures/tp3_sor2d.png)

**Problème** : `∇²V = 0` sur `[0, 1]² `, Dirichlet en x (`V(0,y) = uL`,
`V(1,y) = uR`), Neumann en y (`∂V/∂y = 0`). Pas de source. Attendu :
rampe linéaire indépendante de y.

**Solveur** : `poisson::fv::Solver2D` — volumes finis cell-centered,
SOR red-black avec `ω_opt = 2 / (1 + sin(π/N))` auto-sélectionné.

**Observations** (N = 64) :
- **Panneau de gauche** : heatmap 2D. V ne dépend que de x (comme attendu
  sous Neumann en y).
- **Panneau central** : coupe `y = 0.5`. Le SOR coïncide avec la rampe
  analytique à **~3×10⁻⁹** près (le résidu d'arrêt impose la tolérance).
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
- Pente log-log empirique = **+2.000** (attendu +2 pour un schéma ordre 2).
- À N = 511 (h ≈ 2×10⁻³), l'erreur L∞ vs la solution **continue** vaut
  ~2×10⁻⁵. Cette erreur est purement de **discrétisation** (différence
  entre valeurs propres continues (kπ/L)² et discrètes
  4 sin²(kπh/2)/h²), pas de méthode.
- Si on compare au **mode propre discret** exact (pas au sinus continu),
  l'erreur tombe à **~5×10⁻¹⁶** (précision machine), vérifié par
  [`tests/test_invariants.cpp`](../tests/test_invariants.cpp) — voir
  "DSTSolver2D recovers a 2D discrete eigenmode".

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
- **400 feuilles** au total, réparties :
    - level 3 : 48 cellules (base, h ≈ 1/8)
    - level 4 : 32 cellules
    - level 5 : 64 cellules
    - level 6 : 256 cellules (cœur de la Gaussienne, h ≈ 1/64)
- Un maillage uniforme à level 6 aurait 4⁶ = **4 096 cellules** ⇒ gain
  **×10.2** en nombre d'inconnues pour une précision équivalente dans
  la zone d'intérêt.
- La contrainte 2:1 entre cellules voisines est enforcée par
  `Quadtree::balance_2to1` et vérifiée par
  [`tests/test_quadtree.cpp`](../tests/test_quadtree.cpp).
- Conservation locale des flux : garantie par construction du stencil
  hétérogène, et vérifiée numériquement par
  [`tests/test_conservation.cpp`](../tests/test_conservation.cpp)
  (test "per-cell flux balance to tolerance").

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
