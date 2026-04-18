<div align="center">

# Poisson CPP

**C++20 library of Poisson solvers : finite volume, spectral, AMR + multigrid, Conjugate Gradient.**

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus)
![Tests](https://img.shields.io/badge/tests-66%2F66-brightgreen)
![Build](https://img.shields.io/badge/build-CMake%203.20%2B-064F8C?logo=cmake)
![License](https://img.shields.io/badge/license-BSD--3-green)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)

</div>

![banner](docs/figures/banner.png)

<div align="center">
<sub>
Scène multi-charges (10 gaussiennes ±q) résolue par le solveur AMR :
5 287 feuilles quadtree, SOR convergé à 10⁻⁸ en 300 itérations.
Fond = |V| (viridis), courbes = équipotentielles, points rouges = charges.
Reproduction : <a href="python/make_banner.py"><code>python/make_banner.py</code></a>.
</sub>
</div>

---

## Pourquoi ?

Ce dépôt est un **portage C++** des notebooks pédagogiques
`CourseOnPoisson/notebooks/TP{1..5}_*.ipynb` (Thomas, SOR, spectral DST,
AMR quadtree, multigrille composite). Il fournit une **bibliothèque
réutilisable** pour un simulateur de plasma plus vaste, avec Eigen,
FFTW, OpenMP et Krylov/CG.

> [!NOTE]
> Les notebooks Python restent l'**oracle** : la librairie est validée par
> 57 tests, dont 4 contre des snapshots JSON produits par les notebooks, et
> 9 cross-checks de cohérence physique (loi de Gauss, énergie, continuité D).

## Aperçu des solveurs

| Module | Algorithme | Coût | BC supportées |
|---|---|---|---|
| [`linalg::thomas`](include/poisson/linalg/thomas.hpp) | Tridiagonal direct | **O(N)** | toutes |
| [`fv::Solver1D`](include/poisson/fv/solver1d.hpp) | FV + Thomas | O(N) | Dirichlet (ε uniforme ou variable) |
| [`fv::Solver2D`](include/poisson/fv/solver2d.hpp) | FV + SOR red-black ω_opt | O(N³) | Dirichlet×Neumann |
| [`iter::solve_poisson_cg`](include/poisson/iter/poisson_cg.hpp) | CG / PCG Jacobi (Krylov) | **O(N³), 5× SOR** | Dirichlet×Neumann |
| [`spectral::DSTSolver2D`](include/poisson/spectral/dst2d.hpp) | DST-I via FFTW | **O(N²·logN)** | Dirichlet homogène |
| [`amr::Quadtree` + `sor`](include/poisson/amr/solver.hpp) | Quadtree Morton + FV hétérogène | O(nb_leaves × iter) | Dirichlet V = 0 au bord |
| [`mg::vcycle_amr_composite`](include/poisson/mg/vcycle.hpp) | V-cycle 2-grid composite AMR | O(nb_leaves) per cycle | Dirichlet V = 0 |

**Tous sont matrix-free** (stencils explicites, zéro matrice sparse stockée).
Voir [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) pour le schéma complet
et les conventions de grille.

## Vérification

Trois couches indépendantes de tests, **66/66 passent** :

| Catégorie | # tests | Tolérance | Exemples |
|---|---|---|---|
| Unit / regression | 27 | variable | SOR converge, thomas exact, quadtree 2:1 |
| Référence Python (snapshots JSON) | 4 | 1e-10 abs | `solve_poisson_1d` ≡ notebook |
| Invariants mathématiques | 11 | **1e-13** | réciprocité Green G(A,B)=G(B,A), linéarité, symétrie |
| Lois de conservation physique | 5 | **1e-12** | Gauss ε₀∮∂V/∂n=Q, énergie ½∫ρV=½ε₀∫\|∇V\|² |
| Benchmark Fourier (Jackson ch.2) | 2 | O(h²) | convergence ratio +2.00 exact |
| CG invariants + cross-check DST | 6 | 1e-10 | linéarité, symétrie, scaling O(N) |
| Prolongation MG | 2 | 1e-13 | bilinéaire exacte sur affines |
| AMR V-cycle | 2 | monotone | décroissance résidu par cycle |
| Benchmarks de performance | 4 | tol SOR | Solver2D, gs_smooth, amr_sor |
| Dielectric 1D | 3 | 1e-12 | continuité D à travers interfaces |

Voir [`docs/RESULTS.md`](docs/RESULTS.md) pour les figures TP-style
(TP1–TP5), [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) pour les
benchmarks A/B et le profiling.

---

## Quick start

### Prérequis

- C++20 (AppleClang 16+, GCC 12+, Clang 15+, MSVC 19.36+)
- CMake ≥ 3.20
- Eigen ≥ 3.4 *(fetched automatiquement si absent)*
- FFTW3 *(optionnel, active le solveur spectral)*
- OpenMP *(optionnel, parallel sweeps à partir de N ≥ 384)*
- Python 3.9+ *(optionnel, bindings pybind11)*

### Build

```bash
git clone https://github.com/wolf75222/poisson_cpp.git
cd poisson_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                              # 66/66 doit passer
```

Options CMake :

| Option | Défaut | Rôle |
|---|---|---|
| `POISSON_BUILD_TESTS` | `ON` | Compile Catch2 + toute la suite |
| `POISSON_BUILD_BENCHMARKS` | `ON` | Compile `bench_solvers` et `profile_cg` |
| `POISSON_USE_OPENMP` | `OFF` | Parallélise SOR + gs_smooth au-delà de N ≥ 384 |
| `POISSON_BUILD_PYTHON` | `OFF` | Génère le module `poisson_cpp` via pybind11 |

### Exécuter le démo C++

```bash
# 1D Poisson + dump JSON
./build/examples/poisson_demo --problem poisson1d --N 100 --uL 10 --uR 0 \
    --output /tmp/p1d.json

# SOR 2D avec historique de convergence
./build/examples/poisson_demo --problem sor2d --N 128 --uR 10 --tol 1e-10

# Spectral 2D (nécessite FFTW)
./build/examples/poisson_demo --problem spectral2d --N 256

# Scène AMR multi-charges (utilisée pour la bannière)
./build/examples/poisson_demo --problem amr_scatter --Nmin 4 --Nmax 7 \
    --sigma 0.025 --output data/snapshots/amr_scatter.json
```

---

## Utilisation depuis Python (pybind11)

Build du module :

```bash
cmake -B build -DPOISSON_BUILD_PYTHON=ON
cmake --build build --target poisson_py -j
export PYTHONPATH=$PWD/build/python
```

### Tutoriel — 3 solveurs en 20 lignes

```python
import numpy as np
import poisson_cpp as pc

# ---- 1. Thomas (tridiagonal direct, 1D) --------------------------------
N = 100
a = np.zeros(N); b = 2.0 * np.ones(N); c = np.zeros(N); d = np.ones(N)
a[1:] = -1.0; c[:-1] = -1.0
x = pc.thomas(a, b, c, d)
print("thomas:", x[:3])                    # [1.0, 2.0, ... ]

# ---- 2. Solver2D — FV + SOR red-black ω_opt auto -----------------------
grid = pc.Grid2D(1.0, 1.0, 64, 64)         # [0,1]² cell-centered
sor  = pc.Solver2D(grid, eps=1.0, uL=0.0, uR=10.0)   # Dirichlet x, Neumann y
rho  = np.zeros((64, 64))
V, report = sor.solve(rho, tol=1e-10)      # returns (V, SORReport)
print(report)                              # SORReport(iterations=..., residual=...)

# ---- 3. Conjugate Gradient (4.2× plus rapide que SOR pour N=128) -------
V_cg = np.zeros((128, 128), order="F")     # Fortran-order pour Eigen in-place
grid = pc.Grid2D(1.0, 1.0, 128, 128)
rep, hist = pc.solve_poisson_cg(
    V_cg, np.zeros((128, 128), order="F"), grid,
    uL=0.0, uR=10.0, tol=1e-10, record_history=True)
print(f"CG: {rep.iterations} iterations, residual {rep.residual:.2e}")

# ---- 4. Spectral DST-I (si FFTW disponible) ----------------------------
if pc.has_fftw3:
    dst = pc.DSTSolver2D(127, 127, 1.0, 1.0, eps0=1.0)
    rho_dst = np.random.randn(127, 127)
    V_dst = dst.solve(rho_dst)
    print("DST2D V range:", V_dst.min(), V_dst.max())
```

### Visualisation des résultats (style notebooks TP)

```bash
python3 python/plot_tp_style.py all           # TP1..TP5 reproduites
python3 python/plot_cg.py                     # convergence CG vs SOR
python3 python/make_banner.py                 # bannière AMR multi-charges
```

Les figures sortent dans [`docs/figures/`](docs/figures/) et sont
interprétées dans [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Résultats

<div align="center">

|  |  |
|---|---|
| ![TP1](docs/figures/tp1_poisson_1d.png) | ![TP4](docs/figures/tp4_spectral_convergence.png) |
| **TP1** — Poisson 1D, erreur L∞ = 2.1×10⁻¹⁴ (précision machine) | **TP4** — Convergence spectrale, pente log-log = **+2.000** exact |
| ![CG vs SOR](docs/figures/cg_convergence.png) | ![TP5](docs/figures/tp5_amr.png) |
| **CG** : 187 iter / 8 ms vs SOR 1443 / 53 ms  (**7.7× moins d'itérations**) | **TP5** — AMR sur Gaussienne centrée : 400 feuilles, **×10 vs uniform** |

</div>

---

## Performance

Optimisations validées par A/B testing local avec `sample` / `xctrace`.

| Optimisation | Kernel | Gain | Référence |
|---|---|---|---|
| In-place red-black SOR + `Vc_inv_` précomputé | `Solver2D::solve` N=128 | **−46 %** | [PERFORMANCE.md §3](docs/PERFORMANCE.md#3-fvsolver2dsolve--in-place-red-black--member-vc_inv_) |
| True in-place RB + suppression scratch buffer | `mg::gs_smooth` N=128 | **−68 %** (×3.1) | [§2](docs/PERFORMANCE.md#2-mggs_smooth--true-in-place-red-black--vc_inv) |
| Précalcul `rhs = h²ρ/ε₀`, `Vc_inv` | `amr::sor` | **−16 %** | [§1](docs/PERFORMANCE.md#1-amr-sor--precompute-rhs--hρε-and-vc_inv--1vc) |
| Fold Dirichlet BC → RHS (débloque SIMD) | `Solver2D::solve` | **−5 %** | [§4](docs/PERFORMANCE.md#4-solver2dsolve--fold-dirichlet-bc-into-effective-rhs-to-unblock-auto-vectorization) |
| `unroll_count(4)` + `std::max` réduction | `Solver2D::solve` | **−1.5 % à −3.6 %** | [§5](docs/PERFORMANCE.md#5-solver2dsolve--pragma-unroll-4--stdmax-reduction) |
| Profile-guided `diag_mat` précomputé | CG `apply_neg_laplacian` N=512 | **−18 %** | [CG section](docs/PERFORMANCE.md#cg-hot-loop-optimisation-profile-guided-diag-precomputation) |
| OpenMP parallel sweeps (opt-in) | Solver2D + gs_smooth N=512 | **×1.5 – ×1.9** | [OpenMP section](docs/PERFORMANCE.md#openmp-parallel-sweeps) |

## License

BSD-3-Clause. Voir [LICENSE](LICENSE).

## Crédits

Port C++ des TPs Python du cours Poisson (`CourseOnPoisson/notebooks/`,
basés sur Berger & Colella, Almgren et al.). Benchmark de référence :
Jackson, *Classical Electrodynamics*, ch. 2. Conventions AMR FV :
TP5 du même cours.
