# poisson_cpp

Bibliothèque C++20 de solveurs de Poisson 1D/2D : volumes finis,
spectral DST, AMR quadtree + multigrille, gradient conjugué. Bindings
Python via pybind11.

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
examples
custom_vcycle
api
```

```{toctree}
:maxdepth: 1
:caption: Référence C++

C++ API (Doxygen) <https://wolf75222.github.io/poisson_cpp/cpp/>
```

## En bref

Sept solveurs matrix-free, tous testés à la précision machine sur les
invariants physiques (Gauss, énergie, continuité de D) :

- `Solver1D` (Thomas, O(N))
- `Solver2D` (FV + SOR red-black ω_opt)
- `solve_poisson_cg` (CG / PCG Jacobi)
- `DSTSolver1D/2D` (FFTW, Dirichlet homogène)
- `Quadtree` + `sor` (AMR cellule par cellule)
- `vcycle_amr_composite` (V-cycle 2-grid composite)

Détail des algorithmes et choix de discrétisation :
[ALGORITHMS.md](https://github.com/wolf75222/poisson_cpp/blob/main/docs/ALGORITHMS.md).

## Liens

- Code source : <https://github.com/wolf75222/poisson_cpp>
- Référence C++ Doxygen : [/cpp/](https://wolf75222.github.io/poisson_cpp/cpp/)
- Licence : BSD-3-Clause
