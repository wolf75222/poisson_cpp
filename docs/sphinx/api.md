# API Python

Référence générée automatiquement depuis le module compilé `poisson_cpp`.
Chaque entrée garde son docstring NumPy-style avec sections
`Parameters`, `Returns` et signature typée.

## Grilles

```{eval-rst}
.. autoclass:: poisson_cpp.Grid1D
   :members:
.. autoclass:: poisson_cpp.Grid2D
   :members:
```

## Solveurs 1D

```{eval-rst}
.. autofunction:: poisson_cpp.thomas
.. autofunction:: poisson_cpp.solve_poisson_1d
.. autofunction:: poisson_cpp.solve_poisson_1d_dielectric
```

## Solveur 2D itératif (SOR)

```{eval-rst}
.. autoclass:: poisson_cpp.Solver2D
   :members:
.. autoclass:: poisson_cpp.SORParams
   :members:
.. autoclass:: poisson_cpp.SORReport
   :members:
```

## Gradient Conjugué

```{eval-rst}
.. autofunction:: poisson_cpp.solve_poisson_cg
.. autoclass:: poisson_cpp.CGParams
   :members:
.. autoclass:: poisson_cpp.CGReport
   :members:
```

## Spectral (FFTW)

```{eval-rst}
.. autoclass:: poisson_cpp.DSTSolver1D
   :members:
.. autoclass:: poisson_cpp.DSTSolver2D
   :members:
```

## AMR quadtree

```{eval-rst}
.. autoclass:: poisson_cpp.Quadtree
   :members:
.. autoclass:: poisson_cpp.Cell
   :members:
.. autoclass:: poisson_cpp.Direction
   :members:
.. autoclass:: poisson_cpp.AMRArrays
   :members:
.. autofunction:: poisson_cpp.extract_arrays
.. autofunction:: poisson_cpp.writeback
.. autofunction:: poisson_cpp.amr_sor
.. autofunction:: poisson_cpp.amr_residual
.. autoclass:: poisson_cpp.AMRSORParams
   :members:
.. autoclass:: poisson_cpp.AMRSORReport
   :members:
.. autofunction:: poisson_cpp.make_key
.. autofunction:: poisson_cpp.level_of
.. autofunction:: poisson_cpp.i_of
.. autofunction:: poisson_cpp.j_of
```

## Multigrille

```{eval-rst}
.. autofunction:: poisson_cpp.gs_smooth
.. autofunction:: poisson_cpp.laplacian_fv
.. autofunction:: poisson_cpp.restrict_avg
.. autofunction:: poisson_cpp.prolongate_const
.. autofunction:: poisson_cpp.prolongate_bilinear
.. autofunction:: poisson_cpp.vcycle_uniform
.. autofunction:: poisson_cpp.vcycle_amr_composite
.. autoclass:: poisson_cpp.CompositeParams
   :members:
```

## Utilitaires

```{eval-rst}
.. autofunction:: poisson_cpp.harmonic_mean
.. autofunction:: poisson_cpp.fftw_install_hint
.. autofunction:: poisson_cpp.dump_amr_snapshot
.. autodata:: poisson_cpp.has_fftw3
.. autodata:: poisson_cpp.__version__
```
