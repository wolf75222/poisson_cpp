"""
poisson_cpp Python bindings.

Wraps the compiled extension ``_core``. Warns at import if FFTW3 is missing
(spectral solvers disabled) and returns the install command for the
detected platform via ``fftw_install_hint()``.
"""
from __future__ import annotations
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
import json as json
from pathlib import Path
import platform as platform
from poisson_cpp._core import AMRArrays
from poisson_cpp._core import AMRSORParams
from poisson_cpp._core import AMRSORReport
from poisson_cpp._core import CGParams
from poisson_cpp._core import CGReport
from poisson_cpp._core import Cell
from poisson_cpp._core import CompositeParams
from poisson_cpp._core import DSTSolver1D
from poisson_cpp._core import DSTSolver2D
from poisson_cpp._core import Direction
from poisson_cpp._core import Grid1D
from poisson_cpp._core import Grid2D
from poisson_cpp._core import Quadtree
from poisson_cpp._core import SORParams
from poisson_cpp._core import SORReport
from poisson_cpp._core import Solver2D
from poisson_cpp._core import amr_residual
from poisson_cpp._core import amr_sor
from poisson_cpp._core import extract_arrays
from poisson_cpp._core import gs_smooth
from poisson_cpp._core import harmonic_mean
from poisson_cpp._core import i_of
from poisson_cpp._core import j_of
from poisson_cpp._core import laplacian_fv
from poisson_cpp._core import level_of
from poisson_cpp._core import make_key
from poisson_cpp._core import prolongate_bilinear
from poisson_cpp._core import prolongate_const
from poisson_cpp._core import restrict_avg
from poisson_cpp._core import solve_poisson_1d
from poisson_cpp._core import solve_poisson_1d_dielectric
from poisson_cpp._core import solve_poisson_cg
from poisson_cpp._core import thomas
from poisson_cpp._core import vcycle_amr_composite
from poisson_cpp._core import vcycle_uniform
from poisson_cpp._core import writeback
import typing
from typing import Union
import warnings as warnings
from . import _core
__all__: list = ['AMRArrays', 'AMRSORParams', 'AMRSORReport', 'CGParams', 'CGReport', 'Cell', 'CompositeParams', 'DSTSolver1D', 'DSTSolver2D', 'Direction', 'Grid1D', 'Grid2D', 'Quadtree', 'SORParams', 'SORReport', 'Solver2D', '__version__', 'amr_residual', 'amr_sor', 'dump_amr_snapshot', 'extract_arrays', 'fftw_install_hint', 'gs_smooth', 'harmonic_mean', 'has_fftw3', 'i_of', 'j_of', 'laplacian_fv', 'level_of', 'make_key', 'prolongate_bilinear', 'prolongate_const', 'restrict_avg', 'solve_poisson_1d', 'solve_poisson_1d_dielectric', 'solve_poisson_cg', 'thomas', 'vcycle_amr_composite', 'vcycle_uniform', 'writeback']
def dump_amr_snapshot(tree: '_core.Quadtree', path: str | Path, extra: dict | None = None) -> Path:
    """
    Serialize an AMR :class:`Quadtree` to a JSON file.
    
    The schema matches the one written by the C++ CLI ``poisson_demo
    --problem amr --output ...``: a list of leaves with their geometry
    (``key``, ``level``, ``x``, ``y``, ``h``) and per-leaf scientific
    data (``V``, ``rho``).
    
    Parameters
    ----------
    tree : Quadtree
        Tree whose leaves should be persisted.
    path : str or Path
        Output file. Parent directory must exist.
    extra : dict, optional
        Free-form metadata merged into the top-level JSON object (for
        example ``{"sigma": 0.04}`` to record problem parameters).
    
    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
def fftw_install_hint() -> str:
    """
    Return the FFTW install command for the current platform.
    
    DSTSolver1D/2D need FFTW3 at build time. After installing FFTW with the
    command below, reinstall poisson-cpp from source so CMake picks it up.
    """
__version__: str = '0.1.0'
has_fftw3: bool = True
