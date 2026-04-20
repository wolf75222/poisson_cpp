"""
Python bindings for the poisson_cpp C++20 solver library. Eigen matrices are auto-converted to/from numpy arrays.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AMRArrays', 'AMRSORParams', 'AMRSORReport', 'CGParams', 'CGReport', 'Cell', 'CompositeParams', 'DSTSolver1D', 'DSTSolver2D', 'Direction', 'Grid1D', 'Grid2D', 'Quadtree', 'SORParams', 'SORReport', 'Solver2D', 'amr_residual', 'amr_sor', 'extract_arrays', 'gs_smooth', 'harmonic_mean', 'has_fftw3', 'i_of', 'j_of', 'laplacian_fv', 'level_of', 'make_key', 'prolongate_bilinear', 'prolongate_const', 'restrict_avg', 'solve_poisson_1d', 'solve_poisson_1d_dielectric', 'solve_poisson_cg', 'thomas', 'vcycle_amr_composite', 'vcycle_uniform', 'writeback']
class AMRArrays:
    """
    
    Flat array view of a Quadtree, suitable for fast iterative solves.
    
    Built by :func:`extract_arrays`. ``V`` and ``rho`` are mutable; modify
    them then call :func:`writeback` to push results back into the tree.
    """
    @property
    def V(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Potential per leaf.
        """
    @V.setter
    def V(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    @property
    def Vc(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Diagonal coefficient of the FV stencil per leaf.
        """
    @property
    def h(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Cell size per leaf.
        """
    @property
    def keys(self) -> list[int]:
        """
        Morton key of each leaf, ordered to match V/rho/h.
        """
    @property
    def nb0(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int64], "[m, 4]"]:
        """
        First neighbour index per face (-1 if absent).
        """
    @property
    def nb1(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int64], "[m, 4]"]:
        """
        Second neighbour index per face (-1 if absent).
        """
    @property
    def rho(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Charge density per leaf.
        """
    @rho.setter
    def rho(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    @property
    def w0(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 4]"]:
        """
        Off-diagonal weight for ``nb0`` per face.
        """
    @property
    def w1(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 4]"]:
        """
        Off-diagonal weight for ``nb1`` per face.
        """
class AMRSORParams:
    """
    Tuning parameters for the AMR SOR solver.
    """
    def __init__(self) -> None:
        ...
    @property
    def eps0(self) -> float:
        """
        Permittivity factor in ``rhs = h^2 rho / eps0``.
        """
    @eps0.setter
    def eps0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_iter(self) -> int:
        """
        Iteration cap.
        """
    @max_iter.setter
    def max_iter(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def omega(self) -> float:
        """
        Relaxation factor.
        """
    @omega.setter
    def omega(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        """
        Stopping criterion on max-norm of the per-iteration update.
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class AMRSORReport:
    """
    Result of an AMR SOR run.
    """
    def __repr__(self) -> str:
        ...
    @property
    def iterations(self) -> int:
        """
        Number of sweeps performed.
        """
    @property
    def residual(self) -> float:
        """
        Final ``||V_new - V||_inf``.
        """
class CGParams:
    """
    Tuning parameters for the Conjugate Gradient solver.
    """
    def __init__(self) -> None:
        ...
    @property
    def max_iter(self) -> int:
        """
        Iteration cap.
        """
    @max_iter.setter
    def max_iter(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        """
        Stopping criterion on the relative residual ``||r|| / ||b||``.
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CGReport:
    """
    Result of a CG / PCG run.
    """
    def __repr__(self) -> str:
        ...
    @property
    def iterations(self) -> int:
        """
        Number of CG iterations performed.
        """
    @property
    def residual(self) -> float:
        """
        Final ``||r|| / ||b||``.
        """
class Cell:
    """
    Per-leaf data: potential ``V`` and charge density ``rho``.
    """
    def __init__(self) -> None:
        ...
    @property
    def V(self) -> float:
        """
        Potential at the leaf.
        """
    @V.setter
    def V(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        """
        Charge density at the leaf.
        """
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CompositeParams:
    """
    Tuning parameters for the 2-grid composite V-cycle on AMR.
    """
    def __init__(self) -> None:
        ...
    @property
    def eps0(self) -> float:
        """
        Permittivity factor.
        """
    @eps0.setter
    def eps0(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_coarse_cycles(self) -> int:
        """
        Number of uniform V-cycles on the coarse problem.
        """
    @n_coarse_cycles.setter
    def n_coarse_cycles(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_post(self) -> int:
        """
        Post-smoothing SOR sweeps on AMR.
        """
    @n_post.setter
    def n_post(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_pre(self) -> int:
        """
        Pre-smoothing SOR sweeps on AMR.
        """
    @n_pre.setter
    def n_pre(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def omega(self) -> float:
        """
        SOR omega for AMR smoothing.
        """
    @omega.setter
    def omega(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class DSTSolver1D:
    """
    
    Spectral 1D Poisson solver via DST-I (FFTW), homogeneous Dirichlet BC.
    
    ``O(N log N)``: forward DST, divide by eigenvalues, backward DST.
    The FFTW plan is built once at construction.
    
    Parameters
    ----------
    N : int
        Number of interior nodes.
    L : float
        Domain length.
    eps0 : float, optional
        Permittivity. Default ``1.0``.
    """
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, L: typing.SupportsFloat | typing.SupportsIndex, eps0: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def solve(self, rho: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Solve ``eps0 V'' = -rho`` on the ``N`` interior nodes.
        
        Parameters
        ----------
        rho : numpy.ndarray of shape (N,)
            Charge density at the interior nodes.
        
        Returns
        -------
        numpy.ndarray of shape (N,)
            Potential ``V`` at the interior nodes (boundary values are zero).
        """
class DSTSolver2D:
    """
    
    Spectral 2D Poisson solver via DST-I (FFTW), homogeneous Dirichlet on
    all four faces.
    
    ``O(N^2 log N)`` via tensor product of 1D DSTs.
    
    Parameters
    ----------
    Nx, Ny : int
        Interior grid sizes.
    Lx, Ly : float
        Domain extents.
    eps0 : float, optional
        Permittivity. Default ``1.0``.
    """
    def __init__(self, Nx: typing.SupportsInt | typing.SupportsIndex, Ny: typing.SupportsInt | typing.SupportsIndex, Lx: typing.SupportsFloat | typing.SupportsIndex, Ly: typing.SupportsFloat | typing.SupportsIndex, eps0: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
        ...
    def solve(self, rho: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Solve ``-eps0 Laplacian V = rho`` on the interior grid.
        
        Parameters
        ----------
        rho : numpy.ndarray of shape (Nx, Ny)
            Charge density at the interior nodes.
        
        Returns
        -------
        numpy.ndarray of shape (Nx, Ny)
            Potential ``V`` at the interior nodes (boundary values are zero).
        """
class Direction:
    """
    Geometric direction across a face of a quadtree cell.
    
    Members:
    
      N
    
      S
    
      E
    
      W
    """
    E: typing.ClassVar[Direction]  # value = <Direction.E: 2>
    N: typing.ClassVar[Direction]  # value = <Direction.N: 0>
    S: typing.ClassVar[Direction]  # value = <Direction.S: 1>
    W: typing.ClassVar[Direction]  # value = <Direction.W: 3>
    __members__: typing.ClassVar[dict[str, Direction]]  # value = {'N': <Direction.N: 0>, 'S': <Direction.S: 1>, 'E': <Direction.E: 2>, 'W': <Direction.W: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Grid1D:
    """
    
    1D node-centered grid covering ``[0, L]`` with ``N`` equispaced nodes.
    
    Parameters
    ----------
    L : float
        Domain length.
    N : int
        Number of nodes (boundaries included).
    """
    def __init__(self, L: typing.SupportsFloat | typing.SupportsIndex, N: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def x(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Return the node coordinates as a length-``N`` numpy array.
        """
    @property
    def L(self) -> float:
        """
        Domain length.
        """
    @property
    def N(self) -> int:
        """
        Number of nodes.
        """
    @property
    def dx(self) -> float:
        """
        Spacing ``L / (N - 1)``.
        """
class Grid2D:
    """
    
    2D cell-centered grid on ``[0, Lx] x [0, Ly]`` with ``Nx x Ny`` cells.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain extents in x and y.
    Nx, Ny : int
        Number of cells in each direction. Cell centers sit at
        ``(i + 0.5) * dx``, ``(j + 0.5) * dy``.
    """
    def __init__(self, Lx: typing.SupportsFloat | typing.SupportsIndex, Ly: typing.SupportsFloat | typing.SupportsIndex, Nx: typing.SupportsInt | typing.SupportsIndex, Ny: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def Nx(self) -> int:
        """
        Number of cells along x.
        """
    @property
    def Ny(self) -> int:
        """
        Number of cells along y.
        """
    @property
    def dx(self) -> float:
        """
        Cell width ``Lx/Nx``.
        """
    @property
    def dy(self) -> float:
        """
        Cell height ``Ly/Ny``.
        """
class Quadtree:
    """
    
    Cell-centered quadtree on the square domain ``[0, L] x [0, L]`` with
    Morton-encoded keys. Only leaves are stored.
    
    Parameters
    ----------
    L : float
        Side length of the root domain.
    level_min : int
        Initial uniform refinement level. Builds a ``2^level_min x 2^level_min``
        grid where every cell is a leaf with ``V = rho = 0``.
    """
    def L(self) -> float:
        """
        Domain side length.
        """
    def __init__(self, L: typing.SupportsFloat | typing.SupportsIndex, level_min: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def at(self, key: typing.SupportsInt | typing.SupportsIndex) -> Cell:
        """
        Copy of the leaf's :class:`Cell` data.
        """
    def balance_2to1(self) -> None:
        """
        Enforce the 2:1 balance constraint on the current tree.
        """
    def build(self, predicate: collections.abc.Callable[[typing.SupportsInt | typing.SupportsIndex], bool], level_max: typing.SupportsInt | typing.SupportsIndex, rho_func: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float]) -> None:
        """
        Refine until every leaf either fails ``predicate`` or hits ``level_max``,
        enforce 2:1 balance, then evaluate ``rho_func(x, y)`` at every leaf
        center and store it in ``Cell.rho``.
        
        Parameters
        ----------
        predicate : callable[(int) -> bool]
            Receives the Morton CellKey of a leaf, returns True to refine.
        level_max : int
            Maximum refinement level.
        rho_func : callable[(float, float) -> float]
            Evaluated at every leaf center after refinement.
        """
    def cell_center(self, key: typing.SupportsInt | typing.SupportsIndex) -> tuple[float, float]:
        """
        Geometric center ``(x, y)`` of the given cell key.
        """
    def cell_size(self, level: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Side length of a cell at the given level.
        """
    def is_leaf(self, key: typing.SupportsInt | typing.SupportsIndex) -> bool:
        """
        Return True if the cell key is currently a leaf.
        """
    def leaves(self) -> dict[int, Cell]:
        """
        Return ``{key: Cell}`` for all leaves (copy).
        """
    def level_min(self) -> int:
        """
        Initial uniform refinement level.
        """
    def neighbour_leaves(self, key: typing.SupportsInt | typing.SupportsIndex, dir: Direction) -> list[int]:
        """
        Return up to 2 leaf neighbours of ``key`` across face ``dir``.
        
        Returns
        -------
        list[int]
            Empty if ``key`` is at the boundary, one key for same-level or coarser
            neighbours, two keys for finer neighbours.
        """
    def num_leaves(self) -> int:
        """
        Total number of leaves in the tree.
        """
    def refine(self, key: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Subdivide the leaf ``key`` into its 4 children. ``key`` must currently be a leaf.
        """
class SORParams:
    """
    Tuning parameters for the SOR solver.
    """
    def __init__(self) -> None:
        ...
    @property
    def max_iter(self) -> int:
        """
        Iteration cap.
        """
    @max_iter.setter
    def max_iter(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def omega(self) -> float:
        """
        Relaxation factor. ``-1`` means auto (``omega_opt`` from grid size).
        """
    @omega.setter
    def omega(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        """
        Stopping criterion on the max-norm of the residual.
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SORReport:
    """
    Result of an SOR run.
    """
    def __repr__(self) -> str:
        ...
    @property
    def iterations(self) -> int:
        """
        Number of sweeps performed.
        """
    @property
    def residual(self) -> float:
        """
        Final ``||V_new - V_old||_inf``.
        """
class Solver2D:
    """
    
    2D finite-volume Poisson solver using SOR red-black with optimal omega.
    
    Boundary conditions: Dirichlet in x (``uL`` at left, ``uR`` at right),
    homogeneous Neumann in y. Permittivity may be constant or spatial.
    
    Parameters
    ----------
    grid : Grid2D
        Cell-centered grid.
    eps : float or numpy.ndarray
        Constant permittivity, or per-cell array of shape ``(Nx, Ny)``.
    uL, uR : float
        Dirichlet values on the x-boundaries.
    """
    @typing.overload
    def __init__(self, grid: Grid2D, eps: typing.SupportsFloat | typing.SupportsIndex, uL: typing.SupportsFloat | typing.SupportsIndex, uR: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Construct with a constant permittivity ``eps``.
        """
    @typing.overload
    def __init__(self, grid: Grid2D, eps: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], uL: typing.SupportsFloat | typing.SupportsIndex, uR: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Construct with a spatially-varying permittivity ``eps[i,j]``.
        """
    def solve(self, rho: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], omega: typing.SupportsFloat | typing.SupportsIndex = -1.0, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_iter: typing.SupportsInt | typing.SupportsIndex = 20000) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], SORReport]:
        """
        Run SOR from a zero initial guess.
        
        Parameters
        ----------
        rho : numpy.ndarray of shape (Nx, Ny)
            Charge density per cell.
        omega : float, optional
            Relaxation factor. Default ``-1`` selects
            ``omega_opt = 2 / (1 + sin(pi/N))``.
        tol : float, optional
            Convergence threshold on max-norm residual. Default ``1e-8``.
        max_iter : int, optional
            Iteration cap. Default ``20_000``.
        
        Returns
        -------
        V : numpy.ndarray of shape (Nx, Ny)
            Potential field.
        report : SORReport
            Iteration count and final residual.
        """
    def solve_inplace(self, V: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], rho: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], omega: typing.SupportsFloat | typing.SupportsIndex = -1.0, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_iter: typing.SupportsInt | typing.SupportsIndex = 20000) -> SORReport:
        """
        In-place variant of :meth:`solve`.
        
        ``V`` and ``rho`` must be Fortran-ordered numpy arrays (``order='F'``)
        matching Eigen's column-major layout, otherwise pybind11 will copy them.
        
        Parameters
        ----------
        V : numpy.ndarray of shape (Nx, Ny), order='F'
            Initial guess; overwritten with the solution.
        rho : numpy.ndarray of shape (Nx, Ny), order='F'
            Charge density per cell.
        omega, tol, max_iter
            Same meaning as :meth:`solve`.
        
        Returns
        -------
        SORReport
            Iteration count and final residual.
        """
def amr_residual(arr: AMRArrays, eps0: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    FV residual at every leaf:
    ``r_i = sum w * V_neigh + h_i^2 rho_i / eps0 - Vc_i V_i``.
    
    Parameters
    ----------
    arr : AMRArrays
    eps0 : float, optional
        Permittivity factor. Default ``1.0``.
    
    Returns
    -------
    numpy.ndarray of shape (n_leaves,)
    """
def amr_sor(arr: AMRArrays, omega: typing.SupportsFloat | typing.SupportsIndex = 1.85, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_iter: typing.SupportsInt | typing.SupportsIndex = 20000, eps0: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> AMRSORReport:
    """
    Run Gauss-Seidel-like SOR on the AMR arrays, in place.
    
    Parameters
    ----------
    arr : AMRArrays
        Flat view from :func:`extract_arrays`. ``arr.V`` is updated in place.
    omega : float, optional
        Relaxation factor. Default ``1.85``.
    tol : float, optional
        Convergence threshold. Default ``1e-8``.
    max_iter : int, optional
        Iteration cap. Default ``20_000``.
    eps0 : float, optional
        Permittivity factor. Default ``1.0``.
    
    Returns
    -------
    AMRSORReport
    """
def extract_arrays(tree: Quadtree) -> AMRArrays:
    """
    Build an :class:`AMRArrays` view from a fully built :class:`Quadtree`.
    
    The returned view stores precomputed FV stencil weights handling 2:1
    coarse/fine interfaces and Dirichlet (V=0) ghost cells at the boundary.
    
    Parameters
    ----------
    tree : Quadtree
        Source tree. Must already have ``rho`` populated (e.g. via ``build``).
    
    Returns
    -------
    AMRArrays
    """
def gs_smooth(V: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], rho: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], h: typing.SupportsFloat | typing.SupportsIndex, n_iter: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Gauss-Seidel red-black smoothing sweeps on the uniform cell-centered FV
    Poisson operator with Dirichlet ``V = 0`` on the full boundary.
    
    Parameters
    ----------
    V : numpy.ndarray of shape (N, N), order='F'
        Initial guess; updated in place after ``n_iter`` sweeps.
    rho : numpy.ndarray of shape (N, N), order='F'
        Right-hand side.
    h : float
        Cell size.
    n_iter : int
        Number of sweeps.
    """
def harmonic_mean(a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Element-wise harmonic mean ``2 a b / (a + b)``.
    
    Used to compute the effective permittivity at the face between two cells
    with different ``eps_r`` (preserves continuity of the normal component
    of ``D = eps0 eps_r grad V``).
    
    Parameters
    ----------
    a, b : numpy.ndarray of shape (N,)
        Per-element values, must be strictly positive.
    
    Returns
    -------
    numpy.ndarray of shape (N,)
    """
def i_of(key: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    x-coordinate (cell index at the cell's level) encoded in the key.
    """
def j_of(key: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    y-coordinate (cell index at the cell's level) encoded in the key.
    """
def laplacian_fv(V: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], h: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Apply the uniform FV Laplacian: returns ``A V`` where ``A`` is the
    5-point stencil with Dirichlet ``V = 0`` on the boundary.
    
    Parameters
    ----------
    V : numpy.ndarray of shape (N, N)
    h : float
        Cell size.
    
    Returns
    -------
    numpy.ndarray of shape (N, N)
    """
def level_of(key: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Refinement level encoded in the CellKey.
    """
def make_key(level: typing.SupportsInt | typing.SupportsIndex, i: typing.SupportsInt | typing.SupportsIndex, j: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Encode ``(level, i, j)`` into a Morton CellKey.
    """
def prolongate_bilinear(c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Bilinear prolongation (order 2): ``(M, M) -> (2M, 2M)`` for cell-centered
    FV. Each fine cell gets a weighted combination of the enclosing coarse
    cell and its two neighbours along the fine cell's offset.
    
    Parameters
    ----------
    c : numpy.ndarray of shape (M, M)
    
    Returns
    -------
    numpy.ndarray of shape (2M, 2M)
    """
def prolongate_const(c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Piecewise-constant prolongation (order 0): ``(N/2, N/2) -> (N, N)``.
    
    Parameters
    ----------
    c : numpy.ndarray of shape (M, M)
    
    Returns
    -------
    numpy.ndarray of shape (2M, 2M)
    """
def restrict_avg(r: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    4-cell averaging restriction: ``(N, N) -> (N/2, N/2)``. ``N`` must be even.
    
    Parameters
    ----------
    r : numpy.ndarray of shape (N, N)
    
    Returns
    -------
    numpy.ndarray of shape (N/2, N/2)
    """
def solve_poisson_1d(rho: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], uL: typing.SupportsFloat | typing.SupportsIndex, uR: typing.SupportsFloat | typing.SupportsIndex, grid: Grid1D) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Finite-volume 1D Poisson with Dirichlet BC, uniform permittivity.
    
    Solves ``-V''(x) = rho(x) / eps0`` on a node-centered grid with
    ``V(0) = uL`` and ``V(L) = uR``.
    
    Parameters
    ----------
    rho : numpy.ndarray of shape (N,)
        Charge density at each node.
    uL, uR : float
        Dirichlet values at ``x = 0`` and ``x = L``.
    grid : Grid1D
        Discretization grid.
    
    Returns
    -------
    numpy.ndarray of shape (N,)
        Potential ``V`` at each node.
    """
def solve_poisson_1d_dielectric(rho: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], eps_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], uL: typing.SupportsFloat | typing.SupportsIndex, uR: typing.SupportsFloat | typing.SupportsIndex, grid: Grid1D, eps0: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Finite-volume 1D Poisson with spatially-varying permittivity.
    
    Solves ``eps0 * d/dx[eps_r(x) dV/dx] = -rho`` on a node-centered grid
    with ``V(0) = uL`` and ``V(L) = uR``. The face permittivity is the
    harmonic mean of the two adjacent cell values, which preserves the
    normal component of ``D = eps0 eps_r grad V`` across a dielectric
    interface.
    
    Parameters
    ----------
    rho : numpy.ndarray of shape (N,)
        Charge density at each node.
    eps_r : numpy.ndarray of shape (N,)
        Relative permittivity at each node. Must be strictly positive.
    uL, uR : float
        Dirichlet values at ``x = 0`` and ``x = L``.
    grid : Grid1D
        Discretization grid.
    eps0 : float, optional
        Vacuum permittivity. Default ``1.0``.
    
    Returns
    -------
    numpy.ndarray of shape (N,)
        Potential ``V`` at each node.
    """
def solve_poisson_cg(V: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], rho: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], grid: Grid2D, eps: typing.SupportsFloat | typing.SupportsIndex = 1.0, uL: typing.SupportsFloat | typing.SupportsIndex = 0.0, uR: typing.SupportsFloat | typing.SupportsIndex = 0.0, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-08, max_iter: typing.SupportsInt | typing.SupportsIndex = 10000, use_preconditioner: bool = False, record_history: bool = False) -> tuple[CGReport, list[float]]:
    """
    Solve ``-eps Laplacian V = rho`` by Conjugate Gradient.
    
    Cell-centered FV discretization with Dirichlet in x (``uL`` left,
    ``uR`` right) and homogeneous Neumann in y. Boundary values are folded
    into the right-hand side so the operator stays SPD.
    
    Parameters
    ----------
    V : numpy.ndarray of shape (Nx, Ny), order='F'
        Initial guess; updated in place with the solution.
    rho : numpy.ndarray of shape (Nx, Ny), order='F'
        Charge density per cell.
    grid : Grid2D
        Cell-centered grid.
    eps : float, optional
        Constant permittivity. Default ``1.0``.
    uL, uR : float, optional
        Dirichlet values on the x-boundaries. Default ``0.0``.
    tol : float, optional
        Stopping criterion on relative residual. Default ``1e-8``.
    max_iter : int, optional
        Iteration cap. Default ``10_000``.
    use_preconditioner : bool, optional
        If True, use Jacobi PCG. Marginal gain unless ``eps`` varies
        spatially. Default False.
    record_history : bool, optional
        If True, record ``||r|| / ||b||`` at every iteration. Default False.
    
    Returns
    -------
    report : CGReport
        Iteration count and final relative residual.
    history : list of float
        Per-iteration relative residuals when ``record_history=True``,
        otherwise an empty list.
    """
def thomas(a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], c: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], d: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Solve a tridiagonal linear system ``A x = d`` by the Thomas algorithm.
    
    The matrix ``A`` has ``a`` on the sub-diagonal, ``b`` on the main diagonal,
    and ``c`` on the super-diagonal. Runs in ``O(N)``.
    
    Parameters
    ----------
    a : numpy.ndarray of shape (N,)
        Sub-diagonal; ``a[0]`` is unused.
    b : numpy.ndarray of shape (N,)
        Main diagonal.
    c : numpy.ndarray of shape (N,)
        Super-diagonal; ``c[N-1]`` is unused.
    d : numpy.ndarray of shape (N,)
        Right-hand side.
    
    Returns
    -------
    numpy.ndarray of shape (N,)
        Solution vector ``x``.
    
    Raises
    ------
    RuntimeError
        If a zero pivot is encountered (matrix not diagonally dominant).
    """
def vcycle_amr_composite(arr: AMRArrays, tree: Quadtree, params: CompositeParams = ...) -> None:
    """
    One composite 2-grid V-cycle on an AMR tree.
    
    Steps:
      1. SOR pre-smoothing on AMR leaves.
      2. AMR residual ``r``.
      3. Volume-weighted restriction ``r -> r_c`` on a uniform
         ``2^level_min`` grid.
      4. ``n_coarse_cycles`` uniform V-cycles to approximately solve
         ``A delta = r_c``.
      5. Bilinear prolongation of ``delta`` back to AMR leaves
         (``V += delta``).
      6. SOR post-smoothing on AMR leaves.
    
    The AMR array state is updated in place.
    
    Parameters
    ----------
    arr : AMRArrays
        Flat view, updated in place.
    tree : Quadtree
        The tree from which ``arr`` was extracted.
    params : CompositeParams, optional
        Tuning parameters; defaults match the C++ defaults.
    """
def vcycle_uniform(V: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], rho: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], h: typing.SupportsFloat | typing.SupportsIndex, n_pre: typing.SupportsInt | typing.SupportsIndex = 2, n_post: typing.SupportsInt | typing.SupportsIndex = 2, n_min: typing.SupportsInt | typing.SupportsIndex = 4) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Recursive V-cycle multigrid on a uniform grid with Dirichlet ``V = 0``
    on the boundary. Coarsens by factor 2 down to size ``<= n_min``.
    
    Parameters
    ----------
    V : numpy.ndarray of shape (N, N)
        Initial guess.
    rho : numpy.ndarray of shape (N, N)
        Right-hand side.
    h : float
        Cell size at the finest level.
    n_pre, n_post : int, optional
        Pre/post Gauss-Seidel sweeps at each level. Default ``2``.
    n_min : int, optional
        Coarsest grid size at which to stop recursing. Default ``4``.
    
    Returns
    -------
    numpy.ndarray of shape (N, N)
        Updated potential.
    """
def writeback(tree: Quadtree, keys: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], V: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> None:
    """
    Push the flat ``V`` array back into the tree leaves.
    
    Parameters
    ----------
    tree : Quadtree
        Tree to update in place.
    keys : list[int]
        Leaf keys, must match the order of ``V`` (use ``arr.keys``).
    V : numpy.ndarray of shape (n_leaves,)
    """
__version__: str = '0.1.0'
has_fftw3: bool = True
