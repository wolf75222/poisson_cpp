// pybind11 bindings for poisson_cpp. Built only when
// `-DPOISSON_BUILD_PYTHON=ON` is passed to CMake. Compatible with C++20.
//
// Build (from the repo root):
//   cmake -B build -DPOISSON_BUILD_PYTHON=ON
//   cmake --build build --target poisson_py -j
//
// After building, `build/python/_core*.so` is the importable module
// (re-exported by the python/poisson_cpp/__init__.py shim).

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "poisson/amr/morton.hpp"
#include "poisson/amr/quadtree.hpp"
#include "poisson/amr/solver.hpp"
#include "poisson/core/grid.hpp"
#include "poisson/fv/dielectric.hpp"
#include "poisson/fv/solver1d.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/iter/poisson_cg.hpp"
#include "poisson/linalg/thomas.hpp"
#include "poisson/mg/vcycle.hpp"

#if defined(POISSON_HAVE_FFTW3)
#include "poisson/spectral/dst1d.hpp"
#include "poisson/spectral/dst2d.hpp"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for the poisson_cpp C++20 solver library. "
            "Eigen matrices are auto-converted to/from numpy arrays.";

  // --- core ----------------------------------------------------------------
  py::class_<poisson::Grid1D>(m, "Grid1D", R"doc(
1D node-centered grid covering ``[0, L]`` with ``N`` equispaced nodes.

Parameters
----------
L : float
    Domain length.
N : int
    Number of nodes (boundaries included).
)doc")
      .def(py::init<double, int>(), py::arg("L"), py::arg("N"))
      .def_property_readonly("N",  [](const poisson::Grid1D& g){ return g.N; },
                             "Number of nodes.")
      .def_property_readonly("L",  [](const poisson::Grid1D& g){ return g.L; },
                             "Domain length.")
      .def_property_readonly("dx", &poisson::Grid1D::dx,
                             "Spacing ``L / (N - 1)``.")
      .def("x", &poisson::Grid1D::x,
           "Return the node coordinates as a length-``N`` numpy array.")
      .def("__repr__", [](const poisson::Grid1D& g) {
        return "Grid1D(L=" + std::to_string(g.L) +
               ", N=" + std::to_string(g.N) + ")";
      });

  py::class_<poisson::Grid2D>(m, "Grid2D", R"doc(
2D cell-centered grid on ``[0, Lx] x [0, Ly]`` with ``Nx x Ny`` cells.

Parameters
----------
Lx, Ly : float
    Domain extents in x and y.
Nx, Ny : int
    Number of cells in each direction. Cell centers sit at
    ``(i + 0.5) * dx``, ``(j + 0.5) * dy``.
)doc")
      .def(py::init<double, double, int, int>(),
           py::arg("Lx"), py::arg("Ly"), py::arg("Nx"), py::arg("Ny"))
      .def_readonly("Nx", &poisson::Grid2D::Nx, "Number of cells along x.")
      .def_readonly("Ny", &poisson::Grid2D::Ny, "Number of cells along y.")
      .def_property_readonly("dx", &poisson::Grid2D::dx, "Cell width ``Lx/Nx``.")
      .def_property_readonly("dy", &poisson::Grid2D::dy, "Cell height ``Ly/Ny``.");

  // --- linalg --------------------------------------------------------------
  m.def("thomas", &poisson::linalg::thomas,
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
        R"doc(
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
)doc");

  // --- FV solvers ----------------------------------------------------------
  m.def("solve_poisson_1d",
        [](const Eigen::VectorXd& rho, double uL, double uR,
           const poisson::Grid1D& grid) {
          return poisson::fv::solve_poisson_1d(rho, uL, uR, grid);
        },
        py::arg("rho"), py::arg("uL"), py::arg("uR"), py::arg("grid"),
        R"doc(
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
)doc");

  m.def("solve_poisson_1d_dielectric",
        [](const Eigen::VectorXd& rho, const Eigen::VectorXd& eps_r,
           double uL, double uR, const poisson::Grid1D& grid, double eps0) {
          return poisson::fv::solve_poisson_1d(rho, eps_r, uL, uR, grid, eps0);
        },
        py::arg("rho"), py::arg("eps_r"), py::arg("uL"), py::arg("uR"),
        py::arg("grid"), py::arg("eps0") = 1.0,
        R"doc(
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
)doc");

  py::class_<poisson::fv::SORParams>(m, "SORParams",
                                     "Tuning parameters for the SOR solver.")
      .def(py::init<>())
      .def_readwrite("omega",    &poisson::fv::SORParams::omega,
                     "Relaxation factor. ``-1`` means auto "
                     "(``omega_opt`` from grid size).")
      .def_readwrite("tol",      &poisson::fv::SORParams::tol,
                     "Stopping criterion on the max-norm of the residual.")
      .def_readwrite("max_iter", &poisson::fv::SORParams::max_iter,
                     "Iteration cap.");

  py::class_<poisson::fv::SORReport>(m, "SORReport",
                                     "Result of an SOR run.")
      .def_readonly("iterations", &poisson::fv::SORReport::iterations,
                    "Number of sweeps performed.")
      .def_readonly("residual",   &poisson::fv::SORReport::residual,
                    "Final ``||V_new - V_old||_inf``.")
      .def("__repr__", [](const poisson::fv::SORReport& r) {
        return "SORReport(iterations=" + std::to_string(r.iterations) +
               ", residual=" + std::to_string(r.residual) + ")";
      });

  py::class_<poisson::fv::Solver2D>(m, "Solver2D", R"doc(
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
)doc")
      .def(py::init<const poisson::Grid2D&, double, double, double>(),
           py::arg("grid"), py::arg("eps"), py::arg("uL"), py::arg("uR"),
           "Construct with a constant permittivity ``eps``.")
      .def(py::init<const poisson::Grid2D&, Eigen::MatrixXd, double, double>(),
           py::arg("grid"), py::arg("eps"), py::arg("uL"), py::arg("uR"),
           "Construct with a spatially-varying permittivity ``eps[i,j]``.")
      .def("solve",
           [](const poisson::fv::Solver2D& s,
              const Eigen::MatrixXd& rho,
              double omega, double tol, int max_iter) {
             Eigen::MatrixXd V = Eigen::MatrixXd::Zero(rho.rows(), rho.cols());
             const auto r = s.solve(V, rho, {.omega = omega, .tol = tol,
                                              .max_iter = max_iter});
             return py::make_tuple(V, r);
           },
           py::arg("rho"),
           py::arg("omega") = -1.0, py::arg("tol") = 1e-8,
           py::arg("max_iter") = 20'000,
           R"doc(
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
)doc")
      .def("solve_inplace",
           [](const poisson::fv::Solver2D& s, Eigen::Ref<Eigen::MatrixXd> V,
              Eigen::Ref<const Eigen::MatrixXd> rho,
              double omega, double tol, int max_iter) {
             return s.solve(V, rho, {.omega = omega, .tol = tol,
                                      .max_iter = max_iter});
           },
           py::arg("V"), py::arg("rho"),
           py::arg("omega") = -1.0, py::arg("tol") = 1e-8,
           py::arg("max_iter") = 20'000,
           R"doc(
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
)doc");

  // --- CG / PCG iterative solvers -----------------------------------------
  py::class_<poisson::iter::CGParams>(m, "CGParams",
                                      "Tuning parameters for the Conjugate "
                                      "Gradient solver.")
      .def(py::init<>())
      .def_readwrite("tol",      &poisson::iter::CGParams::tol,
                     "Stopping criterion on the relative residual "
                     "``||r|| / ||b||``.")
      .def_readwrite("max_iter", &poisson::iter::CGParams::max_iter,
                     "Iteration cap.");

  py::class_<poisson::iter::CGReport>(m, "CGReport",
                                      "Result of a CG / PCG run.")
      .def_readonly("iterations", &poisson::iter::CGReport::iterations,
                    "Number of CG iterations performed.")
      .def_readonly("residual",   &poisson::iter::CGReport::residual,
                    "Final ``||r|| / ||b||``.")
      .def("__repr__", [](const poisson::iter::CGReport& r) {
        return "CGReport(iterations=" + std::to_string(r.iterations) +
               ", residual=" + std::to_string(r.residual) + ")";
      });

  m.def("solve_poisson_cg",
        [](Eigen::Ref<Eigen::MatrixXd> V,
           Eigen::Ref<const Eigen::MatrixXd> rho,
           const poisson::Grid2D& grid,
           double eps, double uL, double uR,
           double tol, int max_iter,
           bool use_preconditioner, bool record_history) {
          std::vector<double> hist;
          const auto rep = poisson::iter::solve_poisson_cg(
              V, rho, grid, eps, uL, uR,
              {.tol = tol, .max_iter = max_iter},
              use_preconditioner,
              record_history ? &hist : nullptr);
          return py::make_tuple(rep, hist);
        },
        py::arg("V"), py::arg("rho"), py::arg("grid"),
        py::arg("eps") = 1.0, py::arg("uL") = 0.0, py::arg("uR") = 0.0,
        py::arg("tol") = 1e-8, py::arg("max_iter") = 10'000,
        py::arg("use_preconditioner") = false,
        py::arg("record_history") = false,
        R"doc(
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
)doc");

#if defined(POISSON_HAVE_FFTW3)
  py::class_<poisson::spectral::DSTSolver1D>(m, "DSTSolver1D", R"doc(
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
)doc")
      .def(py::init<int, double, double>(),
           py::arg("N"), py::arg("L"), py::arg("eps0") = 1.0)
      .def("solve", &poisson::spectral::DSTSolver1D::solve, py::arg("rho"),
           R"doc(
Solve ``eps0 V'' = -rho`` on the ``N`` interior nodes.

Parameters
----------
rho : numpy.ndarray of shape (N,)
    Charge density at the interior nodes.

Returns
-------
numpy.ndarray of shape (N,)
    Potential ``V`` at the interior nodes (boundary values are zero).
)doc");

  py::class_<poisson::spectral::DSTSolver2D>(m, "DSTSolver2D", R"doc(
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
)doc")
      .def(py::init<int, int, double, double, double>(),
           py::arg("Nx"), py::arg("Ny"), py::arg("Lx"), py::arg("Ly"),
           py::arg("eps0") = 1.0)
      .def("solve", &poisson::spectral::DSTSolver2D::solve, py::arg("rho"),
           R"doc(
Solve ``-eps0 Laplacian V = rho`` on the interior grid.

Parameters
----------
rho : numpy.ndarray of shape (Nx, Ny)
    Charge density at the interior nodes.

Returns
-------
numpy.ndarray of shape (Nx, Ny)
    Potential ``V`` at the interior nodes (boundary values are zero).
)doc");
#endif

  // ========================================================================
  // FV utility
  // ========================================================================

  m.def("harmonic_mean", &poisson::fv::harmonic_mean,
        py::arg("a"), py::arg("b"),
        R"doc(
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
)doc");

  // ========================================================================
  // AMR (quadtree)
  // ========================================================================

  py::enum_<poisson::amr::Direction>(m, "Direction",
        "Geometric direction across a face of a quadtree cell.")
      .value("N", poisson::amr::Direction::N)
      .value("S", poisson::amr::Direction::S)
      .value("E", poisson::amr::Direction::E)
      .value("W", poisson::amr::Direction::W);

  py::class_<poisson::amr::Cell>(m, "Cell",
        "Per-leaf data: potential ``V`` and charge density ``rho``.")
      .def(py::init<>())
      .def_readwrite("V",   &poisson::amr::Cell::V,
                     "Potential at the leaf.")
      .def_readwrite("rho", &poisson::amr::Cell::rho,
                     "Charge density at the leaf.");

  py::class_<poisson::amr::Quadtree>(m, "Quadtree", R"doc(
Cell-centered quadtree on the square domain ``[0, L] x [0, L]`` with
Morton-encoded keys. Only leaves are stored.

Parameters
----------
L : float
    Side length of the root domain.
level_min : int
    Initial uniform refinement level. Builds a ``2^level_min x 2^level_min``
    grid where every cell is a leaf with ``V = rho = 0``.
)doc")
      .def(py::init<double, int>(), py::arg("L"), py::arg("level_min"))
      .def("cell_size", &poisson::amr::Quadtree::cell_size, py::arg("level"),
           "Side length of a cell at the given level.")
      .def("cell_center", &poisson::amr::Quadtree::cell_center, py::arg("key"),
           "Geometric center ``(x, y)`` of the given cell key.")
      .def("refine", &poisson::amr::Quadtree::refine, py::arg("key"),
           "Subdivide the leaf ``key`` into its 4 children. "
           "``key`` must currently be a leaf.")
      .def("build",
           [](poisson::amr::Quadtree& self,
              const std::function<bool(poisson::amr::CellKey)>& predicate,
              uint8_t level_max,
              const std::function<double(double, double)>& rho_func) {
             self.build(predicate, level_max, rho_func);
           },
           py::arg("predicate"), py::arg("level_max"), py::arg("rho_func"),
           R"doc(
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
)doc")
      .def("balance_2to1", &poisson::amr::Quadtree::balance_2to1,
           "Enforce the 2:1 balance constraint on the current tree.")
      .def("neighbour_leaves",
           &poisson::amr::Quadtree::neighbour_leaves,
           py::arg("key"), py::arg("dir"),
           R"doc(
Return up to 2 leaf neighbours of ``key`` across face ``dir``.

Returns
-------
list[int]
    Empty if ``key`` is at the boundary, one key for same-level or coarser
    neighbours, two keys for finer neighbours.
)doc")
      .def("is_leaf", &poisson::amr::Quadtree::is_leaf, py::arg("key"),
           "Return True if the cell key is currently a leaf.")
      .def("num_leaves", &poisson::amr::Quadtree::num_leaves,
           "Total number of leaves in the tree.")
      .def("L", &poisson::amr::Quadtree::L,
           "Domain side length.")
      .def("level_min", &poisson::amr::Quadtree::level_min,
           "Initial uniform refinement level.")
      .def("at",
           [](const poisson::amr::Quadtree& self, poisson::amr::CellKey k) {
             return self.at(k);
           },
           py::arg("key"), py::return_value_policy::copy,
           "Copy of the leaf's :class:`Cell` data.")
      .def("leaves",
           [](const poisson::amr::Quadtree& self) {
             return self.leaves();
           },
           "Return ``{key: Cell}`` for all leaves (copy).");

  m.def("make_key", &poisson::amr::make_key,
        py::arg("level"), py::arg("i"), py::arg("j"),
        "Encode ``(level, i, j)`` into a Morton CellKey.");
  m.def("level_of", &poisson::amr::level_of, py::arg("key"),
        "Refinement level encoded in the CellKey.");
  m.def("i_of", &poisson::amr::i_of, py::arg("key"),
        "x-coordinate (cell index at the cell's level) encoded in the key.");
  m.def("j_of", &poisson::amr::j_of, py::arg("key"),
        "y-coordinate (cell index at the cell's level) encoded in the key.");

  py::class_<poisson::amr::AMRArrays>(m, "AMRArrays", R"doc(
Flat array view of a Quadtree, suitable for fast iterative solves.

Built by :func:`extract_arrays`. ``V`` and ``rho`` are mutable; modify
them then call :func:`writeback` to push results back into the tree.
)doc")
      .def_readonly("keys", &poisson::amr::AMRArrays::keys,
                    "Morton key of each leaf, ordered to match V/rho/h.")
      .def_readwrite("V",   &poisson::amr::AMRArrays::V,
                     "Potential per leaf.")
      .def_readwrite("rho", &poisson::amr::AMRArrays::rho,
                     "Charge density per leaf.")
      .def_readonly("h",   &poisson::amr::AMRArrays::h,
                    "Cell size per leaf.")
      .def_readonly("Vc",  &poisson::amr::AMRArrays::Vc,
                    "Diagonal coefficient of the FV stencil per leaf.")
      .def_readonly("nb0", &poisson::amr::AMRArrays::nb0,
                    "First neighbour index per face (-1 if absent).")
      .def_readonly("nb1", &poisson::amr::AMRArrays::nb1,
                    "Second neighbour index per face (-1 if absent).")
      .def_readonly("w0",  &poisson::amr::AMRArrays::w0,
                    "Off-diagonal weight for ``nb0`` per face.")
      .def_readonly("w1",  &poisson::amr::AMRArrays::w1,
                    "Off-diagonal weight for ``nb1`` per face.");

  m.def("extract_arrays", &poisson::amr::extract_arrays, py::arg("tree"),
        R"doc(
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
)doc");

  m.def("writeback", &poisson::amr::writeback,
        py::arg("tree"), py::arg("keys"), py::arg("V"),
        R"doc(
Push the flat ``V`` array back into the tree leaves.

Parameters
----------
tree : Quadtree
    Tree to update in place.
keys : list[int]
    Leaf keys, must match the order of ``V`` (use ``arr.keys``).
V : numpy.ndarray of shape (n_leaves,)
)doc");

  py::class_<poisson::amr::SORParams>(m, "AMRSORParams",
        "Tuning parameters for the AMR SOR solver.")
      .def(py::init<>())
      .def_readwrite("omega",    &poisson::amr::SORParams::omega,
                     "Relaxation factor.")
      .def_readwrite("tol",      &poisson::amr::SORParams::tol,
                     "Stopping criterion on max-norm of the per-iteration "
                     "update.")
      .def_readwrite("max_iter", &poisson::amr::SORParams::max_iter,
                     "Iteration cap.")
      .def_readwrite("eps0",     &poisson::amr::SORParams::eps0,
                     "Permittivity factor in ``rhs = h^2 rho / eps0``.");

  py::class_<poisson::amr::SORReport>(m, "AMRSORReport",
        "Result of an AMR SOR run.")
      .def_readonly("iterations", &poisson::amr::SORReport::iterations,
                    "Number of sweeps performed.")
      .def_readonly("residual",   &poisson::amr::SORReport::residual,
                    "Final ``||V_new - V||_inf``.")
      .def("__repr__", [](const poisson::amr::SORReport& r) {
        return "AMRSORReport(iterations=" + std::to_string(r.iterations) +
               ", residual=" + std::to_string(r.residual) + ")";
      });

  m.def("amr_sor",
        [](poisson::amr::AMRArrays& a, double omega, double tol,
           int max_iter, double eps0) {
          return poisson::amr::sor(a, {.omega = omega, .tol = tol,
                                       .max_iter = max_iter, .eps0 = eps0});
        },
        py::arg("arr"), py::arg("omega") = 1.85, py::arg("tol") = 1e-8,
        py::arg("max_iter") = 20'000, py::arg("eps0") = 1.0,
        R"doc(
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
)doc");

  m.def("amr_residual", &poisson::amr::residual,
        py::arg("arr"), py::arg("eps0") = 1.0,
        R"doc(
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
)doc");

  // ========================================================================
  // Multigrille
  // ========================================================================

  m.def("gs_smooth",
        [](Eigen::Ref<Eigen::MatrixXd> V,
           Eigen::Ref<const Eigen::MatrixXd> rho,
           double h, int n_iter) {
          poisson::mg::gs_smooth(V, rho, h, n_iter);
        },
        py::arg("V"), py::arg("rho"), py::arg("h"), py::arg("n_iter"),
        R"doc(
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
)doc");

  m.def("laplacian_fv", &poisson::mg::laplacian_fv,
        py::arg("V"), py::arg("h"),
        R"doc(
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
)doc");

  m.def("restrict_avg", &poisson::mg::restrict_avg, py::arg("r"),
        R"doc(
4-cell averaging restriction: ``(N, N) -> (N/2, N/2)``. ``N`` must be even.

Parameters
----------
r : numpy.ndarray of shape (N, N)

Returns
-------
numpy.ndarray of shape (N/2, N/2)
)doc");

  m.def("prolongate_const", &poisson::mg::prolongate_const, py::arg("c"),
        R"doc(
Piecewise-constant prolongation (order 0): ``(N/2, N/2) -> (N, N)``.

Parameters
----------
c : numpy.ndarray of shape (M, M)

Returns
-------
numpy.ndarray of shape (2M, 2M)
)doc");

  m.def("prolongate_bilinear", &poisson::mg::prolongate_bilinear,
        py::arg("c"),
        R"doc(
Bilinear prolongation (order 2): ``(M, M) -> (2M, 2M)`` for cell-centered
FV. Each fine cell gets a weighted combination of the enclosing coarse
cell and its two neighbours along the fine cell's offset.

Parameters
----------
c : numpy.ndarray of shape (M, M)

Returns
-------
numpy.ndarray of shape (2M, 2M)
)doc");

  m.def("vcycle_uniform", &poisson::mg::vcycle_uniform,
        py::arg("V"), py::arg("rho"), py::arg("h"),
        py::arg("n_pre") = 2, py::arg("n_post") = 2, py::arg("n_min") = 4,
        R"doc(
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
)doc");

  py::class_<poisson::mg::CompositeParams>(m, "CompositeParams",
        "Tuning parameters for the 2-grid composite V-cycle on AMR.")
      .def(py::init<>())
      .def_readwrite("n_pre",
                     &poisson::mg::CompositeParams::n_pre,
                     "Pre-smoothing SOR sweeps on AMR.")
      .def_readwrite("n_post",
                     &poisson::mg::CompositeParams::n_post,
                     "Post-smoothing SOR sweeps on AMR.")
      .def_readwrite("n_coarse_cycles",
                     &poisson::mg::CompositeParams::n_coarse_cycles,
                     "Number of uniform V-cycles on the coarse problem.")
      .def_readwrite("omega",
                     &poisson::mg::CompositeParams::omega,
                     "SOR omega for AMR smoothing.")
      .def_readwrite("eps0",
                     &poisson::mg::CompositeParams::eps0,
                     "Permittivity factor.");

  m.def("vcycle_amr_composite",
        [](poisson::amr::AMRArrays& a, const poisson::amr::Quadtree& tree,
           const poisson::mg::CompositeParams& p) {
          poisson::mg::vcycle_amr_composite(a, tree, p);
        },
        py::arg("arr"), py::arg("tree"),
        py::arg("params") = poisson::mg::CompositeParams{},
        R"doc(
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
)doc");

  // Module metadata
  m.attr("__version__") = "0.1.0";
#if defined(POISSON_HAVE_FFTW3)
  m.attr("has_fftw3") = true;
#else
  m.attr("has_fftw3") = false;
#endif
}
