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
#include <pybind11/stl.h>

#include "poisson/core/grid.hpp"
#include "poisson/fv/solver1d.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/iter/poisson_cg.hpp"
#include "poisson/linalg/thomas.hpp"

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
``V(0) = uL`` and ``V(L) = uR``. Internally calls the Thomas algorithm.

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

  py::class_<poisson::fv::SORParams>(m, "SORParams",
                                     "Tuning parameters for the SOR solver.")
      .def(py::init<>())
      .def_readwrite("omega",    &poisson::fv::SORParams::omega,
                     "Relaxation factor. ``-1`` means auto "
                     "(``omega_opt`` from grid size).")
      .def_readwrite("tol",      &poisson::fv::SORParams::tol,
                     "Stopping criterion on the max-norm of the residual.")
      .def_readwrite("max_iter", &poisson::fv::SORParams::max_iter,
                     "Hard iteration cap.");

  py::class_<poisson::fv::SORReport>(m, "SORReport",
                                     "Result of an SOR run.")
      .def_readonly("iterations", &poisson::fv::SORReport::iterations,
                    "Number of sweeps actually performed.")
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
    Relaxation factor. Default ``-1`` triggers
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
In-place variant of :meth:`solve` that reuses the caller's ``V``.

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
                     "Hard iteration cap.");

  py::class_<poisson::iter::CGReport>(m, "CGReport",
                                      "Result of a CG / PCG run.")
      .def_readonly("iterations", &poisson::iter::CGReport::iterations,
                    "Number of CG iterations actually performed.")
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

Direct ``O(N log N)`` solver: forward DST, divide by eigenvalues,
backward DST. The FFTW plan is built once at construction.

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

Direct ``O(N^2 log N)`` solver via tensor product of 1D DSTs.

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

  // Module metadata
  m.attr("__version__") = "0.1.0";
#if defined(POISSON_HAVE_FFTW3)
  m.attr("has_fftw3") = true;
#else
  m.attr("has_fftw3") = false;
#endif
}
