// pybind11 bindings for poisson_cpp. Built only when
// `-DPOISSON_BUILD_PYTHON=ON` is passed to CMake. Compatible with C++20.
//
// Build (from the repo root):
//   cmake -B build -DPOISSON_BUILD_PYTHON=ON
//   cmake --build build --target poisson_py -j
//
// After building, `build/python/poisson_cpp*.so` is the importable module.
// Adjust PYTHONPATH or copy the .so to use it:
//   PYTHONPATH=build/python python -c "import poisson_cpp; ..."

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
  py::class_<poisson::Grid1D>(m, "Grid1D")
      .def(py::init<double, int>(), py::arg("L"), py::arg("N"))
      .def_property_readonly("N",  [](const poisson::Grid1D& g){ return g.N; })
      .def_property_readonly("L",  [](const poisson::Grid1D& g){ return g.L; })
      .def_property_readonly("dx", &poisson::Grid1D::dx)
      .def("x", &poisson::Grid1D::x)
      .def("__repr__", [](const poisson::Grid1D& g) {
        return "Grid1D(L=" + std::to_string(g.L) +
               ", N=" + std::to_string(g.N) + ")";
      });

  py::class_<poisson::Grid2D>(m, "Grid2D")
      .def(py::init<double, double, int, int>(),
           py::arg("Lx"), py::arg("Ly"), py::arg("Nx"), py::arg("Ny"))
      .def_readonly("Nx", &poisson::Grid2D::Nx)
      .def_readonly("Ny", &poisson::Grid2D::Ny)
      .def_property_readonly("dx", &poisson::Grid2D::dx)
      .def_property_readonly("dy", &poisson::Grid2D::dy);

  // --- linalg --------------------------------------------------------------
  m.def("thomas", &poisson::linalg::thomas,
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
        "Solve a tridiagonal linear system A x = d by the Thomas algorithm.");

  // --- FV solvers ----------------------------------------------------------
  m.def("solve_poisson_1d",
        [](const Eigen::VectorXd& rho, double uL, double uR,
           const poisson::Grid1D& grid) {
          return poisson::fv::solve_poisson_1d(rho, uL, uR, grid);
        },
        py::arg("rho"), py::arg("uL"), py::arg("uR"), py::arg("grid"),
        "Finite-volume 1D Poisson with Dirichlet BC, uniform permittivity.");

  py::class_<poisson::fv::SORParams>(m, "SORParams")
      .def(py::init<>())
      .def_readwrite("omega",    &poisson::fv::SORParams::omega)
      .def_readwrite("tol",      &poisson::fv::SORParams::tol)
      .def_readwrite("max_iter", &poisson::fv::SORParams::max_iter);

  py::class_<poisson::fv::SORReport>(m, "SORReport")
      .def_readonly("iterations", &poisson::fv::SORReport::iterations)
      .def_readonly("residual",   &poisson::fv::SORReport::residual)
      .def("__repr__", [](const poisson::fv::SORReport& r) {
        return "SORReport(iterations=" + std::to_string(r.iterations) +
               ", residual=" + std::to_string(r.residual) + ")";
      });

  py::class_<poisson::fv::Solver2D>(m, "Solver2D")
      .def(py::init<const poisson::Grid2D&, double, double, double>(),
           py::arg("grid"), py::arg("eps"), py::arg("uL"), py::arg("uR"),
           "FV 2D SOR solver with constant permittivity.")
      .def(py::init<const poisson::Grid2D&, Eigen::MatrixXd, double, double>(),
           py::arg("grid"), py::arg("eps"), py::arg("uL"), py::arg("uR"),
           "FV 2D SOR solver with spatially-varying permittivity.")
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
           "Run SOR from a zero initial guess on the given right-hand side. "
           "Returns (V, report): V is the solution as a numpy array, report "
           "carries the iteration count and final residual.")
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
           "In-place variant. Requires V and rho to be Fortran-ordered "
           "numpy arrays (`order='F'`) matching Eigen's column-major layout.");

  // --- CG / PCG iterative solvers -----------------------------------------
  py::class_<poisson::iter::CGParams>(m, "CGParams")
      .def(py::init<>())
      .def_readwrite("tol",      &poisson::iter::CGParams::tol)
      .def_readwrite("max_iter", &poisson::iter::CGParams::max_iter);

  py::class_<poisson::iter::CGReport>(m, "CGReport")
      .def_readonly("iterations", &poisson::iter::CGReport::iterations)
      .def_readonly("residual",   &poisson::iter::CGReport::residual)
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
        "Solve -eps Laplacian V = rho on a cell-centered FV grid with "
        "Dirichlet in x / Neumann in y, via Conjugate Gradient. "
        "V must be Fortran-ordered (`order='F'`) and is updated in place. "
        "Returns (report, history): report carries iteration count and "
        "final relative residual; history is a list of ||r||/||b|| at "
        "each iteration when record_history=True, else empty.");

#if defined(POISSON_HAVE_FFTW3)
  py::class_<poisson::spectral::DSTSolver1D>(m, "DSTSolver1D")
      .def(py::init<int, double, double>(),
           py::arg("N"), py::arg("L"), py::arg("eps0") = 1.0,
           "Spectral 1D Poisson solver via DST-I (FFTW).")
      .def("solve", &poisson::spectral::DSTSolver1D::solve, py::arg("rho"),
           "Solve eps0 V'' = -rho on the N interior nodes.");

  py::class_<poisson::spectral::DSTSolver2D>(m, "DSTSolver2D")
      .def(py::init<int, int, double, double, double>(),
           py::arg("Nx"), py::arg("Ny"), py::arg("Lx"), py::arg("Ly"),
           py::arg("eps0") = 1.0,
           "Spectral 2D Poisson solver via DST-I (FFTW), homogeneous "
           "Dirichlet BC on all four faces.")
      .def("solve", &poisson::spectral::DSTSolver2D::solve, py::arg("rho"),
           "Solve -eps0 Laplacian V = rho on the (Nx, Ny) interior grid.");
#endif

  // Module metadata
  m.attr("__version__") = "0.1.0";
#if defined(POISSON_HAVE_FFTW3)
  m.attr("has_fftw3") = true;
#else
  m.attr("has_fftw3") = false;
#endif
}
