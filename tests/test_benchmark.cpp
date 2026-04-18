// Research-grade physical benchmarks for the Poisson solvers.
//
// Problem A: 2D Gaussian charge blob in a grounded rectangular box.
//
//   -eps0 (V_xx + V_yy) = rho(x, y)     on (0, Lx) x (0, Ly)
//   V = 0                                on the boundary
//   rho(x, y) = Q / (2 pi sigma^2) * exp(-r^2 / (2 sigma^2))
//     with r^2 = (x - xc)^2 + (y - yc)^2.
//
// This is the textbook problem in Jackson, *Classical Electrodynamics*
// (3rd ed.), chapter 2, with a smooth bounded source instead of a delta.
// It models a charge blob (e.g. a plasma filament) in a grounded cavity.
//
// Exact reference: eigenfunction expansion on the box.
//   V(x, y) = sum_{m,n=1}^inf V_hat_{mn} sin(m pi x/Lx) sin(n pi y/Ly)
//   V_hat_{mn} = rho_hat_{mn} / (eps0 * lambda_{mn})
//   lambda_{mn} = (m pi / Lx)^2 + (n pi / Ly)^2
// For the Gaussian source, when sigma << Lx, Ly and (xc, yc) is not near
// a boundary, the boundary-tail error is bounded by erfc((xc - 0)/(sigma*sqrt 2))
// and can be driven below 1e-14 by choosing xc > ~7 sigma. The Fourier
// coefficients are then
//   rho_hat_{mn} = 4Q/(Lx Ly) sin(m pi xc/Lx) sin(n pi yc/Ly)
//                  * exp(-0.5 ((m pi sigma/Lx)^2 + (n pi sigma/Ly)^2)).
// The Gaussian factor guarantees exponential convergence in m, n so
// truncating at M = 80 gives machine precision for sigma/L >= 0.05.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "poisson/spectral/dst2d.hpp"

using poisson::spectral::DSTSolver2D;

namespace {

struct GaussianBox {
  double Lx, Ly;
  double xc, yc;
  double sigma;
  double Q;
  double eps0;
};

Eigen::MatrixXd gaussian_rho(int Nx, int Ny, const GaussianBox& p) {
  const double hx = p.Lx / (Nx + 1);
  const double hy = p.Ly / (Ny + 1);
  Eigen::MatrixXd rho(Nx, Ny);
  const double norm = p.Q / (2.0 * std::numbers::pi * p.sigma * p.sigma);
  for (int j = 1; j <= Ny; ++j) {
    const double y = j * hy;
    for (int i = 1; i <= Nx; ++i) {
      const double x = i * hx;
      const double r2 = (x - p.xc) * (x - p.xc) + (y - p.yc) * (y - p.yc);
      rho(i - 1, j - 1) = norm * std::exp(-r2 / (2.0 * p.sigma * p.sigma));
    }
  }
  return rho;
}

// Truncated Fourier series reference, continuous formulation.
Eigen::MatrixXd fourier_reference(int Nx, int Ny,
                                   const GaussianBox& p, int M) {
  const double hx = p.Lx / (Nx + 1);
  const double hy = p.Ly / (Ny + 1);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(Nx, Ny);

  // Precompute sin tables for O(M Nx + M Ny + M^2 Nx Ny) complexity.
  Eigen::MatrixXd sx(M, Nx);   // sx(m-1, i-1) = sin(m pi i hx / Lx)
  Eigen::MatrixXd sy(M, Ny);
  for (int m = 1; m <= M; ++m) {
    const double km = m * std::numbers::pi / p.Lx;
    for (int i = 1; i <= Nx; ++i) sx(m - 1, i - 1) = std::sin(km * i * hx);
  }
  for (int n = 1; n <= M; ++n) {
    const double kn = n * std::numbers::pi / p.Ly;
    for (int j = 1; j <= Ny; ++j) sy(n - 1, j - 1) = std::sin(kn * j * hy);
  }

  for (int m = 1; m <= M; ++m) {
    const double km = m * std::numbers::pi / p.Lx;
    const double smc = std::sin(km * p.xc);
    const double gm  = std::exp(-0.5 * km * km * p.sigma * p.sigma);
    for (int n = 1; n <= M; ++n) {
      const double kn = n * std::numbers::pi / p.Ly;
      const double snc = std::sin(kn * p.yc);
      const double gn  = std::exp(-0.5 * kn * kn * p.sigma * p.sigma);
      const double rho_hat = 4.0 * p.Q / (p.Lx * p.Ly) * smc * snc * gm * gn;
      const double lam = km * km + kn * kn;
      const double V_hat = rho_hat / (p.eps0 * lam);
      // Rank-1 outer-product accumulation.
      V += V_hat * sx.row(m - 1).transpose() * sy.row(n - 1);
    }
  }
  return V;
}

// Solve the same problem on an (Nx, Ny) DST grid and return the L-infinity
// error against the truncated Fourier reference.
double solve_and_measure(int Nx, int Ny, const GaussianBox& p, int M) {
  const Eigen::MatrixXd rho = gaussian_rho(Nx, Ny, p);
  DSTSolver2D solver(Nx, Ny, p.Lx, p.Ly, p.eps0);
  const Eigen::MatrixXd V_num = solver.solve(rho);
  const Eigen::MatrixXd V_ref = fourier_reference(Nx, Ny, p, M);
  return (V_num - V_ref).cwiseAbs().maxCoeff();
}

// Energy-norm convergence measure: sum_{m,n} lambda_{mn} |V_hat_{mn}|^2 is the
// discrete H^1 semi-norm of V_num - V_ref. This is a mesh-independent scalar
// that converges cleanly at the asymptotic rate.
// We approximate it via the L2 norm of (V_num - V_ref_on_grid) weighted by
// the grid spacing: this gives an h-independent quantity that decreases as h^2
// (one order higher than L_inf thanks to error cancellation under integration).
double l2_err(int Nx, int Ny, const GaussianBox& p, int M) {
  const double hx = p.Lx / (Nx + 1), hy = p.Ly / (Ny + 1);
  const Eigen::MatrixXd rho = gaussian_rho(Nx, Ny, p);
  DSTSolver2D solver(Nx, Ny, p.Lx, p.Ly, p.eps0);
  const Eigen::MatrixXd V_num = solver.solve(rho);
  const Eigen::MatrixXd V_ref = fourier_reference(Nx, Ny, p, M);
  const double sq = (V_num - V_ref).array().square().sum();
  return std::sqrt(sq * hx * hy);
}

}  // namespace

TEST_CASE("Benchmark: Gaussian charge in a grounded 2D box (Jackson Ch. 2)",
          "[benchmark][spectral][2d]") {
  const GaussianBox p{.Lx = 1.0, .Ly = 1.0,
                       .xc = 0.5, .yc = 0.5,
                       .sigma = 0.08, .Q = 1.0, .eps0 = 1.0};
  const int M = 80;   // truncation; Gaussian decay => ~1e-14 convergence

  // Reference V_max (for scaling) on a 127^2 grid. V is O(Q / eps0), of
  // order unity here.
  const int N = 127;
  const Eigen::MatrixXd rho = gaussian_rho(N, N, p);
  DSTSolver2D solver(N, N, p.Lx, p.Ly, p.eps0);
  const Eigen::MatrixXd V_num = solver.solve(rho);
  const Eigen::MatrixXd V_ref = fourier_reference(N, N, p, M);

  const double err     = (V_num - V_ref).cwiseAbs().maxCoeff();
  const double v_max   = V_ref.cwiseAbs().maxCoeff();
  const double err_rel = err / v_max;

  // V is smooth (sigma = 0.08), dominant modes m, n ~ L / sigma ~ 12.
  // At N = 127 we expect Linf error ~ 1e-4 (relative ~ 2e-4).
  REQUIRE(err_rel < 5e-4);
  REQUIRE(err < 2e-4);
  // Sanity: the peak potential is order unity for Q = 1, box = unit square.
  REQUIRE(v_max > 0.1);
  REQUIRE(v_max < 10.0);
}

TEST_CASE("Benchmark: DSTSolver2D shows 2nd-order convergence on the Gaussian",
          "[benchmark][spectral][2d]") {
  const GaussianBox p{.Lx = 1.0, .Ly = 1.0,
                       .xc = 0.5, .yc = 0.5,
                       .sigma = 0.08, .Q = 1.0, .eps0 = 1.0};
  const int M = 80;

  // With the Gaussian well inside the box (xc = yc = 0.5, sigma = 0.08),
  // the boundary tail is < 1e-9 and the reference is free of wrap-around
  // bias. Both L_inf and L2 errors then show a clean O(h^2) rate.
  const double e1 = solve_and_measure( 63,  63, p, M);
  const double e2 = solve_and_measure(127, 127, p, M);
  const double e3 = solve_and_measure(255, 255, p, M);
  // Expect factor-4 reduction; allow 10% slack against round-off.
  REQUIRE(e1 / e2 > 3.6);
  REQUIRE(e2 / e3 > 3.6);
  REQUIRE(e3 < 5e-5);

  const double l1 = l2_err( 63,  63, p, M);
  const double l2 = l2_err(127, 127, p, M);
  const double l3 = l2_err(255, 255, p, M);
  REQUIRE(l1 / l2 > 3.6);
  REQUIRE(l2 / l3 > 3.6);
  REQUIRE(l3 < 1e-5);
}
