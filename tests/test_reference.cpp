#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "poisson/core/grid.hpp"
#include "poisson/fv/dielectric.hpp"
#include "poisson/fv/solver1d.hpp"
#include "poisson/fv/solver2d.hpp"
#include "poisson/io/json_io.hpp"
#include "poisson/linalg/thomas.hpp"

namespace fs = std::filesystem;

namespace {

// Locate the repository's data/reference/ directory by walking up from the
// current test binary. Returns empty path if not found.
fs::path reference_dir() {
  for (fs::path p : {fs::path(__FILE__).parent_path(),
                     fs::current_path(),
                     fs::current_path().parent_path()}) {
    while (!p.empty() && p != p.root_path()) {
      if (fs::exists(p / "data" / "reference")) {
        return p / "data" / "reference";
      }
      p = p.parent_path();
    }
  }
  return {};
}

nlohmann::json load_json(const fs::path& p) {
  std::ifstream in(p);
  return nlohmann::json::parse(in);
}

}  // namespace

TEST_CASE("Reference: thomas matches Python snapshot", "[reference][thomas]") {
  const fs::path dir = reference_dir();
  if (dir.empty()) {
    SKIP("data/reference/ not found; run python/dump_reference.py first.");
  }
  const fs::path jpath = dir / "thomas_dominant_N40.json";
  if (!fs::exists(jpath)) {
    SKIP("thomas_dominant_N40.json not found; run dump_reference.py first.");
  }
  const auto j = load_json(jpath);

  const auto a = poisson::io::vector_from_json(j.at("a"));
  const auto b = poisson::io::vector_from_json(j.at("b"));
  const auto c = poisson::io::vector_from_json(j.at("c"));
  const auto d = poisson::io::vector_from_json(j.at("d"));
  const auto x_ref = poisson::io::vector_from_json(j.at("x_ref"));

  const Eigen::VectorXd x = poisson::linalg::thomas(a, b, c, d);
  REQUIRE((x - x_ref).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Reference: solve_poisson_1d matches Python snapshot",
          "[reference][fv]") {
  const fs::path dir = reference_dir();
  if (dir.empty()) SKIP("data/reference/ not found.");
  const fs::path jpath = dir / "solver1d_uniform_rho_N50.json";
  if (!fs::exists(jpath)) SKIP("solver1d_uniform_rho_N50.json not found.");
  const auto j = load_json(jpath);

  const int N = j.at("N").get<int>();
  const double L = j.at("L").get<double>();
  const double uL = j.at("uL").get<double>();
  const double uR = j.at("uR").get<double>();
  const Eigen::VectorXd rho = poisson::io::vector_from_json(j.at("rho"));
  const Eigen::VectorXd V_ref = poisson::io::vector_from_json(j.at("V_ref"));

  poisson::Grid1D grid(L, N);
  const Eigen::VectorXd V = poisson::fv::solve_poisson_1d(rho, uL, uR, grid);
  REQUIRE((V - V_ref).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Reference: dielectric 1D matches Python snapshot",
          "[reference][fv][dielectric]") {
  const fs::path dir = reference_dir();
  if (dir.empty()) SKIP("data/reference/ not found.");
  const fs::path jpath = dir / "dielectric_zero_rho_N40.json";
  if (!fs::exists(jpath)) SKIP("dielectric_zero_rho_N40.json not found.");
  const auto j = load_json(jpath);

  const int N = j.at("N").get<int>();
  const double L = j.at("L").get<double>();
  const double uL = j.at("uL").get<double>();
  const double uR = j.at("uR").get<double>();
  const Eigen::VectorXd eps_r = poisson::io::vector_from_json(j.at("eps_r"));
  const Eigen::VectorXd rho   = poisson::io::vector_from_json(j.at("rho"));
  const Eigen::VectorXd V_ref = poisson::io::vector_from_json(j.at("V_ref"));

  poisson::Grid1D grid(L, N);
  const Eigen::VectorXd V =
      poisson::fv::solve_poisson_1d(rho, eps_r, uL, uR, grid);
  REQUIRE((V - V_ref).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Reference: SOR 2D matches Python snapshot", "[reference][fv][sor]") {
  const fs::path dir = reference_dir();
  if (dir.empty()) SKIP("data/reference/ not found.");
  const fs::path jpath = dir / "sor2d_linear_N24.json";
  if (!fs::exists(jpath)) SKIP("sor2d_linear_N24.json not found.");
  const auto j = load_json(jpath);

  const int N = j.at("N").get<int>();
  const double L = j.at("L").get<double>();
  const double uL = j.at("uL").get<double>();
  const double uR = j.at("uR").get<double>();
  const Eigen::MatrixXd V_ref = poisson::io::matrix_from_json(j.at("V_ref"));

  poisson::Grid2D grid(L, L, N, N);
  poisson::fv::Solver2D solver(grid, 1.0, uL, uR);
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(N, N);
  const auto r = solver.solve(V, rho, {.tol = 1e-11, .max_iter = 50'000});
  REQUIRE(r.residual < 1e-10);

  // SOR is iterative: allow a small tolerance on top of the Python snapshot
  // (both should have converged below 1e-10).
  REQUIRE((V - V_ref).cwiseAbs().maxCoeff() < 1e-8);
}
