#pragma once

#include <string>
#include <string_view>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace poisson::io {

/// Convert an Eigen vector to a nlohmann::json array.
inline nlohmann::json to_json(Eigen::Ref<const Eigen::VectorXd> v) {
  nlohmann::json j = nlohmann::json::array();
  for (Eigen::Index i = 0; i < v.size(); ++i) j.push_back(v(i));
  return j;
}

/// Convert an Eigen matrix to a nlohmann::json array-of-arrays (row-major).
inline nlohmann::json to_json(Eigen::Ref<const Eigen::MatrixXd> M) {
  nlohmann::json j = nlohmann::json::array();
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    nlohmann::json row = nlohmann::json::array();
    for (Eigen::Index k = 0; k < M.cols(); ++k) row.push_back(M(i, k));
    j.push_back(std::move(row));
  }
  return j;
}

/// Parse a 1D JSON array into an Eigen vector.
inline Eigen::VectorXd vector_from_json(const nlohmann::json& j) {
  Eigen::VectorXd v(j.size());
  for (std::size_t i = 0; i < j.size(); ++i) v(i) = j.at(i).get<double>();
  return v;
}

/// Parse a row-major 2D JSON array into an Eigen matrix.
inline Eigen::MatrixXd matrix_from_json(const nlohmann::json& j) {
  const Eigen::Index rows = j.size();
  const Eigen::Index cols = rows > 0 ? j[0].size() : 0;
  Eigen::MatrixXd M(rows, cols);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index k = 0; k < cols; ++k) {
      M(i, k) = j.at(i).at(k).get<double>();
    }
  }
  return M;
}

}  // namespace poisson::io
