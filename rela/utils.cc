#include "rela/utils.h"

#include <algorithm>
#include <functional>

namespace rela {
namespace utils {

std::vector<std::pair<float, int>> getSortedProb(const TensorDict& reply,
                                                 const std::string& key,
                                                 float threshold) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const int n = prob.numel();
  const float* prob_data = prob.data_ptr<float>();
  std::vector<std::pair<float, int>> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (prob_data[i] > threshold) {
      result.emplace_back(prob_data[i], i);
    }
  }
  std::sort(
      result.begin(), result.end(), std::greater<std::pair<float, int>>());
  return result;
}

std::vector<float> getVectorSel(const TensorDict& reply, const std::string& key, const std::vector<int>& sel) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const int n = prob.numel();
  const float* prob_data = prob.data_ptr<float>();
  std::vector<float> result;
  result.reserve(sel.size());
  for (const auto& idx : sel) {
    result.emplace_back(prob_data[idx]);
  }
  return result;
}

}  // namespace utils
}  // namespace rela
