#include "rela/utils.h"

#include <algorithm>
#include <functional>

namespace rela {
namespace utils {

void copyTensors(const TensorDict& src, TensorDict& dst) {
  RELA_CHECK_EQ(src.size(), dst.size());
  for (const auto& it : src) {
    const auto& key = it.first;
    const auto& srcTensor = it.second;
    auto& dstTensor = dst.at(key);
    RELA_CHECK_EQ(dstTensor.sizes(), srcTensor.sizes());
    RELA_CHECK_EQ(dstTensor.dtype(), srcTensor.dtype());
    dstTensor.copy_(srcTensor);
  }
}

void copyTensorsByIndex(const TensorDict& src, TensorDict& dst,
                        const torch::Tensor& idx) {
  RELA_CHECK_EQ(src.size(), dst.size());
  RELA_CHECK_GT(idx.size(0), 0);
  const int64_t idx_size = idx.size(0);
  for (const auto& it : src) {
    const auto& key = it.first;
    const auto& srcTensor = it.second;
    auto& dstTensor = dst.at(key);
    RELA_CHECK_EQ(srcTensor.size(0), idx_size);
    RELA_CHECK_EQ(dstTensor.dtype(), srcTensor.dtype());
    dstTensor.index_copy_(0, idx, srcTensor);
  }
}

bool tensorDictEq(const TensorDict& a, const TensorDict& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (const auto& ita : a) {
    const auto itb = b.find(ita.first);
    if (itb == b.cend()) {
      return false;
    }
    if ((itb->second != ita.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

TensorDict tensorDictNarrow(const TensorDict& dict, int64_t dim, int64_t start,
                            int64_t len, bool squeeze, bool clone) {
  TensorDict result;
  for (const auto& it : dict) {
    auto cur = it.second.narrow(dim, start, len);
    if (squeeze) {
      RELA_CHECK_EQ(len, 1);
      cur = cur.squeeze(dim);
    }
    if (clone) {
      result.emplace(it.first, cur.clone());
    } else {
      result.emplace(it.first, std::move(cur));
    }
  }
  return result;
}

TensorDict tensorDictJoin(const TensorVecDict& dict, int64_t dim) {
  TensorDict result;
  for (const auto& it : dict) {
    result.emplace(it.first, torch::stack(it.second, dim));
  }
  return result;
}

TensorDict vectorTensorDictJoin(const std::vector<TensorDict>& vec,
                                int64_t dim) {
  RELA_CHECK(!vec.empty());
  TensorVecDict dict;
  for (const auto& v : vec) {
    for (const auto& kv : v) {
      auto it = dict.find(kv.first);
      if (it == dict.end()) {
        dict.emplace(kv.first, std::vector<torch::Tensor>{kv.second});
      } else {
        it->second.emplace_back(kv.second);
      }
    }
  }
  return tensorDictJoin(dict, dim);
}

void appendTensorDict(TensorDict& dict, const TensorDict& add) {
  for (const auto& it : add) {
    const auto ret = dict.insert(it);
    RELA_CHECK(ret.second, "The key ", it.first, " already exists.");
  }
}

std::vector<int> getTensorSize(const TensorDict& dict, const std::string& key) {
  const auto it = dict.find(key);
  RELA_CHECK(it != dict.cend(), "The key " + key + "doesn't exist.");
  const auto sizes = it->second.sizes();
  return std::vector<int>(sizes.cbegin(), sizes.cend());
}

std::pair<float, int> getMaxProb(const TensorDict& reply,
                                 const std::string& key) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const int n = prob.numel();
  const float* prob_data = prob.data_ptr<float>();
  std::pair<float, int> res(std::numeric_limits<float>::lowest(), -1);
  for (int i = 0; i < n; ++i) {
    if (prob_data[i] > res.first) {
      res.first = prob_data[i];
      res.second = i;
    }
  }
  return res;
}

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
  std::sort(result.begin(), result.end(),
            std::greater<std::pair<float, int>>());
  return result;
}

std::vector<float> getVectorSel(const TensorDict& reply, const std::string& key,
                                const std::vector<int>& sel) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const float* prob_data = prob.data_ptr<float>();
  std::vector<float> result;
  result.reserve(sel.size());
  for (const auto& idx : sel) {
    result.emplace_back(prob_data[idx]);
  }
  return result;
}

std::vector<float> getVectorSel(const TensorDict& reply, const std::string& key,
                                const std::vector<std::pair<int, std::string>>& sel) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const float* prob_data = prob.data_ptr<float>();
  std::vector<float> result;
  result.reserve(sel.size());
  for (const auto& idx : sel) {
    result.emplace_back(prob_data[idx.first]);
  }
  return result;
}

std::vector<std::pair<float, int>> getVectorSelPair(
    const TensorDict& reply, const std::string& key,
    const std::vector<int>& sel) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const float* prob_data = prob.data_ptr<float>();
  std::vector<std::pair<float, int>> result;
  result.reserve(sel.size());
  for (const auto& idx : sel) {
    result.emplace_back(prob_data[idx], idx);
  }
  return result;
}

std::vector<std::pair<float, int>> getVectorSelPair(
    const TensorDict& reply, const std::string& key,
    const std::vector<std::pair<int, std::string>>& sel) {
  const auto& prob = rela::utils::get(reply, key);
  assert(prob.is_contiguous());
  const float* prob_data = prob.data_ptr<float>();
  std::vector<std::pair<float, int>> result;
  result.reserve(sel.size());
  for (const auto& idx : sel) {
    result.emplace_back(prob_data[idx.first], idx.first);
  }
  return result;
}

}  // namespace utils
}  // namespace rela
