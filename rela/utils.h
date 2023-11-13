// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "rela/logging.h"
#include "rela/types.h"

namespace rela {
namespace utils {

template <typename T>
struct ToDataType;

template <>
struct ToDataType<float> {
  static constexpr torch::ScalarType value = torch::kFloat;
};

template <>
struct ToDataType<int64_t> {
  static constexpr torch::ScalarType value = torch::kInt64;
};

inline int64_t getProduct(const std::vector<int64_t>& nums) {
  return std::accumulate(
      nums.cbegin(), nums.cend(), int64_t(1), std::multiplies<int64_t>());
}

template <typename T>
inline std::vector<T> pushLeft(T left, const std::vector<T>& vals) {
  std::vector<T> result(vals.size() + 1);
  result[0] = left;
  std::memcpy(result.data() + 1, vals.data(), vals.size() * sizeof(T));
  return result;
}

template <typename T>
inline void printVector(const std::vector<T>& vec) {
  for (const auto& v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMapKey(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMap(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ": " << name2sth.second << std::endl;
  }
  // std::cout << std::endl;
}

// TODO: rewrite the above functions with this template
template <typename Func>
inline TensorDict tensorDictApply(const TensorDict& dict, const Func& func) {
  TensorDict output;
  for (const auto& name2tensor : dict) {
    output.emplace(name2tensor.first, func(name2tensor.second));
  }
  return output;
}

void copyTensors(const TensorDict& src, TensorDict& dst);

void copyTensorsByIndex(const TensorDict& src, TensorDict& dst,
                        const torch::Tensor& idx);

bool tensorDictEq(const TensorDict& a, const TensorDict& b);

/*
 * indexes into a TensorDict
 */
inline TensorDict tensorDictIndex(const TensorDict& batch, size_t i) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

TensorDict tensorDictNarrow(const TensorDict& dict, int64_t dim, int64_t start,
                            int64_t len, bool squeeze, bool clone);

inline TensorDict tensorDictSqueeze(const TensorDict& batch, int dim) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second.squeeze(dim)});
  }
  return result;
}

inline TensorDict tensorDictUnsqueeze(const TensorDict& batch, int dim) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second.unsqueeze(dim)});
  }
  return result;
}


inline TensorDict tensorDictClone(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

inline TensorDict tensorDictZerosLike(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

// TODO: remove this function if not used
/*
 * Appends a TensorDict to a TensorVecDict.
 * If batch is empty, initialize it as input.
 * This function should be used in conjunction with tensorDictJoin to batch
 * together many TensorDicts
 */
inline void tensorVecDictAppend(TensorVecDict& batch, const TensorDict& input) {
  for (auto& name2tensor : input) {
    auto it = batch.find(name2tensor.first);
    if (it == batch.end()) {
      std::vector<torch::Tensor> singleton = {name2tensor.second};
      batch.insert({name2tensor.first, singleton});
    } else {
      it->second.push_back(name2tensor.second);
    }
  }
}

// Given a TensorVecDict, returns a Tensor that concats them together
TensorDict tensorDictJoin(const TensorVecDict& dict, int64_t dim);

TensorDict vectorTensorDictJoin(const std::vector<TensorDict>& vec,
                                int64_t dim);

// utils for convert dict[str, tensor] <-> ivalue
inline TensorDict iValueToTensorDict(const torch::IValue& value,
                                     torch::DeviceType device,
                                     bool detach) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();
  // auto ivalMap = dict->elements();
  for (auto& name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
    if (detach) {
      tensor = tensor.detach();
    }
    tensor = tensor.to(device);
    map.insert({name->string(), tensor});
  }
  return map;
}

// TODO: this may be simplified with constructor in the future version
inline TorchTensorDict tensorDictToTorchDict(const TensorDict& tensorDict,
                                             const torch::Device& device) {
  TorchTensorDict dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return dict;
}

inline void assertKeyExists(const TensorDict& tensorDict,
                            const std::vector<std::string>& keys) {
  for (const auto& k : keys) {
    if (tensorDict.find(k) == tensorDict.end()) {
      std::cout << "Key " << k << " does not exist! " << std::endl;
      std::cout << "Checking keys: " << std::endl;
      for (const auto& kk : keys) {
        std::cout << kk << ", ";
      }
      std::cout << std::endl;
      assert(false);
    }
  }
}

inline std::string printTensorDict(const TensorDict& tensorDict) {
  std::stringstream ss;
  for (const auto& k2v : tensorDict) {
    ss << k2v.first << ": [";
    for (int k = 0; k < k2v.second.dim(); ++k) {
      if (k > 0)
        ss << ", ";
      ss << k2v.second.size(k);
    }
    ss << "]" << std::endl;
  }
  return ss.str();
}

void appendTensorDict(TensorDict& dict, const TensorDict& add);

inline TensorDict splitTensorDict(TensorDict& t, char prefix) {
  TensorDict d;
  for (auto it = t.begin(); it != t.end();) {
    if (it->first[0] == prefix) {
      d[it->first] = it->second;
      it = t.erase(it);
    } else {
      ++it;
    }
  }
  return d;
}

inline TensorDict splitTensorDictClone(const TensorDict& t, char prefix) {
  TensorDict d;
  for (const auto& kv : t) {
    if (kv.first[0] == prefix) {
      d[kv.first] = kv.second.clone();
    }
  }
  return d;
}

inline void _combineTensorDictArgs(TensorDict& combined,
                                   int i,
                                   const TensorDict& d) {
  for (const auto& kv : d) {
    combined[std::to_string(i) + "." + kv.first] = kv.second;
  }
}

inline TensorDict combineTensorDictArgs(const TensorDict& d1,
                                        const TensorDict& d2) {
  TensorDict res;
  _combineTensorDictArgs(res, 0, d1);
  _combineTensorDictArgs(res, 1, d2);
  return res;
}

inline bool hasKey(const TensorDict& tensorDict, const std::string& key) {
  auto it = tensorDict.find(key);
  return it != tensorDict.end();
}

inline const torch::Tensor& get(const TensorDict& tensorDict,
                                const std::string& key) {
  auto it = tensorDict.find(key);
  if (it == tensorDict.end()) {
    std::cout << printTensorDict(tensorDict);
    throw std::runtime_error("key " + key + " doesn't exist");
  }
  return it->second;
}

template <typename T>
T getTensorDictScalar(const TensorDict& dict, const std::string& key) {
  const auto it = dict.find(key);
  if (it == dict.cend()) {
    std::cout << "key: " << key << " not found!" << std::endl;
    std::cout << printTensorDict(dict);
  }
  RELA_CHECK(it != dict.cend());
  RELA_CHECK_EQ(it->second.numel(), 1);
  return it->second.item<T>();
}

std::vector<int> getTensorSize(const TensorDict& dict, const std::string& key);

template <typename T>
inline TensorDict setTensorDictScalar(const std::string& key, T val) {
  TensorDict d;
  d.emplace(key, torch::zeros({1}, ToDataType<T>::value));
  d.begin()->second.fill_(val);
  return d;
}

inline std::vector<int> getIncSeq(int N, int start = 0) {
  std::vector<int> seq(N);
  // Everything is legal.
  std::iota(seq.begin(), seq.end(), start);
  return seq;
}

inline std::vector<std::pair<int, std::string>> intSeq2intStrSeq(const std::vector<int>& seq) {
  std::vector<std::pair<int, std::string>> res(seq.size());
  for (size_t i = 0; i < seq.size(); ++i) {
    res[i].first = seq[i];
    res[i].second = std::to_string(seq[i]);
  }
  return res;
}

std::pair<float, int> getMaxProb(const TensorDict& reply, const std::string& key);

std::vector<std::pair<float, int>> getSortedProb(const TensorDict& reply,
                                                 const std::string& key,
                                                 float threshold);

std::vector<float> getVectorSel(const TensorDict& reply, const std::string& key, const std::vector<int> &sel);
std::vector<float> getVectorSel(const TensorDict& reply, const std::string& key, const std::vector<std::pair<int, std::string>> &sel);
std::vector<std::pair<float, int>> getVectorSelPair(const TensorDict& reply, const std::string& key, const std::vector<int>& sel);
std::vector<std::pair<float, int>> getVectorSelPair(const TensorDict& reply, const std::string& key, const std::vector<std::pair<int, std::string>>& sel);

}  // namespace utils
}  // namespace rela
