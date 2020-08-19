// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 
// 
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <random>

#include "rela/types.h"

template <typename T>
std::string printVector(const std::vector<T> &v) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << v[i];
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::string printVectorCompact(const std::vector<T> &v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size(); ++i) {
    ss << v[i];
  }
  return ss.str();
}

template <typename Gen, typename T>
void addUniformNoise(Gen &rng, std::vector<T>& vec, T sigma) {
  if (sigma == 0)
    return;

  std::uniform_real_distribution<T> gen;
  for (auto &v : vec) {
    v += gen(rng) * sigma;
  }
}

template <typename Gen, typename T>
void addGaussianNoise(Gen &rng, std::vector<T>& vec, T sigma) {
  if (sigma == 0)
    return;

  std::normal_distribution<T> gen(0, sigma);
  for (auto &v : vec) {
    v += gen(rng);
  }
}

inline bool normalize(std::vector<float> &v) {
  float sum = 0.0;
  for (auto &vv : v) {
    sum += vv;
  } 
  if (sum == 0) return false;
  
  for (auto &vv : v) {
    vv /= sum;
  } 
  return true;
}

inline void relu(std::vector<float> &v) {
  for (auto &vv : v) {
    vv = std::max(vv, 0.0f);
  } 
}

inline void uniform(std::vector<float> &v) {
  for (auto &vv : v) {
    vv = 1.0f / v.size();
  }
}

inline void addMulti(std::vector<float> &v, const std::vector<float> &v1, float alpha = 1.0) {
  assert(v.size() == v1.size());
  for (int i = 0; i < (int)v.size(); ++i) {
    v[i] += alpha * v1[i];
  }
}

inline void multiply(std::vector<float> &v, float alpha) {
  for (int i = 0; i < (int)v.size(); ++i) {
    v[i] *= alpha;
  }
}
