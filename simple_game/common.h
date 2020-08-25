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
#include <unordered_map>

namespace tabular {

enum VerboseLevel { SILENT = 0, NORMAL, VERBOSE };

struct Options {
  std::string method = "search";
  std::string firstRandomInfoSetKey;
  bool computeReach = false;
  bool showBetter = false;

  VerboseLevel verbose = NORMAL;
  int seed = 1;
  float exploreFactor = 0.01;
  float alpha = 0.5;
  float perturbChance = 0.0;
  float perturbPolicy = 0.0;

  int maxDepth = 0;
  int numSample = 0;

  bool gtCompute = false;
  bool gtOverride = false;

  bool use2ndOrder = false;

  bool skipSingleInfoSetOpt = false;
  bool skipSameDeltaPolicy = false;
};

using Policies = std::unordered_map<std::string, std::vector<float>>; 

}  // namespace tabular
