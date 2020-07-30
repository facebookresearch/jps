#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace tabular {

struct Options {
  std::string method = "search";
  std::string firstRandomInfoSetKey;
  bool computeReach = false;
  bool showBetter = false;

  bool verbose = false;
  int seed = 1;
  float exploreFactor = 0.01;
  float alpha = 0.5;
  float perturbChance = 0.0;
  float perturbPolicy = 0.0;

  int maxDepth = 0;

  bool gtCompute = false;
  bool gtOverride = false;

  bool use2ndOrder = false;

  bool skipSingleInfoSetOpt = false;
  bool skipSameDeltaPolicy = false;
};

using Policies = std::unordered_map<std::string, std::vector<float>>; 

}  // namespace tabular
