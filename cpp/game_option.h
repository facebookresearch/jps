#pragma once
#include <string>

namespace bridge {

struct GameOption {
  int tables = 2;
  bool verbose = false;
  float discount = 1.0;
  bool trainBidding = true;
  bool trainPlaying = false;
  int displayFreq = 20000;

  // sampler:
  //    "uniform": random sampling
  //    "seq":     sequential (and returns false for reset() if we run out of
  //    samples)
  std::string sampler = "uniform";
  std::string featureVer;
  bool saveFeatureHistory = false;

  bool saveOutput = false;

  // Whether we use fixed vul or fixed dealer (-1 = random)
  int fixedVul = -1;
  int fixedDealer = -1;
};

}  // namespace bridge
