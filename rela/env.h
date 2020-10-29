// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// /

#pragma once

#include <future>
#include <vector>

#include "rela/types.h"
#include "rela/utils.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace rela {

enum PlayerGroup { GRP_NATURE = 0, GRP_1, GRP_2, GRP_3 };

struct EnvSpec {
  // = 0 means that no feature is available.
  int featureSize;
  // Max #round since start. -1 means not usable.
  int maxActionRound;
  // Index by #players (e.g., for bridge the size of the two following vectors
  // would be 4)
  std::vector<int> maxNumActions;
  std::vector<PlayerGroup> players;
};

using LegalAction = std::pair<int, std::string>;

class Env {
 public:
  Env() = default;

  virtual ~Env() = default;

  // Return false if the environment won't work anymore (e.g., it has gone
  // through all samples).
  virtual bool reset() = 0;

  virtual void step(int action) = 0;

  // If the env also wants a complete reply from the neural network, override
  // this function.
  virtual void setReply(const TensorDict&) {
  }

  // State retrievers.
  virtual bool terminated() const = 0;
  // Some game might have multiple subgames in one games. Like Bridge.
  virtual bool subgameEnd() const {
    return false;
  }
  virtual int playerIdx() const = 0;
  virtual float playerReward(int) const = 0;

  // TODO: Not a good design.
  virtual float playerRawScore(int playerIdx) const {
    (void)playerIdx;
    throw std::runtime_error("playerRawScore is not implemented!");
  }

  virtual int maxNumAction() const = 0;
  virtual std::vector<LegalAction> legalActions() const {
    // Get legal actions for that particular state.
    // Default behavior: everything is legal. The derived class can override
    // this.
    if (terminated()) return {};
    return rela::utils::intSeq2intStrSeq(rela::utils::getIncSeq(maxNumAction()));
  }

  // Return partners playerIndices.
  virtual std::vector<int> partnerIndices(int playerIdx) const {
    (void)playerIdx;
    throw std::runtime_error("partnerIndices is not implemented!");
  }

  // Optional
  // Get feature representation.
  virtual TensorDict feature() const {
    return {};
  }

  virtual int featureDim() const { 
    return spec().featureSize;
  }

  // Game description for each agent. Each string corresponds to a unique
  // information set. Different information set is represented by a different
  // string.
  virtual std::string infoSet() const {
    throw std::runtime_error("infoSet() is not implemented!");
  }

  // Description of complete information, return "" if the function is unusable.
  virtual std::string completeCompactDesc() const {
    throw std::runtime_error("completeCompactDesc() is not implemented!");
  }

  // Printable game Description.
  virtual std::string info() const {
    return "";
  }

  // Json.
  virtual json jsonObj() const {
    return json();
  }

  // clone method.
  // It is allowed that the object is not clonable.
  virtual std::unique_ptr<Env> clone() const {
    return nullptr;
  }

  // This should be class method.
  virtual EnvSpec spec() const = 0;

  // Some visualization code, can be overridden.
  virtual std::string action2str(int action) const { return std::to_string(action); }
  virtual int str2action(const std::string &s) const { return std::stoi(s); }

 protected:
  torch::Tensor legalActionMask() const { 
    torch::Tensor legalMove = torch::zeros({maxNumAction()});
    auto legals = legalActions();

    auto f = legalMove.accessor<float, 1>();
    for (const auto & idx : legals) {
      f[idx.first] = 1.0;
    }
    return legalMove;
  }
};

class OptimalStrategy {
 public:
  virtual std::vector<float> getOptimalStrategy(
      const std::string& infoSet) const = 0;
};

}  // namespace rela
