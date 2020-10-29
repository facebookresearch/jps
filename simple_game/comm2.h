// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#pragma once

// #include "game_interface.h"
#include <sstream>
#include <cassert>

#include "rela/env.h"
#include "comm_options.h"

namespace simple {

static const std::vector<std::vector<float>> kPayOff = {
  { 10, 0, 0 }, { 4, 8, 4}, {10, 0, 0}, // card1 = card2 = 0
  { 0, 0, 10 }, { 4, 8, 4}, {0, 0, 10}, // card1 = 0, card2 = 1
  { 0, 0, 10 }, { 4, 8, 4}, {0, 0, 0},  // card1 = 1, card2 = 0
  { 10, 0, 0 }, { 4, 8, 4}, {10, 0, 0}  // card1 = 1, card2 = 1
};

static const std::unordered_map<std::string, std::vector<float>> kOptimalPolicy = {
  { "p1-c1=1", { 1, 0, 0} }, 
  { "p1-c1=0", { 0, 0, 1} },
  { "p2-c2=0-a=0", { 0, 0, 1} }, 
  { "p2-c2=0-a=1", { 0, 0, 1} }, // arbitrary
  { "p2-c2=0-a=2", { 1, 0, 0} },
  { "p2-c2=1-a=0", { 1, 0, 0} }, 
  { "p2-c2=1-a=1", { 0, 0, 1} }, // arbitrary
  { "p2-c2=1-a=2", { 0, 0, 1} }
};

// The game is from Bayesian Action Decoder (https://arxiv.org/abs/1811.01458)
class Communicate2 : public rela::Env {
 public:
   Communicate2(const CommOptions &opt) 
     : options_(opt) { 
   } 

   bool reset() override {
     public_action_ = -1;
     p2_action_ = -1;
     card1_ = -1;
     card2_ = -1;

     numGameFinished_ ++;

     bool noMoreGames = options_.seqEnumerate && (numGameFinished_ >= 4);
     return ! noMoreGames;
   }

   std::string info() const override {
     std::stringstream ss;
     ss << "InfoSet: " << infoSet() << ", card1: " << card1_ << ", card2: " << card2_ 
        << ", P1 Action: " << public_action_ << ", P2 Action: " << p2_action_ 
        << ", player: " << playerIdx() 
        << ", terminal: " << terminated();
     return ss.str();
   }

   int maxNumAction() const override {
     return 4;
   }

   std::vector<rela::LegalAction> legalActions() const override {
     if (terminated()) return {};
     
     std::vector<int> actions;
     if (card1_ < 0 && card2_ < 0) {
       actions = std::vector<int>{0, 1, 2, 3};
     } else {
       actions = std::vector<int>{0, 1, 2};
     }
     return rela::utils::intSeq2intStrSeq(actions);
   }

   int playerIdx() const override {
     // natural's turn
     if (card1_ < 0 && card2_ < 0) return 0;
     if (public_action_ < 0) return 1;
     return 2;
   }

   std::vector<int> partnerIndices(int playerIdx) const override {
     if (playerIdx == 0) return {};
     return { 3 - playerIdx };
   }

   std::string infoSet() const override {
     if (terminated()) return completeCompactDesc();

     if (card1_ < 0 && card2_ < 0) return "s";
     if (public_action_ < 0) return "p1-c1=" + std::to_string(card1_);
     return "p2-c2=" + std::to_string(card2_) + "-a=" + std::to_string(public_action_);
   }

   std::string completeCompactDesc() const override {
     if (card1_ < 0 && card2_ < 0) return "s";

     auto s = "c1=" + std::to_string(card1_) + "-c2=" + std::to_string(card2_);
     if (public_action_ >= 0) {
       s += "-a=" + std::to_string(public_action_); 
     }

     if (terminated()) s = "done-" + s + "-a=" + std::to_string(p2_action_);
     return s;
   }

   std::unique_ptr<rela::Env> clone() const override {
     return std::make_unique<Communicate2>(*this);
   } 
     
   void step(int action) override {
     if (card1_ < 0 && card2_ < 0) {
       card1_ = action % 2;
       card2_ = action / 2;
     } else if (public_action_ < 0) {
       assert(action < 3);
       public_action_ = action;
     } else {
       assert(action < 3);
       p2_action_ = action;
     } 
   }

   bool terminated() const override {
     return p2_action_ >= 0;
   }

   bool subgameEnd() const override { return terminated(); }

   float playerReward(int playerIdx) const override {
     if (! terminated() || playerIdx == 0) return 0.0f;

      int idx = card1_ * 2 + card2_;
      return kPayOff[idx * 3 + public_action_][p2_action_];
   }

   float playerRawScore(int idx) const override { return playerReward(idx); }

   // Include chance.
   rela::EnvSpec spec() const override { 
     return { 
       // featureSize: card1 with one-hot, card2 with one-hot, public_action
       4 + 3,
       // max number of round
       2,
       // Max number of actions for each player.
       { 4, 3, 3 }, 
       // Two players share the model.
       { rela::PlayerGroup::GRP_NATURE, rela::PlayerGroup::GRP_1,  rela::PlayerGroup::GRP_1 } 
     };
   }

   rela::TensorDict feature() const override {
     auto s = torch::zeros({ 4 + 3 });
     auto accessor = s.accessor<float, 1>();

     int idx = playerIdx();

     if (idx == 1 && card1_ >= 0) {
       accessor[card1_] = 1.0f;
     } else if (idx == 2 && card2_ >= 0) {
       accessor[2 + card2_] = 1.0f;
     }

     if (public_action_ >= 0) {
       accessor[4 + public_action_] = 1.0f;
     }

     auto tensorPlayerIdx = (playerIdx() == 1 ? torch::zeros({1}) : torch::ones({1}));

     return { { "s", s }, {"legal_move", legalActionMask() }, { "player_idx", tensorPlayerIdx } };
   }

 private:
   int public_action_ = -1;
   int p2_action_ = -1;
   int card1_ = -1;
   int card2_ = -1;

   const CommOptions options_;
   int numGameFinished_ = -1;
};

class Communicate2Policy : public rela::OptimalStrategy {
 public:
   Communicate2Policy() {
   }

   // Given infoset, return policy.
   std::vector<float> getOptimalStrategy(const std::string &key) const override { 
     if (key == "s" || key.substr(0, 5) == "done-") return {};
     auto it = kOptimalPolicy.find(key);
     assert(it != kOptimalPolicy.end());
     return it->second;
   }
};

}  // namespace simple
