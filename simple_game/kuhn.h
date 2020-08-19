// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 
// 
#pragma once

#include <set>
#include <sstream>
#include <cassert>

#include "rela/env.h"
#include "utils.h"

namespace simple {

static const std::vector<std::pair<int, int>> kCardDeals = { 
  {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1} 
};

static const std::string kActionTypes = "cb";

static const std::set<std::string> kTerminalStates = {
  "rrcc", "rrcbc", "rrcbb", "rrbc", "rrbb"
};


class KuhnPoker : public rela::Env {
 public:
   KuhnPoker() {
     reset();
   }

   std::unique_ptr<rela::Env> clone() const override {
     return std::make_unique<KuhnPoker>(*this);
   }

   bool reset() override {
     public_ = "s";
     card1_ = -1;
     card2_ = -1;
     return true;
   }

   // Include chance.
   rela::EnvSpec spec() const override { 
     return { 
       9,  // feature size
       { (int)kCardDeals.size(),  2, 2 },
       { rela::PlayerGroup::GRP_NATURE, rela::PlayerGroup::GRP_1, rela::PlayerGroup::GRP_2 } 
     };
   }

   std::string info() const override {
     std::stringstream ss;
     ss << "History: " << public_ << ", card1: "<< card1_ << ", card2: " << card2_;
     return ss.str();
   }

   rela::TensorDict feature() const override {
     auto s = torch::zeros({3 + 3 + 3}, torch::kFloat32);
     if (public_ != "s") {
       auto accessor = s.accessor<float, 1>();
       if (isFirstPlayer()) {
         accessor[card1_] = 1.0f;
       } else {
         accessor[card2_ + 3] = 1.0f;
       }
       for (size_t i = 2; i < public_.size(); ++i) {
         accessor[6 + i - 2] = public_[i] == 'b' ? 1.0f : 0.0f;
       }
     }

     return { { "s", s } };
   }

   int maxNumAction() const override {
     return (int)std::max(kCardDeals.size(), kActionTypes.size());
   }

   std::vector<int> legalActions() const override {
     if (public_ == "s") return rela::utils::getIncSeq(kCardDeals.size());
     return rela::utils::getIncSeq(kActionTypes.size());
   }

   /*
   int numAction() const override {
     if (public_ == "s") return (int)kCardDeals.size();
     return (int)kActionTypes.size();
   }
   */

   std::string infoSet() const override {
     if (public_ == "s") return public_;
     if (isFirstPlayer()) return public_ + "-" + getCardStr(card1_);
     else return public_ + "-" + getCardStr(card2_);
   }

   int playerIdx() const override {
     if (public_ == "s") return 0;
     return isFirstPlayer() ? 1 : 2;
   }

   void step(int action) override {
     if (public_ == "s") {
       public_ = "rr";
       card1_ = kCardDeals[action].first;
       card2_ = kCardDeals[action].second;
     } else {
       public_ += kActionTypes[action];
     }
   }

   bool terminated() const override {
     auto it = kTerminalStates.find(public_);
     return it != kTerminalStates.end();
   }

   float playerReward(int pIdx) const override {
     if (! terminated() || pIdx == 0) return 0.0f;

     int cardP = card1_;
     int cardO = card2_;

     if (! isFirstPlayer()) std::swap(cardP, cardO);

     int curr_util = 0;

     if (public_ == "rrcbc" || public_ == "rrbc") { 
       // Last player folded. The current player wins.
       curr_util = 1;
     } else if (public_ == "rrcc") {
       // Showdown with no bets
       curr_util = cardP > cardO ? 1 : -1;
     } else {
       // Showdown with 1 bet
       assert(public_ == "rrcbb" || public_ == "rrbb");
       curr_util = cardP > cardO ? 2 : -2;
     }

     int currIdx = isFirstPlayer() ? 1 : 2;
     return pIdx == currIdx ? curr_util : -curr_util;
   } 
  
 private:
   std::string public_;
   int card1_ = -1;
   int card2_ = -1;

   static char getCardStr(int card) {
     switch (card) {
       case 0: return 'J';
       case 1: return 'Q';
       case 2: return 'K';
       default: return '?';
     }
   }

   bool isFirstPlayer() const {
     return public_.size() % 2 == 0;
   }
};

}  // namespace simple
