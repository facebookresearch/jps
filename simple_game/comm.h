#pragma once

#include "rela/env.h"
#include "comm_options.h"
#include "utils.h"
#include <sstream>
#include <cassert>
#include <bitset>

namespace simple {

class Communicate : public rela::Env {
 public:
   Communicate(const CommOptions &commOptions) 
     : commOptions_(commOptions) {
       if (commOptions_.possibleCards < 0) {
         commOptions_.possibleCards = 1 << commOptions_.numRound;
       }
       assert(commOptions_.numRound >= 1);
       assert(commOptions_.possibleCards >= 2);
   }

   int numRound() const { return commOptions_.numRound; }

   bool reset() override {
     publicActions_.clear();
     card1_ = -1;
     finalBet_ = -1;
     numGameFinished_ ++;

     bool noMoreGames = commOptions_.seqEnumerate && (numGameFinished_ >= commOptions_.possibleCards);
     return ! noMoreGames;
   }

   std::string info() const override {
     std::stringstream ss;
     ss << "Public: " << printVector(publicActions_) << ", card1: " << card1_ 
        << " [" << std::bitset<8>(card1_) << "], finalbet: " << finalBet_ 
        << ", player: " << playerIdx() 
        << ", terminal: " << terminated() << ", infoSet: " << infoSet();
     return ss.str();
   }

   int maxNumAction() const override { 
     return commOptions_.possibleCards;
   }

   std::vector<int> legalActions() const override {
     if (terminated()) return {};
     if (playerIdx() == 1) return {0,1};
     return rela::utils::getIncSeq(commOptions_.possibleCards);
   }

   std::unique_ptr<rela::Env> clone() const override {
     return std::make_unique<Communicate>(*this);
   }

   // Observation TensorDict, reward, terminal 
   void step(int action) override {
     assert (! terminated());

     if (card1_ < 0) {
       // [TODO]: HACK here. 
       //   If seqEnumerate is true (e.g. in evaluation mode), 
       //   then the environment would alter the nature action to enumerate all possible 
       //        situations (different card1_) and see whether they works. 
       if (commOptions_.seqEnumerate) {
         card1_ = numGameFinished_; 
       } else { 
         card1_ = action;
       }
     } else if ((int)publicActions_.size() < commOptions_.numRound) {
       assert(action < 2);
       publicActions_.push_back(action);
     } else {
       finalBet_ = action;
     }
   }

   // Include chance.
   rela::EnvSpec spec() const override { 
     int n = commOptions_.possibleCards;
     return { 
       // featureSize (for player 1, see card (one hot) and history
       //            , for player 2, all entries in possibleCards are zero.
       commOptions_.possibleCards + commOptions_.numRound,
       // Nature max action, player 1 max action, player 2 max action.
       { n, 2, n }, 
       // Two player share the model.
       { rela::PlayerGroup::GRP_NATURE, rela::PlayerGroup::GRP_1,  rela::PlayerGroup::GRP_1 } 
     };
   }

   bool terminated() const override {
     return finalBet_ >= 0;
   }

   bool subgameEnd() const override {
     return terminated();
   }

   float playerReward(int idx) const override {
    if (! terminated() || idx == 0) return 0.0f;
    return finalBet_ == card1_ ? 1.0f : 0.0f;
   }

   float playerRawScore(int idx) const override { return playerReward(idx); }

   std::string infoSet() const override { 
     if (card1_ < 0) return "s";
     
     std::string s = "P" + std::to_string(playerIdx()) + "-";

     s += "r" + printVectorCompact(publicActions_);
     if (playerIdx() == 1) s +=   "-" + std::to_string(card1_);
     if (terminated()) s = "done-" + s + "-f" + std::to_string(finalBet_);

     return s;
   }

   std::string completeCompactDesc() const override { 
     if (card1_ < 0) return "s";

     std::string s = "P" + std::to_string(playerIdx()) + "-";

     s += "r" + printVectorCompact(publicActions_);
     s +=  "-" + std::to_string(card1_);
     if (terminated()) s = "done-" + s + "-f" + std::to_string(finalBet_);

     return s;
   }

   rela::TensorDict feature() const override {
     auto s = torch::zeros({ commOptions_.possibleCards + commOptions_.numRound });
     auto accessor = s.accessor<float, 1>();
     for (size_t i = 0; i < publicActions_.size(); ++i) {
       accessor[commOptions_.possibleCards + i] = publicActions_[i] == 0 ? -1.0f : 1.0f;
     }

     if (playerIdx() == 1) {
       accessor[card1_] = 1.0;
     } 

     auto tensorPlayerIdx = (playerIdx() == 1 ? torch::zeros({1}) : torch::ones({1}));

     return { { "s", s }, {"legal_move", legalActionMask() }, { "player_idx", tensorPlayerIdx } };
   }

   int playerIdx() const override { 
     if (card1_ < 0) return 0;
     if ((int)publicActions_.size() == commOptions_.numRound) return 2;
     return 1;
   }

   std::vector<int> partnerIndices(int playerIdx) const override {
     if (playerIdx == 0) return {};
     return { 3 - playerIdx };
   }

 private:
   std::vector<int> publicActions_;
   int card1_;
   int finalBet_;

   CommOptions commOptions_;
   int numGameFinished_ = -1;
};

class CommunicatePolicy : public rela::OptimalStrategy {
 public:
   CommunicatePolicy(const CommOptions &commOptions) 
     : commOptions_(commOptions) {
       if (commOptions_.possibleCards < 0) {
         commOptions_.possibleCards = 1 << commOptions_.numRound;
       }
   }

   // Given infoset, return policy.
   std::vector<float> getOptimalStrategy(const std::string &key) const override { 
     if (key == "s" || key.substr(0, 5) == "done-") return {};

     // Skip "P1-r" or "P2-r"; 
     int i = 4;
     std::vector<int> actions;
     while (i < (int)key.size()) {
       if (key[i] == '-') break; 
       else {
         actions.push_back(key[i] - '0');
         i ++;
       } 
     }

     if (i == (int)key.size()) {
       // Player 2
       assert((int)actions.size() == commOptions_.numRound);
       int n = 0;
       for (int i = commOptions_.numRound - 1; i >= 0; --i) {
         // Little Endian
         n = ((n << 1) | actions[i]);
       }
       std::vector<float> policy(commOptions_.possibleCards, 0.0f);

       if (n < (int)policy.size()) {
         policy[n] = 1.0;
       } else {
         // Reachability should be zero here. 
         // But since the policy of any infoSet is requested, we need to return something here.
         std::fill(policy.begin(), policy.end(), 1.0f / commOptions_.possibleCards);
       }
       return policy;
     } else {
       // Player 1. Binary encoding.
       int n = std::stoi(key.substr(i + 1, -1));
       std::vector<int> action_seq;
       for (int k = 0; k < commOptions_.numRound; ++k) {
         // Little Endian. 8 = 1000 => Action sequence [0, 0, 0, 1].
         action_seq.push_back(n % 2);
         n /= 2;
       }

       if (action_seq[actions.size()] == 0) return {1.0, 0.0};
       else return {0.0, 1.0};
     }
   }

 private:
   CommOptions commOptions_;
};

} // namespace simple
