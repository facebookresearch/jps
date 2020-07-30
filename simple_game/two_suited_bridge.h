#pragma once

#include "comm_options.h"
#include "rela/env.h"
#include "utils.h"
#include <bitset>
#include <cassert>
#include <sstream>

namespace simple {

class TwoSuitedBridge : public rela::Env {
 public:
  static constexpr int kPass = 0;

  TwoSuitedBridge(const CommOptions& options)
      : commOptions_(options) {
    // possible #actions.
    int n = commOptions_.N + 1;

    // nH, nS, plus Pass (0)
    numBidding_ = commOptions_.N * 2 + 1;

    // Nature has N^2 actions (pick 0 to N for player 1 and 2).
    commOptions_.possibleCards = n * n;
  }

  bool reset() override {
    publicActions_.clear();

    // card here means number of hearts. number of spades is commOptions_.N - card
    card1_ = -1;
    card2_ = -1;
    lastBid_ = kPass;
    reward_ = 0;
    terminated_ = false;
    numGameFinished_++;

    bool noMoreGames = commOptions_.seqEnumerate &&
                       (numGameFinished_ >= commOptions_.possibleCards);
    return !noMoreGames;
  }

  std::string info() const override {
    std::stringstream ss;
    ss << "Public: " << printVector(publicActions_) << ", card1: " << card1_
       << " card2: " << card2_ << ", player: " << playerIdx()
       << ", #bid: " << numBidding_ 
       << ", N: " << commOptions_.N << ", terminal: " << terminated()
       << ", infoSet: " << infoSet();
    return ss.str();
  }

  int maxNumAction() const override {
    // Just return total possible actions.
    return std::max((commOptions_.N + 1) * (commOptions_.N + 1), numBidding_);
  }

  std::vector<int> legalActions() const override {
    // Get legal actions for that particular state.
    std::vector<int> legals;
    if (terminated()) return legals;
    
    if (card1_ < 0) {
      return rela::utils::getIncSeq((commOptions_.N + 1) * (commOptions_.N + 1)); 
    } else {
      legals.resize(numBidding_ - lastBid_);
      // Pass is always legal.
      legals[0] = kPass;
      std::iota(legals.begin() + 1, legals.end(), lastBid_ + 1);
    }
    return legals;
  }

  std::unique_ptr<rela::Env> clone() const override {
    return std::make_unique<TwoSuitedBridge>(*this);
  }

  // Observation TensorDict, reward, terminal
  void step(int action) override {
    // Any input action has to be legal action. 
    assert(!terminated());

    if (card1_ < 0) {
      // [TODO]: HACK here.
      //   If seqEnumerate is true (e.g. in evaluation mode),
      //   then the environment would alter the nature action to enumerate all
      //   possible
      //        situations (different card1_) and see whether they works.
      if (commOptions_.seqEnumerate) {
        action = numGameFinished_;
      }
      // 0 .. N
      card1_ = action % (commOptions_.N + 1);
      card2_ = action / (commOptions_.N + 1);
      return;
    }

    // Check if bidding is valid.
    if (publicActions_.size() >= 1) {
      if (action > kPass && lastBid_ >= action) {
        // illegal action.
        throw std::runtime_error("action " + std::to_string(action) +
                                 " is not a legal action!");
      } else if (action == kPass) {
        // 1 pass, not initial and we end the game.
        terminated_ = true;
        if (lastBid_ == kPass) {
          // no contract.
          reward_ = 0;
        } else {
          int finalContractLevel = (lastBid_ + 1) / 2;
          int finalContractSuit = lastBid_ % 2;
          int contractReward = 1;
          for (int i = 0; i < finalContractLevel - 1; i++) {
            contractReward <<= 1;
          }
          bool contractMake = false;
          if (((finalContractSuit == 1) && (commOptions_.N + finalContractLevel <= card1_ + card2_)) || 
              ((finalContractSuit == 0) && (commOptions_.N + finalContractLevel <= 2 * commOptions_.N - card1_ - card2_))) {
            contractMake = true;
          }
          if (contractMake) {
            reward_ = contractReward;
          } else {
            // Failed the contract.
            reward_ = -1;
          }
        }
      }
    }
    publicActions_.push_back(action);
    if (action != kPass) {
      lastBid_ = action;
    }
  }

  // Include chance.
  rela::EnvSpec spec() const override {
    int n = commOptions_.possibleCards;
    return {// featureSize (for player 1/2, see card (one hot), so 2*(N+1) entries,
            //              plus numBidding_ * (numBidding_ + 1) for public
            //              bidding actions)
            2 * (commOptions_.N + 1) + numBidding_ * (numBidding_ + 1),
            // Nature max action, player 1 max action, player 2 max action.
            {n, numBidding_, numBidding_},
            // Two player share the model.
            {rela::PlayerGroup::GRP_NATURE,
             rela::PlayerGroup::GRP_1,
             rela::PlayerGroup::GRP_1}};
  }

  bool terminated() const override {
    // check
    return terminated_;
  }

  bool subgameEnd() const override {
    return terminated();
  }

  float playerReward(int idx) const override {
    if (!terminated() || idx == 0)
      return 0.0f;
    return reward_;
  }

  float playerRawScore(int idx) const override {
    return playerReward(idx);
  }

  std::string infoSet() const override {
    if (card1_ < 0)
      return "s";

    std::string s = "P" + std::to_string(playerIdx()) + "-";
    if (playerIdx() == 1)
      s += "c1=" + std::to_string(card1_);
    else
      s += "c2=" + std::to_string(card2_);

    s += "-r" + printVectorCompact(publicActions_);
    if (terminated())
      s = "done-" + s;

    return s;
  }

  std::string completeCompactDesc() const override {
    if (card1_ < 0)
      return "s";

    std::string s = "P" + std::to_string(playerIdx()) + "-";
    s += "c1=" + std::to_string(card1_) + "-";
    s += "c2=" + std::to_string(card2_);

    s += "-r" + printVectorCompact(publicActions_);
    if (terminated())
      s = "done-" + s;

    return s;
  }

  rela::TensorDict feature() const override {
    auto s = torch::zeros({2 * (commOptions_.N + 1) + numBidding_ * (numBidding_ + 1)});
        
    auto accessor = s.accessor<float, 1>(); 
    if (playerIdx() == 1) {
      accessor[card1_] = 1.0f;
    } else if (playerIdx() == 2) {
      accessor[commOptions_.N + 1 + card2_] = 1.0f;
    }

    // public information. 
    for (size_t i = 0; i < publicActions_.size(); ++i) { 
      accessor[2 * (commOptions_.N + 1) + numBidding_ * i + publicActions_[i]] = 1.0f;
    }

    auto tensorPlayerIdx = (playerIdx() == 1 ? torch::zeros({1}) : torch::ones({1}));

    return { { "s", s }, {"legal_move", legalActionMask() }, { "player_idx", tensorPlayerIdx } };
  }

  int playerIdx() const override {
    if (card1_ < 0)
      return 0;
    return publicActions_.size() % 2 + 1;
  }

  std::vector<int> partnerIndices(int playerIdx) const override {
    if (playerIdx == 0)
      return {};
    return {3 - playerIdx};
  }

 private:
  std::vector<int> publicActions_;
  int card1_;
  int card2_;
  int lastBid_ = 0;
  int reward_ = 0;
  bool terminated_ = false;

  int numBidding_;

  CommOptions commOptions_;
  int numGameFinished_ = -1;
};

}  // namespace simple
