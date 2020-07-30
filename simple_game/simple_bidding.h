#pragma once

#include "comm_options.h"
#include "rela/env.h"
#include "utils.h"
#include <bitset>
#include <cassert>
#include <sstream>

namespace simple {

class SimpleBidding : public rela::Env {
 public:
  static constexpr int kPass = 0;

  SimpleBidding(const CommOptions& options)
      : commOptions_(options) {
    // possible #actions.
    int n = commOptions_.N - 1;
    numBidding_ = 0;
    while (n >= 1) {
      n >>= 1;
      numBidding_++;
    }
    // Here we got maximal numBidding_ so that pow(2, numBidding_) <= 2 * (N - 1)
    // From 0 to numBidding_ there are numBidding_ + 1 actions, plus Pass (0)
    numBidding_ += 2;

    // Nature has N^2 actions (pick 0 to N - 1 for player 1 and 2).
    commOptions_.possibleCards = commOptions_.N * commOptions_.N;
  }

  bool reset() override {
    publicActions_.clear();
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
       << ", terminal: " << terminated()
       << ", infoSet: " << infoSet();
    return ss.str();
  }

  int maxNumAction() const override {
    // Just return total possible actions.
    return std::max(commOptions_.N * commOptions_.N, numBidding_);
  }

  std::vector<int> legalActions() const override {
    // Get legal actions for that particular state.
    std::vector<int> legals;
    if (terminated()) return legals;
    
    if (card1_ < 0) {
      return rela::utils::getIncSeq(commOptions_.N * commOptions_.N); 
    } else {
      if (publicActions_.size() == 0) {
        // Pass is not allowed.
        legals.resize(numBidding_ - 1);
        std::iota(legals.begin(), legals.end(), 1);
      } else {
        legals.resize(numBidding_ - lastBid_);
        // Pass is always legal except at the beginning of the game.
        legals[0] = kPass;
        std::iota(legals.begin() + 1, legals.end(), lastBid_ + 1);
      }
    }
    return legals;
  }

  std::unique_ptr<rela::Env> clone() const override {
    return std::make_unique<SimpleBidding>(*this);
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
      // 0 .. N - 1
      card1_ = action % commOptions_.N;
      card2_ = action / commOptions_.N;
      return;
    }

    bool illegal = false;
    if (publicActions_.empty() && action == kPass) {
      illegal = true;
    }

    // Check if bidding is valid.
    if (publicActions_.size() >= 1) {
      if (action > kPass && lastBid_ >= action) {
        illegal = true;
      } else if (action == kPass) {
        // 1 pass and we end the game.
        terminated_ = true;
        if (lastBid_ == kPass) {
          // no contract.
          reward_ = 0;
        } else {
          int finalContract = 1 << (lastBid_ - 1);
          if (finalContract <= card1_ + card2_) {
            reward_ = finalContract;
          } else {
            // Failed the contract.
            reward_ = 0;
          }
        }
      }
    }

    if (illegal) {
        // illegal action.
        throw std::runtime_error("action " + std::to_string(action) +
                                 " is not a legal action!");
    }

    publicActions_.push_back(action);
    if (action != kPass) {
      lastBid_ = action;
    }
  }

  // Include chance.
  rela::EnvSpec spec() const override {
    int n = commOptions_.possibleCards;
    return {// featureSize (for player 1/2, see card (one hot), so 2*N entries,
            //              plus numBidding_ * (numBidding_ + 1) for public
            //              bidding actions)
            2 * commOptions_.N + numBidding_ * (numBidding_ + 1),
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
    auto s = torch::zeros({2 * commOptions_.N + numBidding_ * (numBidding_ + 1)});
        
    auto accessor = s.accessor<float, 1>(); 
    if (playerIdx() == 1) {
      accessor[card1_] = 1.0f;
    } else if (playerIdx() == 2) {
      accessor[commOptions_.N + card2_] = 1.0f;
    }

    // public information. 
    for (size_t i = 0; i < publicActions_.size(); ++i) { 
      accessor[2 * commOptions_.N + numBidding_ * i + publicActions_[i]] = 1.0f;
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
  bool terminated_;

  int numBidding_;

  CommOptions commOptions_;
  int numGameFinished_ = -1;
};

}  // namespace simple
