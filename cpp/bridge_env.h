#pragma once

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "action.h"
#include "console_messenger.h"
#include "game_option.h"
#include "game_state.h"
#include "rela/env.h"

#include "util_func.h"

namespace bridge {

class BridgeEnv : public rela::Env {
 public:
  BridgeEnv() = default;

  BridgeEnv(const BridgeEnv& env)
      : threadIdx_(env.threadIdx_),
        currentIdx_(env.currentIdx_),
        database_(env.database_),
        handle_(env.handle_),
        option_(env.option_),
        state_(env.state_),
        terminated_(env.terminated_),
        tableOver_(env.tableOver_),
        numStep_(env.numStep_),
        rewards_(env.rewards_),
        rng_(env.rng_) {}

  BridgeEnv(std::shared_ptr<DBInterface> database,
            const std::unordered_map<std::string, std::string>& gameParams,
            bool verbose)
      : database_(database),
        state_(option_.tables),
        numStep_(0),
        verbose_(verbose),
        consoleMessenger_(nullptr) {
    if (verbose_) {
      std::cout << "Bridge game created, with parameters:\n";
      for (const auto& item : gameParams) {
        std::cout << "  " << item.first << "=" << item.second << "\n";
      }
      option_.verbose = verbose;
    }

    extractParams(gameParams, "thread_idx", &threadIdx_, true);
    // rng_.seed(threadIdx_);

    int seed;
    if (extractParams(gameParams, "seed", &seed, false)) {
      rng_.seed(seed);
    }

    extractParams(gameParams, "feature_version", &option_.featureVer, true);
    extractParams(gameParams, "sampler", &option_.sampler, false);
    extractParams(gameParams, "save_feature_history",
                  &option_.saveFeatureHistory, false);

    extractParams(gameParams, "fixed_vul", &option_.fixedVul, false);
    extractParams(gameParams, "fixed_dealer", &option_.fixedDealer, false);

    bool console_eval = false;
    extractParams(gameParams, "console_eval", &console_eval, false);
    if (console_eval) {
      consoleMessenger_ = ConsoleMessenger::get_messenger();
    }

    // Feature size is the max of the two. 
    if (option_.featureVer == "single") {
      featureSize_ = FeatureExtractor::featureDim();
    } else if (option_.featureVer == "old") {
      featureSize_ = FeatureExtractorOld::featureDim();
    } else if (option_.featureVer == "single/old" || option_.featureVer == "old/single") {
      featureSize_ = std::max(FeatureExtractor::featureDim(), FeatureExtractorOld::featureDim());
    } else {
      throw std::runtime_error("Feature type invalid: " + option_.featureVer);
    }
    init();
  }
  
  std::string getFeatureType(int seat, int tableIdx) const {
    if (option_.featureVer == "single" || option_.featureVer == "old") {
      return option_.featureVer;
    } else if (option_.featureVer == "single/old") {
      return (seat + tableIdx) % 2 == 0 ? "single" : "old";
    } else if (option_.featureVer == "old/single") {
      return (seat + tableIdx) % 2 == 0 ? "old" : "single";
    } else {
      throw std::runtime_error("Feature version invalid: " +
                               option_.featureVer);
    }
  }

  torch::Tensor getFeature(int seat, int tableIdx) const {
    std::string featureType = getFeatureType(tableIdx, seat);
    torch::Tensor s = torch::zeros({featureSize_});

    if (featureType == "single") {
      state_.computeFeature2(seat, tableIdx, s);
    } else if (featureType == "old") {
      // Old feature.
      state_.computeFeatureOld(seat, tableIdx, s);
    } else {
      throw std::runtime_error("Feature type invalid: " + featureType);
    }

    return s;
  }

  int maxNumAction() const override { return kAction; }

  rela::EnvSpec spec() const override {
    return {featureSize_,
            -1,
            {kAction, kAction, kAction, kAction},
            {rela::PlayerGroup::GRP_1, rela::PlayerGroup::GRP_2,
             rela::PlayerGroup::GRP_1, rela::PlayerGroup::GRP_2}};
  }

  bool reset() final;

  int playerIdx() const override { return state_.playerIdx(); }

  float playerReward(int playerIdx) const override {
    if (playerIdx % 2 == 0)
      return state_.getReward();
    else
      return -1 * state_.getReward();
  }

  float playerRawScore(int playerIdx) const override {
    assert(tableOver_);
    return state_.playerRawScore(playerIdx);
  }

  virtual std::vector<int> partnerIndices(int playerIdx) const override {
    return {(playerIdx + 2) % 4};
  }

  // Non-virtual functions.
  std::string printHandAndBidding() const {
    return state_.printHandAndBidding();
  }

  //
  void resetTo(int recordIdx, int dealer = -1, int vul = -1);

  //
  GameState getState() const { return state_; }

  const GameOption& getOption() { return option_; }

  void init();

  void biddingStep(int actionIdx);

  void playingStep(int actionIdx) { state_.playingStep(actionIdx); }

  // return {'obs', 'reward', 'terminal'}
  // action_p0 is a tensor of size 1, representing uid of move
  virtual void step(int action) final {
    biddingStep(action);

    tableOver_ = false;
    if (state_.biddingTerminal()) {
      tableOver_ = true;
      terminated_ = state_.nextTable();

      if (terminated_) {
        rewards_.push_back(state_.getReward());
      }
    }

    if (consoleMessenger_) {
      std::ostringstream ss;
      ss << "json_str: " << jsonObj().dump();
      consoleMessenger_->send_env_info(threadIdx_, ss.str());
      if (tableOver_) {
         if (terminated_) {
          consoleMessenger_->send_env_msg(threadIdx_, "end");
          const std::vector<std::string>& msg =
              ConsoleMessenger::get_messenger()->read_env_msg(threadIdx_);
          if (msg.size() < 2 || msg[1] != "ended") {
            std::cout << "not received \"ended\" msg from console!";
          }
        } else {
          consoleMessenger_->send_env_msg(threadIdx_, "subgame_end");
          const std::vector<std::string>& msg =
              ConsoleMessenger::get_messenger()->read_env_msg(threadIdx_);
          if (msg.size() < 2 || msg[1] != "subgame_ended") {
            std::cout << "not received \"subgame_ended\" msg from console!";
          }
        }
      }
    }
  }

  void setReply(const rela::TensorDict& reply) override {
    // Set prob distribution.
    if (!rela::utils::hasKey(reply, "pi")) return;

    const auto vecPi = rela::utils::getSortedProb(reply, "pi", 1e-4);
    std::vector<Bid> otherChoices;

    for (const auto& b : vecPi) {
      // Bid bid(b.second);
      // bid.fromIdx(b.second);
      // bid.prob = b.first;

      // otherChoices.push_back(bid);
      otherChoices.emplace_back(b.second);
    }

    // Save this to the record.
    state_.setBidOtherChoices(std::move(otherChoices));
  }

  rela::TensorDict feature() const override {
    int currSeat = state_.getCurrentSeat();
    int currTable = state_.getCurrentTableIdx();
    return featureWithTableSeat(currTable, currSeat, false);
  }

  bool hasOpeningLead(int tableIdx) const {
    return state_.getDeclarer(tableIdx) >= 0;
  }

  rela::TensorDict featureOpeningLead(int tableIdx, bool gtCards,
                                      bool debugInfo) const {
    int declarer = state_.getDeclarer(tableIdx);
    int strain = state_.getContractStrain(tableIdx);

    int seat = (declarer + 1) % kPlayer;

    torch::Tensor s = getFeature(seat, tableIdx);

    rela::TensorDict dict = {
        {"s", s},
    };

    if (debugInfo) {
      rela::TensorDict moreFeature = {
          {"idx", torch::zeros({1}, torch::kLong).fill_(getIdx())},
          {"tbl", torch::zeros({1}, torch::kLong).fill_(tableIdx)},
          {"dealer", torch::zeros({1}, torch::kLong).fill_(state_.getDealer())},
          {"vul", torch::zeros({1}, torch::kLong).fill_(state_.getVul())},
          {"bid", state_.computeAbsAuctionSeqInfo()},
          {"seat", torch::zeros({1}, torch::kLong).fill_(seat)},
          {"strain", torch::zeros({1}, torch::kLong).fill_(strain)},
      };
      rela::utils::appendTensorDict(dict, moreFeature);
    }

    if (curr_.find("fut") != curr_.end()) {
      torch::Tensor fut = torch::zeros({kDeck}, torch::kLong);
      fut.fill_(-1);
      auto accessor = fut.accessor<long, 1>();
      // Query to get fut_tricks.
      for (int i = 0; i < kDeck; ++i) {
        if (curr_["card_map"][i] == seat) {
          accessor[i] = curr_["fut"][strain][i];
        }
      }
      dict["fut"] = fut;
    }

    if (gtCards) {
      rela::TensorDict moreFeature = {
          {"cards", state_.getCompleteInfoEncoding(seat)},
      };
      rela::utils::appendTensorDict(dict, moreFeature);
    }

    return dict;
  }

  rela::TensorDict featureWithTableSeat(int tableIdx, int seat,
                                        bool gtCards) const {
    torch::Tensor s;
    torch::Tensor legalMove;
    rela::TensorDict baselineS;
    rela::TensorDict partnerInfo;

    s = getFeature(seat, tableIdx);

    legalMove = state_.computeLegalMove2(seat, tableIdx);
    baselineS = state_.computeBaselineFeature2(seat, tableIdx);

    partnerInfo = state_.computePartnerInfo(seat);

    int strain = state_.getContractStrain(tableIdx);
    rela::TensorDict dict = {{"s", s},
                             {"baseline_s", baselineS["s"]},
                             {"baseline_convert", baselineS["convert"]},
                             {"legal_move", legalMove},
                             {"strain", torch::tensor({static_cast<int64_t>(strain)}, torch::kInt64)}};

    rela::utils::appendTensorDict(dict, partnerInfo);

    if (gtCards) {
      rela::TensorDict moreFeature = {
          // {"s_complete", state_.getCompleteInfo() },
          // {"seat", torch::zeros({1}, torch::kLong).fill_(seat) },
          {"cards", state_.getCompleteInfoEncoding(seat)}};
      rela::utils::appendTensorDict(dict, moreFeature);
    }

    // for (const auto& name2sth : dict) {
    //   std::cout << name2sth.first << ": " << name2sth.second.sizes() <<
    //   std::endl;
    // }

    if (option_.saveFeatureHistory) {
      // Save one copy of feature.
      featureHistory_.push_back(dict);
    }

    return dict;
  }

  rela::TensorDict getPlayingFeature(int tableIdx, int seat,
                                     bool gtCards) const;

  bool terminated() const final { return terminated_; }

  bool subgameEnd() const final { return tableOver_; }

  std::string info() const override;
  json jsonObj() const override;

  std::unique_ptr<Env> clone() const override {
    BridgeEnv cloneEnv(*this);
    auto ptr = std::make_unique<BridgeEnv>(std::move(cloneEnv));
    return ptr;
  }

  // Non-virtual
  float getEpisodeReward() const { return state_.getReward(); }

  std::vector<float> getRewards() const { return rewards_; }

  int getDatasetSize() const { return handle_->size; }

  json curr() const { return curr_; }

  std::string currJsonStr() const { return curr_.dump(); }

  int getIdx() const { return handle_->offset + currentIdx_; }

  int trick2Take(int tableIdx) const { return state_.trick2Take(tableIdx); }

 private:
  int threadIdx_ = 0;
  int currentIdx_ = 0;

  std::shared_ptr<DBInterface> database_;
  std::shared_ptr<DBInterface::Handle> handle_;

  GameOption option_;
  GameState state_;

  bool terminated_ = true;
  bool tableOver_ = true;

  int numStep_;
  int featureSize_;

  mutable std::vector<rela::TensorDict> featureHistory_;
  std::vector<float> rewards_;

  json curr_;

  std::mt19937 rng_;

  const bool verbose_ = false;

  std::shared_ptr<ConsoleMessenger> consoleMessenger_;
};

}  // namespace bridge
