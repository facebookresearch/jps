#pragma once

#include "rela/a2c_actor.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/env_actor_base.h"

#include "cpp/bridge_env.h"
#include "cpp/bid.h"

class Pointer {
 public:
  Pointer(int sz, int span) : sz_(sz), span_(span) {
    arranges_.clear();
    for (int i = 0; i < span_; ++i) {
      for (int j = 0; j < span_; ++j) {
        if (i == j) continue;
        arranges_.push_back(std::make_pair(i, j));
      }
    }
  }

  void reset() {
    currIdx_ = 0;
    currArrangeIdx_ = -1;
  }

  bool goNext(int *idx1, int *idx2) {
    currArrangeIdx_ ++;
    
    if (currArrangeIdx_ == (int)arranges_.size()) {
      currIdx_ += span_;
      currArrangeIdx_ = 0;
    }

    if (currIdx_ >= sz_) return false;

    // Otherwise we reset both environments. 
    auto p = arranges_[currArrangeIdx_];
    *idx1 = currIdx_ + p.first;
    *idx2 = currIdx_ + p.second;
    return true;
  }

 private:
  const int sz_;
  const int span_;

  int currIdx_ = 0;
  int currArrangeIdx_ = -1;
  std::vector<std::pair<int, int>> arranges_;
};


class CrossBiddingEnvActor : public EnvActorBase {
 public:
  CrossBiddingEnvActor(std::shared_ptr<bridge::BridgeEnv> env,
           std::vector<std::shared_ptr<rela::Actor2>> actors, 
           const EnvActorOptions &options, 
           std::shared_ptr<rela::PrioritizedReplay2> replayBuffer)
      : EnvActorBase(actors, options)
      , env_(std::move(env))
      , replayBuffer_(replayBuffer) { 
    checkValid(*env_);
    env_->reset();
    envGt_ = std::make_unique<bridge::BridgeEnv>(*env_);

    pointer_ = std::make_unique<Pointer>(env_->getDatasetSize(), kSpan);
    goNext();

    probs_.resize(2);
  }

  void preAct() override {
    assert (!isTerminated());
    auto playerIdx = env_->playerIdx();

    // Get feature. 
    auto obs = env_->feature();

    // std::cout << "EnvActor: Before preAct " << std::endl;
    actFuture_ = actors_[playerIdx]->act(obs);
    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() override {
    assert (!isTerminated());
    // auto playerIdx = env_->playerIdx();
    rela::TensorDict reply = actFuture_();
    //
    // Replay the bidding sequence from envGt_ on current env_
    // Once it is finished, check the probability difference
    // If prob_at_env >= prob_at_env_gt / 10 (infrequent), and dds score is different, 
    // Then we want to sample such situations (since they are likely to be important). 
    //
    // Instead of using action given by the current env, use action from envGt_.
    // auto action = rela::utils::getTensorDictScalar<long>(reply, "a");
    const auto &gt = envGt_->curr();

    std::string bidStr = gt["bidd"][subgameIdx_]["seq"][biddingIdx_];
    if (bidStr[0] == '(') bidStr = bidStr.substr(1, bidStr.size() - 2);

    // bid to int
    // bid.fromStr(bidStr);
    bridge::Bid bid(bidStr);
    int action = bid.index();

    env_->step(action);
    biddingIdx_ ++;

    // Check the probability of that action.
    auto pi = rela::utils::get(reply, "pi");
    auto accessor = pi.accessor<float, 1>();
    probs_[subgameIdx_].push_back(accessor[action]);

    // Some env (e.g.. Bridge) will have multiple subgames in one game. 
    if (env_->subgameEnd()) {
      subgameIdx_ ++;
      biddingIdx_ = 0;
    }

    bool terminated = env_->terminated();
    if (! terminated) return;

    envTerminate();

    // Always North (seat = 0) for now. 
    int seat = 0;
    for (int tableIdx = 0; tableIdx < 2; ++tableIdx) {
      auto thisGtBid = gt["bidd"][tableIdx];

      double logProbGt = 0.0;
      for (const auto& p : thisGtBid["probs"]) {
        logProbGt += log(static_cast<double>(p));
      }
      double logProb = 0.0;
      for (const auto& p : probs_[tableIdx]) {
        logProb += log(static_cast<double>(p));
      }

      // We might check if logProb >= logProbGt - log(10), then we send data, otherwise skip. 
      // if (logProb - logProbGt < -log(10.0)) continue; 

      // Also compare DDS score given the contract determined by the bidding sequence.
      int trickEnv = env_->trick2Take(tableIdx); 
      int trickEnvGt = thisGtBid["trickTaken"];

      // if (trickEnv != trickEnvGt) {
      {
        auto f = env_->featureWithTableSeat(tableIdx, seat, true);
        f["log_prob"] = torch::zeros({1}).fill_(logProb);
        f["trick"] = torch::zeros({1}).fill_(trickEnv);
        
        auto f_gt = envGt_->featureWithTableSeat(tableIdx, seat, true);
        f_gt["log_prob"] = torch::zeros({1}).fill_(logProbGt);
        f_gt["trick"] = torch::zeros({1}).fill_(trickEnvGt);

        const int64_t feature_size = f["s"].numel();
        const float* fs_ptr = f["s"].data_ptr<float>();
        float* fs_gt_ptr = f_gt["s"].data_ptr<float>();
        // [TODO] Some hack here. Xiaomeng will fix it later. 
        std::memcpy(fs_gt_ptr, fs_ptr, feature_size * sizeof(float));
        f_gt["strain"] = f["strain"];
        // Note that since env_ and envGt_ has the same bidding sequence and private cards, 
        // the observable feature extracted from either env will be the same (except for hands of other players).
        // rela::Transition transition(f);
        // replayBuffer_->add(transition, 1.0f);
        const int strain = static_cast<int>(f["strain"].item<float>());
        if (strain >= 0 && strain <= 4) {
          rela::Transition transition_gt(f_gt);
          replayBuffer_->add(transition_gt, 1.0f);
        }
      }
    }

    subgameIdx_ = 0;
    biddingIdx_ = 0;
    for (auto &p : probs_) {
      p.clear();
    }

    // Reset the environment.
    if (!goNext()) {
      if (options_.eval) {
        terminateEnvActor();
      } else {
        // For training we just restart. 
        pointer_->reset();
        goNext();
      }
    }
  }

 protected:
  std::string getSaveData() const override {
    // Always North for now. 
    // Suppose env_ encodes complete information state s, and envGt_ encodes s_gt 
    // Seeing the private hand h and the bidding sequence P(b|s_gt) from envGt_, ideally we should guess other private hands as s_gt.
    // However, if P(b|s) is high, then s can also be a good cadidate for other player's private hands. 
    // If (1) given the contract from b, |DDT(s, b) - DDT(s_gt, b)| is large. 
    //    (2) P(b|s) is also quite high. 
    // Then we need to pay special attention to s, and make sure that our sampling model P(.|b,h) can sample s (in addition to s_gt). 
    /*
    std::cout << " ==== Alt ==== " << std::endl;
    std::cout << env_->printHandAndBidding() << std::endl;
    std::cout << env_->curr()["ddt"] << std::endl;

    std::cout << " ==== Gt ==== " << std::endl;
    std::cout << envGt_->printHandAndBidding() << std::endl;
    std::cout << envGt_->curr()["bidd"][0]["seq"] << std::endl;
    std::cout << envGt_->curr()["bidd"][1]["seq"] << std::endl;
    std::cout << envGt_->curr()["ddt"] << std::endl;
    */

    json j;
    j["pbn_alt"] = env_->curr()["pbn"];
    j["ddt_alt"] = env_->curr()["ddt"];
    j["data_idx_alt"] = env_->getIdx();

    auto gt = envGt_->curr();

    j["pbn"] = gt["pbn"];
    j["ddt"] = gt["ddt"];
    j["data_idx"] = envGt_->getIdx();

    // This also includes prob from gt situation.
    j["bidd"] = gt["bidd"];
    for (int i = 0; i < 2; ++i) {
      // Add prob from alternative situation s.
      j["bidd"][i]["prob_alt"] = probs_[i];
      j["bidd"][i]["trickTaken_alt"] = env_->trick2Take(i); 
    }

    return j.dump();
  }

 private:
  // One environment. Compared to envGt_, it has the same private hand, but different hands + differnt bidding seq. 
  std::shared_ptr<bridge::BridgeEnv> env_;

  // The "ground truth" environment. 
  std::unique_ptr<bridge::BridgeEnv> envGt_;

  std::shared_ptr<rela::PrioritizedReplay2> replayBuffer_;

  std::vector<std::vector<float>> probs_;
  int subgameIdx_ = 0;
  int biddingIdx_ = 0;

  rela::TensorDictFuture actFuture_ = nullptr;

  const int kSpan = 20;

  std::unique_ptr<Pointer> pointer_;
  bool goNext() {
    // Otherwise we reset both environments. 
    int idx1, idx2;
    if (!pointer_->goNext(&idx1, &idx2)) return false;

    env_->resetTo(idx1);
    envGt_->resetTo(idx2);
    return true;
  }
};
