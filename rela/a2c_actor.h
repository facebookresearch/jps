// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <math.h>
#include <torch/script.h>

#include "rela/actor2.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/utils.h"

namespace rela {

class MultiStepTransitionBuffer2 {
 public:
  MultiStepTransitionBuffer2(int multiStep, float gamma, 
      bool calcCumulativeReward = false, bool getTrajNextObs = false)
      : multiStep_(multiStep)
      , gamma_(gamma)
      , calcCumulativeReward_(calcCumulativeReward)
      , getTrajNextObs_(getTrajNextObs) {
  }

  void push(TensorDict& dicts) {
    assert((int)history_.size() <= multiStep_);
    utils::assertKeyExists(dicts, {"reward", "terminal"});

    history_.push_back(dicts);
  }

  size_t size() {
    return history_.size();
  }

  bool canPop() {
    return (int)history_.size() == multiStep_ + 1;
  }

  /* assumes that:
   *  history contains content in t, t+1, ..., t+n
   *  each entry (e.g., "s") has size [1, customized dimension].
   * returns
   *  oldest entry with accumulated reward.
   *  e.g., "s": [multiStep, customized dimension].
   */
  Transition popTransition() {
    assert(canPop());

    TensorDict d = history_.front();

    if (calcCumulativeReward_) {
      // calculate cumulated rewards.
      // history_[multiStep_] is the most recent state.
      torch::Tensor reward = history_[multiStep_]["v"].clone();

      for (int step = multiStep_ - 1; step >= 0; step--) {
        // [TODO] suppose we shouldn't change h.
        auto& h = history_[step];
        const auto& r = h["reward"];

        if (h["terminal"].item<bool>()) {
          // Has terminal. so we reset the reward.
          reward = r;
        } else {
          reward = gamma_ * reward + r;
        }
      }

      d["R"] = reward;
    }

    if (getTrajNextObs_) {
      // This is the next obs after multiStep_ 
      d = utils::combineTensorDictArgs(d, history_.back());
    }

    history_.pop_front();
    return Transition(d);
  }

  void clear() {
    history_.clear();
  }

 private:
  const int multiStep_;
  const float gamma_;
  const bool calcCumulativeReward_;
  const bool getTrajNextObs_;

  std::deque<TensorDict> history_;
};

class A2CActor : public Actor2 {
 public:
  A2CActor(std::shared_ptr<Models> models,
           int multiStep,
           float gamma,
           std::shared_ptr<PrioritizedReplay2> replayBuffer)
      : models_(std::move(models))
      , transitionBuffer_(multiStep, gamma, true, false)
      , replayBuffer_(replayBuffer)
      , numAct_(0) {
  }

  A2CActor(std::shared_ptr<Models> models)
      : A2CActor(models, 1, 1.0, nullptr) {
  }

  int numAct() const {
    return numAct_;
  }

  TensorDictFuture act(TensorDict& obs) override {
    return models_->call("act", obs);
  }

  void sendExperience(TensorDict& d) override {
    if (replayBuffer_ == nullptr) {
      return;
    }

    transitionBuffer_.push(d);
    if (!transitionBuffer_.canPop()) {
      return;
    }

    auto transition = transitionBuffer_.popTransition();
    assert(!transition.empty());
    // utils::tensorDictPrint(transition.d);

    auto priorityFuture = models_->call("compute_priority", transition.d);
    addFuture([this, transition, priorityFuture]() {
      auto priority = priorityFuture();
      replayBuffer_->add(transition, priority["priority"].item<float>());
    });

    /*
    addFuture([this, transition]() {
      replayBuffer_->add(transition, torch::tensor(1.0f));
    });
    */
  }

 private:
  std::shared_ptr<Models> models_;
  MultiStepTransitionBuffer2 transitionBuffer_;
  std::shared_ptr<PrioritizedReplay2> replayBuffer_;
  std::atomic<int> numAct_;
};

}  // namespace rela
