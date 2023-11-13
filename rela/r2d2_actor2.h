// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include "rela/actor2.h"
#include "rela/utils.h"

#include "rela/a2c_actor.h"

#include "rela/model.h"
#include "rela/prioritized_replay2.h"

namespace rela {

class R2D2TransitionBuffer2 {
 public:
  R2D2TransitionBuffer2(int multiStep, int seqLen, int burnin)
      : multiStep_(multiStep)
      , seqLen_(seqLen)
      , burnin_(burnin) {
    assert(burnin_ <= seqLen_);
    assert(multiStep_ <= seqLen_);
  }

  void push(const TensorDict& t) {
    // Just keep pushing back until we reach seqLen_.
    if (seq_.empty()) {
      // Adding burnin frames.
      for (int i = 0; i < burnin_; ++i) {
        seq_.push_back(utils::tensorDictZerosLike(t));
      }
    }
    seq_.push_back(t);
    lastTerminal_ = utils::getTensorDictScalar<bool>(t, "terminal");
  }

  bool canPop() {
    return (int)seq_.size() >= seqLen_ || lastTerminal_;
  }

  std::tuple<Transition, int> popTransition() {
    assert(canPop());

    std::vector<TensorDict> res(seq_.begin(), seq_.end());

    TensorDict h0 = utils::splitTensorDict(res[0], '_');
    // Discard all hidden states but only keep the first one.
    for (size_t i = 1; i < res.size(); ++i) {
      utils::splitTensorDict(res[i], '_');
    }
    // Padding so that the length becomes seqLen_.
    int thisSeqLen = res.size();
    for (size_t i = 0; i < seqLen_ - res.size(); ++i) {
      res.push_back(utils::tensorDictZerosLike(res[0]));
    }

    if (lastTerminal_) {
      seq_.clear();
    } else {
      //  we pop until there are #burnin_ frames left
      while ((int)seq_.size() > burnin_) {
        seq_.pop_front();
      }
    }

    // Stack res together.
    TensorDict finalSeq = utils::vectorTensorDictJoin(res, 0);
    // Add starting hidden states.
    utils::appendTensorDict(finalSeq, h0);

    return std::tuple<Transition, int>(Transition(finalSeq), thisSeqLen);
  }

 private:
  const int multiStep_;
  const int seqLen_;
  const int burnin_;

  std::deque<TensorDict> seq_;
  bool lastTerminal_ = true;
};

class R2D2Actor2 : public Actor2 {
 public:
  R2D2Actor2(std::shared_ptr<Models> models,
             int multiStep,
             float gamma,
             int seqLen,
             int burnin,
             float epsGreedy,
             std::shared_ptr<PrioritizedReplay2> replayBuffer)
      : models_(std::move(models))
      , r2d2Buffer_(multiStep, seqLen, burnin)
      , multiStepBuffer_(multiStep, gamma, false, true)
      , replayBuffer_(std::move(replayBuffer))
      , epsGreedy_(epsGreedy)
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  TensorDictFuture act(TensorDict& obs) override {
    // Send the observation to the batcher and get the results.
    // std::cout << "About to send observation" << std::endl;
    // utils::tensorDictPrint(obs);
    // Add hidden state if we have.
    utils::appendTensorDict(obs, h_);
    auto eps = utils::setTensorDictScalar<float>("eps", epsGreedy_);
    utils::appendTensorDict(obs, eps);

    auto actionFuture = models_->call("act", obs);
    return [actionFuture, this]() {
      rela::TensorDict reply = actionFuture();
      // Keeping another copy of hidden state in h_.
      h_ = utils::splitTensorDictClone(reply, '_');
      return reply;
    };
  }

  void setTerminal() override {
    h_ = models_->callDirect("get_h0");
  }

  void sendExperience(TensorDict& d) override {
    if (replayBuffer_ == nullptr)
      return;

    multiStepBuffer_.push(d);
    if (!multiStepBuffer_.canPop()) {
      assert(!r2d2Buffer_.canPop());
      return;
    }

    Transition transition = multiStepBuffer_.popTransition();
    auto priorityFuture = models_->call("compute_priority", transition.d);

    addFuture([this, priorityFuture, &transition]() {
      transition.d["priority"] = priorityFuture()["priority"];
      r2d2Buffer_.push(transition.d);

      if (!r2d2Buffer_.canPop()) {
        return;
      }

      Transition seq;
      int seqLen;
      std::tie(seq, seqLen) = r2d2Buffer_.popTransition();
      TensorDict priority =
          utils::setTensorDictScalar<int64_t>("seq_len", seqLen);
      priority["priority"] = seq.d["priority"];
      auto aggPriorityFuture = models_->call("aggregate_priority", priority);

      addFuture([this, aggPriorityFuture, &seq]() {
        auto aggPriority = aggPriorityFuture();
        seq.d.erase("priority");
        replayBuffer_->add(seq, aggPriority["priority"].item<float>());
      });
    });
  }

 private:
  std::shared_ptr<Models> models_;

  R2D2TransitionBuffer2 r2d2Buffer_;
  MultiStepTransitionBuffer2 multiStepBuffer_;
  std::shared_ptr<PrioritizedReplay2> replayBuffer_;

  const float epsGreedy_;

  TensorDict h_;

  std::atomic<int> numAct_;
};
}
