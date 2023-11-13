#pragma once

#include "rela/actor.h"
#include "rela/dqn_actor.h"
#include <stdlib.h>

namespace rela {

class R2D2TransitionBuffer {
 public:
  R2D2TransitionBuffer(
      int batchsize, int numPlayer, int multiStep, int seqLen, int burnin)
      : batchsize(batchsize)
      , numPlayer(numPlayer)
      , multiStep(multiStep)
      , seqLen(seqLen)
      , batchNextIdx_(batchsize, 0)
      , batchH0_(batchsize)
      , batchSeqTransition_(batchsize, std::vector<FFTransition>(seqLen * 2))
      , batchSeqPriority_(batchsize, std::vector<float>(seqLen * 2))
      , batchLen_(batchsize, 0)
      , firstTerminals_(batchsize, 0)
      , terminalCount_(batchsize, 0)
      , canPop_(false) {
    assert(burnin == 0);
  }

  void push(const FFTransition& transition,
            const torch::Tensor& priority,
            const TensorDict& hid) {
    assert(priority.size(0) == batchsize);

    auto priorityAccessor = priority.accessor<float, 1>();

    for (int i = 0; i < batchsize; ++i) {
      int nextIdx = batchNextIdx_[i];
      assert(nextIdx < seqLen * 2 && nextIdx >= 0);
      if (nextIdx == 0) {
        batchH0_[i] = utils::tensorDictNarrow(
            hid, 1, i * numPlayer, numPlayer, false, true);
        // for (auto& kv : batchH0_[i]) {
        //   assert(kv.second.sum().item<float>() == 0);
        // }
      }

      auto t = transition.index(i);
      // some sanity check for termination
      if (nextIdx != 0) {
        // should not append after terminal
        // terminal should be processed when it is pushed
        assert(!batchSeqTransition_[i][nextIdx - 1].terminal.item<bool>() ||
               terminalCount_[i] == 1);
        assert(batchLen_[i] == 0);
      }

      batchSeqTransition_[i][nextIdx] = t;
      batchSeqPriority_[i][nextIdx] = priorityAccessor[i];

      ++batchNextIdx_[i];

      if (t.terminal.item<bool>()) {
        terminalCount_[i] += 1;
      }

      if (terminalCount_[i] < 2) {
        continue;
      }
      terminalCount_[i] = 0;

      // pad the rest of the seq in case of terminal
      batchLen_[i] = batchNextIdx_[i];
      /*
      while (batchNextIdx_[i] < seqLen) {
        batchSeqTransition_[i][batchNextIdx_[i]] = t.padLike();
        batchSeqPriority_[i][batchNextIdx_[i]] = 0;
        ++batchNextIdx_[i];
      }*/

      // refill reward

      for (int j = 0; j < batchLen_[i] - 1; j++) {
        if (batchSeqTransition_[i][j].terminal.item<bool>()) {
          firstTerminals_[i] = j;
          assert(j <= seqLen);
          for (int m = 0; m < multiStep; m++) {
            // copy over multistep reward
            auto r = batchSeqTransition_[i][batchLen_[i] - 1 - m].reward;
            batchSeqTransition_[i][j - m].reward = r.clone();
            // also fix priority target
            batchSeqPriority_[i][j - m] =
                batchSeqPriority_[i][j - m] + r.item<float>();
          }
          break;
        }
      }
      // now abs
      for (int j = 0; j < batchLen_[i]; j++) {
        batchSeqPriority_[i][j] = abs(batchSeqPriority_[i][j]);
      }
      /*
      std::cout << "*****" << std::endl;
      for (int j = 0; j < batchLen_[i] - 1; j++) {
        std::cout << batchSeqTransition_[i][j].reward.item<float>() << " ";
      }
      */
      canPop_ = true;
    }
  }

  bool canPop() {
    return canPop_;
  }

  std::tuple<std::vector<RNNTransition>, torch::Tensor, torch::Tensor>
  popTransition() {
    assert(canPop_);

    std::vector<RNNTransition> batchTransition;
    std::vector<torch::Tensor> batchSeqPriority;
    std::vector<float> batchLen;

    for (int i = 0; i < batchsize; ++i) {
      if (batchLen_[i] == 0) {
        continue;
      }
      // assert(batchNextIdx_[i] == seqLen);

      int len1 = firstTerminals_[i] + 1;
      batchLen.push_back(float(len1));

      std::vector<float> batchSeqPriority1(
          batchSeqPriority_[i].begin(), batchSeqPriority_[i].begin() + len1);
      while (batchSeqPriority1.size() < (size_t)seqLen) {
        batchSeqPriority1.push_back(0);
      }
      batchSeqPriority.push_back(torch::tensor(batchSeqPriority1));

      std::vector<FFTransition> batchSeqTransition1(
          batchSeqTransition_[i].begin(),
          batchSeqTransition_[i].begin() + len1);
      while (batchSeqTransition1.size() < (size_t)seqLen) {
        batchSeqTransition1.push_back(batchSeqTransition1[0].padLike());
      }
      auto t1 = RNNTransition(
          batchSeqTransition1, batchH0_[i], torch::tensor(float(len1)));
      batchTransition.push_back(t1);

      // for (int tmp = 0; tmp < len1; tmp++)
      //  t1.print(tmp);

      int len2 = batchLen_[i] - len1;

      batchLen.push_back(float(len2));
      std::vector<float> batchSeqPriority2(
          batchSeqPriority_[i].begin() + len1,
          batchSeqPriority_[i].begin() + len1 + len2);
      while (batchSeqPriority2.size() < (size_t)seqLen) {
        batchSeqPriority2.push_back(0);
      }
      batchSeqPriority.push_back(torch::tensor(batchSeqPriority2));

      std::vector<FFTransition> batchSeqTransition2(
          batchSeqTransition_[i].begin() + len1,
          batchSeqTransition_[i].begin() + len1 + len2);
      while (batchSeqTransition2.size() < (size_t)seqLen) {
        batchSeqTransition2.push_back(batchSeqTransition2[0].padLike());
      }

      auto t2 = RNNTransition(
          batchSeqTransition2, batchH0_[i], torch::tensor(float(len2)));
      batchTransition.push_back(t2);

      batchLen_[i] = 0;
      batchNextIdx_[i] = 0;
    }
    canPop_ = false;
    assert(batchTransition.size() > 0);

    return std::make_tuple(batchTransition,
                           torch::stack(batchSeqPriority, 0),
                           torch::tensor(batchLen));
  }

  const int batchsize;
  const int numPlayer;
  const int multiStep;
  const int seqLen;

 private:
  std::vector<int> batchNextIdx_;
  std::vector<TensorDict> batchH0_;

  std::vector<std::vector<FFTransition>> batchSeqTransition_;
  std::vector<std::vector<float>> batchSeqPriority_;
  std::vector<int> batchLen_;
  std::vector<int> firstTerminals_;
  std::vector<int> terminalCount_;

  bool canPop_;
};

class IndefiniteTransitionBuffer {
 public:
  IndefiniteTransitionBuffer() {
  }

  void clear() {
    obsHistory_.clear();
    actionHistory_.clear();
    rewardHistory_.clear();
    terminalHistory_.clear();
  }
  unsigned int getSize() {
    return obsHistory_.size();
  }

  void pushObsAndAction(TensorDict& obs, TensorDict& action) {
    obsHistory_.push_back(obs);
    actionHistory_.push_back(action);
  }

  void pushRewardAndTerminal(torch::Tensor& reward, torch::Tensor& terminal) {
    rewardHistory_.push_back(reward);
    terminalHistory_.push_back(terminal);
  }

  std::tuple<TensorDict, TensorDict, torch::Tensor, torch::Tensor> popFront() {
    assert(actionHistory_.size() == getSize());
    assert(rewardHistory_.size() == getSize());
    assert(terminalHistory_.size() == getSize());
    auto obs = obsHistory_.front();
    auto action = actionHistory_.front();
    auto reward = rewardHistory_.front();
    auto terminal = terminalHistory_.front();
    obsHistory_.pop_front();
    actionHistory_.pop_front();
    rewardHistory_.pop_front();
    terminalHistory_.pop_front();
    return std::make_tuple(obs, action, reward, terminal);
  }

 private:
  int size_;
  std::deque<TensorDict> obsHistory_;
  std::deque<TensorDict> actionHistory_;
  std::deque<torch::Tensor> rewardHistory_;
  std::deque<torch::Tensor> terminalHistory_;
};

class R2D2Actor : public Actor {
 public:
  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker,
            int multiStep,
            int batchsize,
            float gamma,
            int seqLen,
            float greedyEps,
            int numPlayer,
            std::shared_ptr<RNNPrioritizedReplay> replayBuffer)
      : modelLocker_(std::move(modelLocker))
      , greedyEps_(greedyEps)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(batchsize, numPlayer, multiStep, seqLen, 0)
      , multiStepBuffer_(multiStep, batchsize, gamma)
      , replayBuffer_(std::move(replayBuffer))
      , hidden_(getH0(batchsize * numPlayer))
      , batchsize_(batchsize)
      , numAct_(0) {
  }

  R2D2Actor(std::shared_ptr<ModelLocker> modelLocker,
            int numPlayer,
            float greedyEps)
      : modelLocker_(std::move(modelLocker))
      , greedyEps_(greedyEps)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(1, numPlayer, 1, 1, 0)  // never be used in this mode
      , multiStepBuffer_(1, 1, 1)           // never be used in this mode
      , replayBuffer_(nullptr)
      , hidden_(getH0(1 * numPlayer))
      , batchsize_(1)
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  virtual TensorDict act(TensorDict& obs) override {
    torch::NoGradGuard ng;
    assert(!hidden_.empty());

    if (replayBuffer_ != nullptr) {
      historyHidden_.push_back(hidden_);
    }

    auto eps = torch::zeros({batchsize_}, torch::kFloat32);
    eps.fill_(greedyEps_);
    obs["eps"] = eps;

    TorchJitInput input;
    auto jitObs = utils::tensorDictToTorchDict(obs, modelLocker_->device());
    auto jitHid = utils::tensorDictToTorchDict(hidden_, modelLocker_->device());
    input.push_back(jitObs);
    input.push_back(jitHid);

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("act")(input).toTuple()->elements();
    modelLocker_->releaseModel(id);

    auto action = utils::iValueToTensorDict(output[0], torch::kCPU, true);
    hidden_ = utils::iValueToTensorDict(output[1], torch::kCPU, true);

    if (replayBuffer_ != nullptr) {
      multiStepBuffer_.pushObsAndAction(obs, action);
    }

    numAct_ += batchsize_;
    return action;
  }

  // r is float32 tensor, t is byte tensor
  virtual void setRewardAndTerminal(torch::Tensor& r,
                                    torch::Tensor& t) override {
    if (replayBuffer_ == nullptr) {
      return;
    }
    // assert(replayBuffer_ != nullptr);
    multiStepBuffer_.pushRewardAndTerminal(r, t);

    // if ith state is terminal, reset hidden states
    // h0: [num_layers * num_directions, batch, hidden_size]
    TensorDict h0 = getH0(1);
    auto terminal = t.accessor<bool, 1>();
    // std::cout << "terminal size: " << t.sizes() << std::endl;
    // std::cout << "hid size: " << hidden_["h0"].sizes() << std::endl;
    for (int i = 0; i < terminal.size(0); i++) {
      if (terminal[i]) {
        for (auto& name2tensor : hidden_) {
          // batch dim is 1
          name2tensor.second.narrow(1, i * numPlayer_, numPlayer_) =
              h0.at(name2tensor.first);
        }
      }
    }
  }

  // should be called after setRewardAndTerminal
  // Pops a batch of transitions and inserts it into the replay buffer
  virtual void postStep() override {
    if (replayBuffer_ == nullptr) {
      return;
    }
    // assert(replayBuffer_ != nullptr);
    // assert(multiStepBuffer_.size() == historyHidden_.size());

    if (!multiStepBuffer_.canPop()) {
      assert(!r2d2Buffer_.canPop());
      return;
    }

    {
      FFTransition transition = multiStepBuffer_.popTransition();
      TensorDict hid = historyHidden_.front();
      TensorDict nextHid = historyHidden_.back();
      historyHidden_.pop_front();

      torch::Tensor priority = computePriority(transition, hid, nextHid);
      r2d2Buffer_.push(transition, priority, hid);
    }

    if (!r2d2Buffer_.canPop()) {
      return;
    }

    std::vector<RNNTransition> batch;
    torch::Tensor batchSeqPriority;
    torch::Tensor batchLen;

    std::tie(batch, batchSeqPriority, batchLen) = r2d2Buffer_.popTransition();
    auto priority = aggregatePriority(batchSeqPriority, batchLen);
    replayBuffer_->add(batch, priority);
  }

 private:
  TensorDict getH0(int batchsize) {
    TorchJitInput input;
    input.push_back(batchsize);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto output = model.get_method("get_h0")(input);
    modelLocker_->releaseModel(id);
    return utils::iValueToTensorDict(output, torch::kCPU, true);
  }

  torch::Tensor computePriority(const FFTransition& transition,
                                TensorDict hid,
                                TensorDict nextHid) {
    torch::NoGradGuard ng;
    const auto& device = modelLocker_->device();
    auto input = transition.toJitInput(device);
    input.push_back(utils::tensorDictToTorchDict(hid, device));
    input.push_back(utils::tensorDictToTorchDict(nextHid, device));

    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto priority = model.get_method("compute_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return priority.detach().to(torch::kCPU);
  }

  torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor len) {
    // priority: [batchsize, seqLen]
    TorchJitInput input;
    input.push_back(priority);
    input.push_back(len);
    int id = -1;
    auto model = modelLocker_->getModel(&id);
    auto aggPriority = model.get_method("aggregate_priority")(input).toTensor();
    modelLocker_->releaseModel(id);
    return aggPriority;
  }

  std::shared_ptr<ModelLocker> modelLocker_;
  const float greedyEps_;
  const int numPlayer_;

  std::deque<TensorDict> historyHidden_;
  R2D2TransitionBuffer r2d2Buffer_;
  MultiStepTransitionBuffer multiStepBuffer_;
  // IndefiniteTransitionBuffer indefiniteTransitionBuffer_;
  std::shared_ptr<RNNPrioritizedReplay> replayBuffer_;

  TensorDict hidden_;
  const int batchsize_;
  std::atomic<int> numAct_;
};
}
