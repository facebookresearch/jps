#pragma once
#include "bridge_common.h"
#include "rela/actor.h"
#include "rela/types.h"

class AllPassActor : public rela::Actor {
 public:
  AllPassActor() {
  }

  int numAct() {
    return 0;
  }

  virtual rela::TensorDict act(rela::TensorDict& obs) override {
    int batchsize = obs["s"].size(0);
    torch::Tensor a = torch::zeros({batchsize});
    auto f = a.accessor<float, 1>();
    for (int i = 0; i < batchsize; i++) {
      f[i] = kSpecialBidStart;
    }
    rela::TensorDict action = {{"a", a.to(torch::kInt64)}};
    return action;
  }

  virtual void setRewardAndTerminal(torch::Tensor&, torch::Tensor&) override {
  }

  virtual void postStep() override {
  }
};
