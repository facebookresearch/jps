#pragma once
#include "bridge_common.h"
#include "rela/actor.h"
#include "rela/types.h"
#include "rela/utils.h"
#include <torch/script.h>

class BaselineActor : public rela::Actor {
 public:
  BaselineActor(py::object pyModel, const std::string& device)
      : model_(pyModel.attr("_c").cast<rela::TorchJitModel*>())
      , device_(device) {
  }

  int numAct() {
    return 0;
  }

  virtual rela::TensorDict act(rela::TensorDict& obs) override {
    torch::NoGradGuard ng;
    rela::TorchJitInput input;
    auto jitObs = rela::utils::tensorDictToTorchDict(obs, device_);
    input.push_back(jitObs);

    auto output = model_->get_method("act")(input);
    // std::cout << "got output" << std::endl;
    auto action = rela::utils::iValueToTensorDict(output, torch::kCPU, true);
    // std::cout << "got action" << std::endl;
    // std::cout << obs["baseline_s"] << std::endl;
    // std::cout << obs["baseline_convert"] << std::endl;

    int batchsize = action["pi"].size(0);
    torch::Tensor a = torch::zeros({batchsize}).to(torch::kInt64);
    auto pi_accessor = action["pi"].accessor<float, 2>();

    for (int i = 0; i < batchsize; i++) {
      float minCost = 1;
      int bestAction = kSpecialBidStart;
      auto accessor = pi_accessor[i];
      for (int j = accessor.size(0) - 1; j > 0; j--) {
        if (obs["legal_move"][i][j - 1].item<float>() == 1.0) {
          if (accessor[j] < minCost) {
            minCost = accessor[j];
            bestAction = j - 1;
          }
        } else {
          break;
        }
      }
      if (minCost <= 0.2) {
        a[i] = bestAction;
      } else {
        a[i] = kSpecialBidStart;
      }
      if (accessor[0] < minCost) {
        a[i] = kSpecialBidStart;
      }
      if (obs["baseline_convert"][i].item<float>() == 1.0) {
        a[i] = kSpecialBidStart;
      }
    }
    // std::cout << a[0] << std::endl;
    action["a"] = a;

    return action;
  }

  virtual void setRewardAndTerminal(torch::Tensor&, torch::Tensor&) override {
  }

  virtual void postStep() override {
  }

 private:
  rela::TorchJitModel* model_;
  torch::Device device_;
};
