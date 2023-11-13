#pragma once

#include "bridge_common.h"
#include "hand.h"
#include "rela/actor2.h"
#include "rela/model.h"
#include "rela/types.h"
#include "rela/utils.h"

class BaselineActor2 : public rela::Actor2 {
 public:
  BaselineActor2(std::shared_ptr<rela::Models> model)
      : model_(std::move(model)) {
  }

  int numAct() {
    return 0;
  }

  rela::TensorDictFuture act(rela::TensorDict& obs) override {
    // std::cout << obs["baseline_s"] << std::endl;
    // std::cout << obs["baseline_convert"] << std::endl;
    actionFuture_ = model_->call("act", obs);
    obs_ = obs;

    return [this]() { return this->reply(); };
  }

  std::string visualizeState() const {
    // std::cout << "baseline s: " << std::endl;
    // rela::utils::tensorDictPrint(obs_);

    auto s = rela::utils::get(obs_, "baseline_s");
    auto f = s.accessor<float, 1>();
    bridge::Hand hand;
    // hand.resize(kSuit, std::vector<unsigned int>(kCardsPerSuit));
    for (int i = 0; i < kSuit; i++) {
      for (int j = 0; j < kCardsPerSuit; j++) {
        if (f[i * kCardsPerSuit + j] > 0.0f) {
          hand.add({kSuit - 1 - i, kCardsPerSuit - 1 - j});
        }
      }
    }
    return hand.originalHandString();
  }

 private:
  std::shared_ptr<rela::Models> model_;
  rela::TensorDictFuture actionFuture_;
  rela::TensorDict obs_;

  rela::TensorDict reply() {
    auto action = actionFuture_();

    // std::cout << "got action" << std::endl;
    int a = -1;
    auto accessor = action["pi"].accessor<float, 1>();

    float minCost = 1;
    int bestAction = kSpecialBidStart;
    for (int j = accessor.size(0) - 1; j > 0; j--) {
      if (obs_["legal_move"][j - 1].item<float>() == 1.0) {
        if (accessor[j] < minCost) {
          minCost = accessor[j];
          bestAction = j - 1;
        }
      } else {
        break;
      }
    }

    if (minCost <= 0.2) {
      a = bestAction;
    } else {
      a = kSpecialBidStart;
    }
    if (accessor[0] < minCost) {
      a = kSpecialBidStart;
    }
    if (obs_["baseline_convert"].item<float>() == 1.0) {
      a = kSpecialBidStart;
    }

    /*
    Bid bid;
    bid.fromIdx(a);
    std::cout << "Input state: " << visualizeState() << ", Bid: " << bid.print()
    << std::endl;
    */

    // std::cout << a[0] << std::endl;
    action["a"] = torch::zeros({1}).to(torch::kInt64);
    action["a"][0] = a;
    return action;
  }
};
