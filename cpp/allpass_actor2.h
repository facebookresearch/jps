#pragma once
#include "bridge_common.h"
#include "rela/actor2.h"
#include "rela/types.h"

class AllPassActor2 : public rela::Actor2 {
 public:
  AllPassActor2() {
  }

  int numAct() {
    return 0;
  }

  rela::TensorDictFuture act(rela::TensorDict& obs) override {
    (void)obs;

    return []() -> rela::TensorDict {
      auto a = torch::zeros({1}).to(torch::kInt64);
      auto f = a.accessor<long, 1>();
      f[0] = kSpecialBidStart;
      return {{"a", a}};
    };
  }
};
