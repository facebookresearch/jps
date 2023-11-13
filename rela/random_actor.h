#pragma once

#include "actor2.h"
#include "utils.h"
#include <random>

namespace rela {

class RandomActor : public Actor2 {
 public:
  RandomActor(int numAction, int seed)
      : rng_(seed)
  // , hist_(numAction, 0)
  {
    assert(numAction > 0);
    std::vector<float> weights(numAction, 1.0f / numAction);
    gen_ = std::discrete_distribution<>(weights.begin(), weights.end());
  }

  rela::TensorDictFuture act(TensorDict&) override {
    return [this]() {
      const int64_t action = gen_(rng_);
      auto a = utils::setTensorDictScalar<int64_t>("a", action);
      // std::cout << "RandomActor: " << utils::printTensorDict(a) << std::endl;
      return a;
    };
  }

 private:
  mutable std::mt19937 rng_;
  std::discrete_distribution<> gen_;
  // std::vector<int> hist_;
};
}
