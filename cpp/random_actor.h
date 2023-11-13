#pragma once

#include <random>

#include "rela/actor2.h"
#include "rela/types.h"

namespace bridge {

class RandomActor : public rela::Actor2 {
 public:
  int numAct() const { return 0; }

  rela::TensorDictFuture act(rela::TensorDict& obs) override;

 private:
  rela::TensorDict randomAct() const;

  rela::TensorDict obs_;
};

}  // namespace bridge
