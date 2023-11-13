#pragma once

#include <memory>

#include "rela/actor2.h"
#include "rela/model.h"
#include "rela/types.h"

namespace bridge {

class GreedyPlayActor : public rela::Actor2 {
 public:
  GreedyPlayActor() = default;

  GreedyPlayActor(std::shared_ptr<rela::Models> models) : models_(models) {}

  int numAct() const { return 0; }

  rela::TensorDictFuture act(rela::TensorDict& obs) override;

 private:
  rela::TensorDict actImpl() const;

  rela::TensorDict biddingAct() const;

  rela::TensorDict playingAct() const;

  std::shared_ptr<rela::Models> models_ = nullptr;
  rela::TensorDict obs_;
};

}  // namespace bridge
