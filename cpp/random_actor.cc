#include "cpp/random_actor.h"

#include "rela/logging.h"
#include "rela/types.h"

namespace bridge {

rela::TensorDictFuture RandomActor::act(rela::TensorDict& obs) {
  obs_ = obs;
  return [this]() { return this->randomAct(); };
}

rela::TensorDict RandomActor::randomAct() const {
  const auto it = obs_.find("legal_move");
  RELA_CHECK(it != obs_.cend(), "Missing key \"legal_move\" in obs.");
  const auto& mov = it->second;
  auto pi = mov / mov.sum(-1, /*keepdim=*/true);
  const auto act = pi.multinomial(1, /*replacement=*/true);
  // const auto v = torch::tensor({0}, torch::kFloat);
  return {{"a", act}, {"pi", pi}};
}

}  // namespace bridge
