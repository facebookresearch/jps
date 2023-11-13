#pragma once

#include <rela/actor2.h>
#include <sstream>
#include "bridge_common.h"
#include "console_messenger.h"
#include "util.h"

namespace std {
class thread;
}

namespace bridge {

class ConsoleActor : public rela::Actor2 {
 public:
  ConsoleActor(int id) : id_(id) {
    // std::cout << "ConsoleActor(" << id << ")" << std::endl;
  }

  rela::TensorDictFuture act(rela::TensorDict& obs) override {
    rela::TensorDict reply;
    reply["a"] = torch::zeros({1}).to(torch::kInt64);
    reply["pi"] = torch::zeros({kAction});

    auto messenger = ConsoleMessenger::get_messenger();
    if (messenger) {
      const std::vector<std::string>& tokens =
          ConsoleMessenger::get_messenger()->read_actor_msg(id_);
      std::cout << std::endl;
      if (tokens.size() > 3 && tokens[1] == "bid") {
        Bid b(tokens[3]);
        int a = b.index();
        reply["a"][0] = a;
        reply["pi"][a] = 1.0;
      }
    }

    return [=]() { return reply; };
  }

 protected:
  int id_;
};
}  // namespace bridge