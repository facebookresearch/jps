#pragma once
#include <cassert>

namespace bridge {

class Action {
 public:
  Action()
      : action_(-1) {
  }

  void setAction(int action) {
    action_ = action;
  }

  int getAction() const {
    if (action_ == -1) {
      std::cout << "Error: action not set" << std::endl;
      assert(false);
    }
    return action_;
  }

 private:
  int action_;
};

}  // namespace bridge
