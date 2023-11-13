// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include "rela/types.h"
#include <functional>

namespace rela {

class FutureExecutor {
 public:
  void addFuture(std::function<void()> f) {
    futures_.push_back(f);
  }

  void execute()  {
    int count = 0;
    while (! futures_.empty()) {
      auto f = futures_.front();
      f();
      futures_.pop_front();
      count ++;
      if (count > 2000000) {
        std::cout << "FutureExecutor::execute(): counter exceeded. Something wrong!" << std::endl;
      }

      // [TODO]: This is needed if we use static batching 
      // If there is new request then:
      // for (auto &v : models_) { v->processRequest(); }
    }
  }

 private:
  std::deque<std::function<void()>> futures_;
};

class Actor2 {
 public:
  Actor2() = default;

  void setFutureExecutor(std::shared_ptr<FutureExecutor> executor) {
    executor_ = std::move(executor);
  }

  virtual ~Actor2() {
  }

  virtual TensorDictFuture act(TensorDict&) = 0;
  // Called if the associated environment send a terminal signal.  
  // Useful if the actor has internal state. 
  virtual void setTerminal() { }

  // Send experience to replay buffer.  
  virtual void sendExperience(TensorDict&) { }

 private:
  std::shared_ptr<FutureExecutor> executor_;

 protected:
  void addFuture(std::function<void()> f) {
    assert(executor_ != nullptr);
    assert(f != nullptr);
    executor_->addFuture(f);
  }
};
}
