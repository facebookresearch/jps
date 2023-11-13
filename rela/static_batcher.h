
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "rela/utils.h"

namespace rela {

namespace static_batcher {

class FutureReply {
 public:
  FutureReply()
      : ready_(false) {
  }

  void setNotReady() {
    ready_ = false;
  }

  TensorDict get(int slot) {
    assert(ready_);
    TensorDict e;
    for (const auto& kv : data_) {
      assert(slot >= 0 && slot < kv.second.size(0));
      e[kv.first] = kv.second[slot];
    }
    return e;
  }

  void set(TensorDict&& t) {
    data_ = std::move(t);
    ready_ = true;
  }

 private:
  // no need for protection, only set() can set it
  // torch::Tensor data_;
  TensorDict data_;
  bool ready_;
};

class Batcher {
 public:
  Batcher()
      : reply_(std::make_shared<FutureReply>()) {
  }

  // send data into batcher
  std::shared_ptr<FutureReply> send(const TensorDict& t, int* slot) {
    *slot = (int)dicts_.size();
    dicts_.push_back(t);
    reply_->setNotReady();
    return reply_;
  }

  // get batch input from batcher
  TensorDict get() {
    TensorDict batch;
    if (dicts_.empty())
      return batch;

    int batchsize = dicts_.size();

    // init buffer
    for (const auto& kv : dicts_[0]) {
      auto t = kv.second.sizes();
      std::vector<int64_t> sizes;
      sizes.push_back(batchsize);
      sizes.insert(sizes.end(), t.begin(), t.end());

      batch[kv.first] = torch::zeros(sizes, kv.second.dtype());
    }

    // copy all data in.
    for (int i = 0; i < (int)dicts_.size(); ++i) {
      // this will copy
      for (const auto& kv : dicts_[i]) {
        batch[kv.first][i] = kv.second;
      }
    }

    dicts_.clear();
    return batch;
  }

  // set batch reply for batcher
  void set(TensorDict&& t) {
    for (const auto& kv : t) {
      assert(kv.second.device().is_cpu());
    }
    reply_->set(std::move(t));
  }

 private:
  std::vector<TensorDict> dicts_;
  std::shared_ptr<FutureReply> reply_;
};
}

}  // namespace rela
