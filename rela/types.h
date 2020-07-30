// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/torch.h>
#include <unordered_map>

namespace rela {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;
using TensorVecDict =
    std::unordered_map<std::string, std::vector<torch::Tensor>>;

using TorchTensorDict = torch::Dict<std::string, torch::Tensor>;
using TorchJitInput = std::vector<torch::jit::IValue>;
using TorchJitOutput = torch::jit::IValue;
using TorchJitModel = torch::jit::script::Module;

class Transition {
 public:
  Transition() = default;

  Transition(TensorDict& d)
      : d(d) {
  }

  static Transition makeBatch(std::vector<Transition> transitions, int batchdim,
                              const std::string& device);

  // Transition index(int i) const;

  Transition padLike() const;

  TorchJitInput toJitInput(const torch::Device& device) const;

  bool empty() const {
    bool anyEmpty = false;
    for (const auto& kv : d) {
      if (kv.second.dim() == 0)
        anyEmpty = true;
    }
    return anyEmpty;
  }

  TensorDict d;
};

class FFTransition {
 public:
  FFTransition() = default;

  FFTransition(TensorDict& obs,
               TensorDict& action,
               torch::Tensor& reward,
               torch::Tensor& terminal,
               torch::Tensor& bootstrap,
               TensorDict& nextObs)
      : obs(obs)
      , action(action)
      , reward(reward)
      , terminal(terminal)
      , bootstrap(bootstrap)
      , nextObs(nextObs) {
  }

  static FFTransition makeBatch(const std::vector<FFTransition>& transitions,
                                const std::string& device);

  FFTransition index(int i) const;

  FFTransition padLike() const;

  TorchJitInput toJitInput(const torch::Device& device) const;

  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  TensorDict nextObs;
};

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition(const std::vector<FFTransition>& transitions,
                TensorDict h0,
                torch::Tensor seqLen);

  RNNTransition index(int i) const;

  static RNNTransition makeBatch(const std::vector<RNNTransition>& transitions,
                                 const std::string& device);

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};

using TensorDictFuture = std::function<TensorDict()>;

}  // namespace rela
