#pragma once

#include <math.h>
#include <torch/script.h>

#include "rela/model_locker.h"
#include "rela/utils.h"

#include <chrono>
#include <functional>
using namespace std::chrono;

namespace rela {

template <typename BatchType>
TensorDict sendAndWait(BatchType& batcher, TensorDict input) {
  int slot = -1;
  auto reply = batcher.send(input, &slot);
  auto output = reply->get(slot);
  return output;
}

template <typename BatchType>
std::function<TensorDict()> sendAndGetFuture(BatchType& batcher,
                                             TensorDict input) {
  int slot = -1;
  // std::cout << "About to send " << std::endl;
  // utils::tensorDictPrint(input);
  auto reply = batcher.send(input, &slot);
  return [reply, slot]() -> TensorDict {
    auto r = reply->get(slot);
    // std::cout << "got replay at slot " << slot << std::endl;
    // utils::tensorDictPrint(r);
    return r;
  };
}

template <typename T>
void addOneToJitInput(const torch::Device& device, TorchJitInput &jitInput, T v) {
  jitInput.push_back(v);
}

template <>
inline void addOneToJitInput(const torch::Device& device,
                             TorchJitInput& jitInput, TensorDict input) {
  jitInput.push_back(utils::tensorDictToTorchDict(input, device));
}

inline void addToJitInput(const torch::Device&, TorchJitInput &) { }

template <typename T, typename ...Args>
void addToJitInput(const torch::Device& device, TorchJitInput &jitInput, T input, Args... args) {
  addOneToJitInput<T>(device, jitInput, input);
  addToJitInput(device, jitInput, args...);
}

inline TensorDict modelForward(ModelLocker& modelLocker,
                               const std::string func_name,
                               TorchJitInput &jitInput) {
  torch::NoGradGuard noGrad;

  int id = -1;
  auto model = modelLocker.getModel(&id);
  TorchJitOutput jitOutput = model.get_method(func_name)(jitInput);
  modelLocker.releaseModel(id);

  return utils::iValueToTensorDict(jitOutput, torch::kCPU, true);
}

inline TensorDict modelForward(ModelLocker& modelLocker,
                               const std::string func_name,
                               TensorDict input,
                               const torch::Device& device) {
  TorchJitInput jitInput;
  addOneToJitInput(device, jitInput, input);
  return modelForward(modelLocker, func_name, jitInput);
}

}  // namespace rela
