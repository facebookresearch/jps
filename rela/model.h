#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <torch/script.h>

#include "rela/batcher.h"
#include "rela/model_common.h"
#include "rela/static_batcher.h"

namespace rela {

// Wrapper of multiple models and call with keys.
class BatchProcessor {
 public:
  BatchProcessor(std::shared_ptr<ModelLocker> modelLocker,
                 const std::string& funcName, int batchsize,
                 const std::string& device)
      : modelLocker_(modelLocker),
        funcName_(funcName),
        batcher_(batchsize),
        device_(torch::Device(device)),
        forwardThread_(&BatchProcessor::batchForward, this) {}

  ~BatchProcessor() {
    batcher_.exit();
    running_ = false;
    forwardThread_.join();
  }

  template <typename... Args>
  TensorDict forward(Args... args) {
    TorchJitInput jitInput;
    addToJitInput(device_, jitInput, args...);
    return modelForward(*modelLocker_, funcName_, jitInput);
  }

  Batcher& batcher() { return batcher_; }

  void processRequest() {}

 private:
  void batchForward();

  std::shared_ptr<ModelLocker> modelLocker_;
  const std::string funcName_;
  Batcher batcher_;
  torch::Device device_;

  std::atomic<bool> running_{true};
  std::thread forwardThread_;
};

// Wrapper of multiple models and call with keys.
class BatchProcessorStatic {
 public:
  // [TODO]: Note that static version doesn't need batchsize. But we keep it for
  // a placeholder.
  BatchProcessorStatic(std::shared_ptr<ModelLocker> modelLocker,
                       std::string func_name, int /*batchsize*/,
                       const std::string& device)
      : modelLocker_(modelLocker),
        func_name_(func_name),
        device_(torch::Device(device)) {}

  static_batcher::Batcher& batcher() { return batcher_; }

  void processRequest() { batchForward(); }

 private:
  void batchForward();

  std::shared_ptr<ModelLocker> modelLocker_;
  std::string func_name_;
  static_batcher::Batcher batcher_;
  torch::Device device_;
};

using BatchProcessorUnit = BatchProcessor;
// using BatchProcessorUnit = BatchProcessorStatic;

class Models {
 public:
  void add(const std::string& callname,
           std::shared_ptr<BatchProcessorUnit> processor) {
    processors_[callname] = std::move(processor);
  }

  std::function<TensorDict()> call(const std::string& callname,
                                   const TensorDict& input) {
    return sendAndGetFuture(processors_.at(callname)->batcher(), input);
  }

  template <typename... Args>
  TensorDict callDirect(std::string callname, Args... args) {
    return processors_.at(callname)->forward(args...);
  }

  void processRequest();

 private:
  std::unordered_map<std::string, std::shared_ptr<BatchProcessorUnit>>
      processors_;
};

}  // namespace rela
