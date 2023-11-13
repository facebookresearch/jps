// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>

#include "rela/types.h"

namespace rela {

class ModelLocker {
 public:
  ModelLocker(const std::vector<pybind11::object>& pyModels,
              const std::string& device);

  const torch::Device& device() const { return device_; }

  void updateModel(const pybind11::object& pyModel);

  const TorchJitModel& getModel(int* idx);

  void releaseModel(int idx);

 private:
  const torch::Device device_;
  std::vector<pybind11::object> pyModels_;
  std::vector<int> modelCallCounts_;
  int modelIndex_;

  std::vector<TorchJitModel*> models_;
  std::mutex m_;
  std::condition_variable cv_;
};

}  // namespace rela
