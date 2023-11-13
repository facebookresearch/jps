#include "rela/model_locker.h"

#include "rela/logging.h"

namespace rela {

ModelLocker::ModelLocker(const std::vector<pybind11::object>& pyModels,
                         const std::string& device)
    : device_(torch::Device(device)),
      pyModels_(pyModels),
      modelCallCounts_(pyModels.size(), 0),
      modelIndex_(0) {
  RELA_CHECK(!pyModels_.empty());
  models_.reserve(pyModels_.size());
  for (const auto& model : pyModels_) {
    models_.push_back(model.attr("_c").cast<TorchJitModel*>());
  }
}

void ModelLocker::updateModel(const pybind11::object& pyModel) {
  std::unique_lock<std::mutex> lk(m_);
  const int idx = (modelIndex_ + 1) % modelCallCounts_.size();
  cv_.wait(lk, [this, idx] { return modelCallCounts_[idx] == 0; });
  lk.unlock();

  pyModels_[idx].attr("load_state_dict")(pyModel.attr("state_dict")());

  lk.lock();
  modelIndex_ = idx;
  lk.unlock();
}

const TorchJitModel& ModelLocker::getModel(int* idx) {
  std::lock_guard<std::mutex> lk(m_);
  *idx = modelIndex_;
  ++modelCallCounts_[modelIndex_];
  return *models_[modelIndex_];
}

void ModelLocker::releaseModel(int idx) {
  std::unique_lock<std::mutex> lk(m_);
  --modelCallCounts_[idx];
  if (modelCallCounts_[idx] == 0) {
    cv_.notify_one();
  }
}

}  // namespace rela
