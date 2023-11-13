#include "rela/model.h"

namespace rela {

void BatchProcessor::batchForward() {
  while (running_) {
    const TensorDict input = batcher_.get();
    if (!input.empty()) {
      auto output = modelForward(*modelLocker_, funcName_, input, device_);
      batcher_.set(std::move(output));
    }
  }
}

void BatchProcessorStatic::batchForward() {
  TensorDict input = batcher_.get();

  if (input.empty()) {
    return;
  }

  auto output = modelForward(*modelLocker_, func_name_, input, device_);
  batcher_.set(std::move(output));
}

void Models::processRequest() {
  for (auto& it : processors_) {
    it.second->processRequest();
  }
}

}  // namespace rela
