#include "rela/batcher.h"

namespace rela {

TensorDict allocateBatchStorage(const TensorDict& data, int size) {
  TensorDict storage;
  for (const auto& kv : data) {
    const auto& t = kv.second.sizes();
    std::vector<int64_t> sizes(t.size() + 1);
    sizes[0] = size;
    std::copy(t.cbegin(), t.cend(), sizes.begin() + 1);
    storage[kv.first] = torch::zeros(sizes, kv.second.dtype());
  }
  return storage;
}

}  // namespace rela
