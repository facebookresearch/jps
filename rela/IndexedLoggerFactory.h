#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>

namespace elf {
namespace logging {

class IndexedLoggerFactory {
 public:
  using CreatorT =
      std::function<std::shared_ptr<spdlog::logger>(const std::string& name)>;

  static void registerPy(pybind11::module& m);

  IndexedLoggerFactory(CreatorT creator, size_t initIndex = 0)
      : creator_(std::move(creator)), counter_(initIndex) {}

  std::shared_ptr<spdlog::logger> makeLogger(
      const std::string& prefix,
      const std::string& suffix);

 private:
  CreatorT creator_;
  std::atomic_size_t counter_;
};

} // namespace logging
} // namespace elf
