#include "IndexedLoggerFactory.h"

namespace elf {
namespace logging {

void IndexedLoggerFactory::registerPy(pybind11::module& m) {
  namespace py = pybind11;

  py::class_<IndexedLoggerFactory>(m, "IndexedLoggerFactory")
      .def(py::init<CreatorT>())
      .def(py::init<CreatorT, size_t>())
      .def("makeLogger", &IndexedLoggerFactory::makeLogger);
}

std::shared_ptr<spdlog::logger> IndexedLoggerFactory::makeLogger(
    const std::string& prefix,
    const std::string& suffix) {
  size_t curCount = counter_++;
  std::string loggerName = prefix + std::to_string(curCount) + suffix;
  return creator_(loggerName);
}

} // namespace logging
} // namespace elf
