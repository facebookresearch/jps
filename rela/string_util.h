#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace rela {
namespace utils {

template <class... Args>
std::string strCat(const Args&... args) {
  using Expander = int[];
  std::stringstream ss;
  (void)Expander{0, (void(ss << args), 0)...};
  return ss.str();
}

std::vector<std::string> strSplit(const std::string& str, char delimiter,
                                  bool allowEmpty = true);

}  // namespace utils
}  // namespace rela
