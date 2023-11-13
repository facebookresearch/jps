#include "string_util.h"

#include <utility>

namespace rela {
namespace utils {

std::vector<std::string> strSplit(const std::string& str, char delimiter,
                                  bool allowEmpty) {
  std::vector<std::string> result;
  std::string cur;
  for (const char ch : str) {
    if (ch == delimiter) {
      if (allowEmpty || !cur.empty()) {
        result.emplace_back(std::move(cur));
      }
    } else {
      cur.push_back(ch);
    }
  }
  if (allowEmpty || !cur.empty()) {
    result.emplace_back(std::move(cur));
  }
  return result;
}

}  // namespace utils
}  // namespace rela
