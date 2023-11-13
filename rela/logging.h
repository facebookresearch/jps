#pragma once

#include <cassert>
#include <iostream>
#include <string>

#include "string_util.h"

namespace rela {
namespace logging {

template <class... Args>
void assertWithMessage(bool condition, Args&&... args) {
  if (!condition) {
    const std::string msg = utils::strCat(std::forward<Args>(args)...);
    std::cerr << msg << std::endl;
    abort();
  }
}

#define RELA_CHECK(condition, ...)                                             \
  rela::logging::assertWithMessage(condition, #condition, " check failed at ", \
                                   __FILE__, ":", __LINE__, ". ",              \
                                   ##__VA_ARGS__);

#define RELA_CHECK_NOTNULL(x, ...)                                          \
  rela::logging::assertWithMessage(                                         \
      (x) != nullptr, #x " is not nullptr check failed at ", __FILE__, ":", \
      __LINE__, ". ", ##__VA_ARGS__)

#define RELA_CHECK_EQ(x, y, ...)                                              \
  rela::logging::assertWithMessage(                                           \
      (x) == (y), #x " == " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RELA_CHECK_NE(x, y, ...)                                              \
  rela::logging::assertWithMessage(                                           \
      (x) != (y), #x " != " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RELA_CHECK_LT(x, y, ...)                                            \
  rela::logging::assertWithMessage(                                         \
      (x) < (y), #x " < " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RELA_CHECK_LE(x, y, ...)                                              \
  rela::logging::assertWithMessage(                                           \
      (x) <= (y), #x " <= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RELA_CHECK_GT(x, y, ...)                                            \
  rela::logging::assertWithMessage(                                         \
      (x) > (y), #x " > " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RELA_CHECK_GE(x, y, ...)                                              \
  rela::logging::assertWithMessage(                                           \
      (x) >= (y), #x " >= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

}  // namespace logging
}  // namespace rela
