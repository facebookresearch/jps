#include "hand.h"

namespace bridge {

namespace {

std::string maskToString(const std::array<uint32_t, kNumSuits>& mask) {
  std::string result;
  for (int i = 0; i < kNumSuits; ++i) {
    result += kSuitUnicode[i];
    for (int j = 0; j < kSuitSize; ++j) {
      if (((mask[i] >> j) & 1) == 1) {
        result.push_back(kIndexToCard[j]);
      }
    }
    result.push_back(' ');
  }
  return result;
}

}  // namespace

std::string Hand::toString() const { return maskToString(mask_); }

std::string Hand::originalHandString() const { return maskToString(hand_); }

}  // namespace bridge
