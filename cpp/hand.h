#pragma once

#include <array>
#include <string>

#include "cpp/card.h"

namespace bridge {

constexpr int kHandSize = 13;

class Hand {
 public:
  Hand() { clear(); }

  bool containsCard(const Card& card) const {
    return ((mask_[card.suit()] >> card.value()) & 1) == 1;
  }

  bool containsSuit(int suit) const { return mask_[suit] != 0; }

  void clear() {
    std::fill(hand_.begin(), hand_.end(), 0);
    std::fill(mask_.begin(), mask_.end(), 0);
  }

  bool add(const Card& card) {
    const bool ret = !containsCard(card);
    hand_[card.suit()] |= (1U << card.value());
    mask_[card.suit()] |= (1U << card.value());
    return ret;
  }

  bool remove(const Card& card) {
    const bool ret = containsCard(card);
    if (ret) {
      mask_[card.suit()] ^= (1U << card.value());
    }
    return ret;
  }

  std::string toString() const;

  std::string originalHandString() const;

 private:
  // mask for original cards in each suit.
  std::array<uint32_t, kNumSuits> hand_;
  // mask for availale cards in each suit.
  std::array<uint32_t, kNumSuits> mask_;
};

}  // namespace bridge
