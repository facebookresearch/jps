#pragma once

#include <string>

#include "rela/logging.h"
#include "rela/string_util.h"

namespace bridge {

constexpr int kNumSuits = 4;
constexpr int kSuitSize = 13;
constexpr int kDeckSize = kNumSuits * kSuitSize;

constexpr int kNoSuit = -1;
constexpr int kClub = 0;
constexpr int kDiamond = 1;
constexpr int kHeart = 2;
constexpr int kSpade = 3;
constexpr int kNoTrump = 4;

constexpr char kIndexToSuit[] = "CDHSN";
constexpr char kIndexToCard[] = "AKQJT98765432";

constexpr char kSuitUnicode[][4] = {"\u2663", "\u2666", "\u2665", "\u2660"};

constexpr int cardToIndex(char card) {
  switch (card) {
    case 'A': {
      return 0;
    }
    case 'K': {
      return 1;
    }
    case 'Q': {
      return 2;
    }
    case 'J': {
      return 3;
    }
    case 'T': {
      return 4;
    }
    case '9': {
      return 5;
    }
    case '8': {
      return 6;
    }
    case '7': {
      return 7;
    }
    case '6': {
      return 8;
    }
    case '5': {
      return 9;
    }
    case '4': {
      return 10;
    }
    case '3': {
      return 11;
    }
    case '2': {
      return 12;
    }
    default: { return -1; }
  }
}

constexpr int suitToIndex(char suit) {
  switch (suit) {
    case 'C': {
      return kClub;
    }
    case 'D': {
      return kDiamond;
    }
    case 'H': {
      return kHeart;
    }
    case 'S': {
      return kSpade;
    }
    case 'N': {
      return kNoTrump;
    }
    default: { return kNoSuit; }
  }
}

class Card {
 public:
  Card() = default;

  Card(const Card& c) = default;

  Card(int idx) : suit_(idx / kSuitSize), value_(idx % kSuitSize) {
    RELA_CHECK_GE(idx, 0);
    RELA_CHECK_LT(idx, kDeckSize);
  }

  Card(int suit, int value) : suit_(suit), value_(value) {}

  Card(const std::string& str) {
    RELA_CHECK_EQ(str.length(), 2);
    suit_ = suitToIndex(str[0]);
    RELA_CHECK_NE(suit_, kNoSuit);
    RELA_CHECK_NE(suit_, kNoTrump);
    value_ = cardToIndex(str[1]);
    RELA_CHECK_NE(value_, -1);
  }

  Card& operator=(const Card& c) = default;

  int suit() const { return suit_; }

  int value() const { return value_; }

  int index() const { return suit_ * kSuitSize + value_; }

  std::string toString() const {
    return rela::utils::strCat(kIndexToSuit[suit_], kIndexToCard[value_]);
  }

  std::string toUnicodeString() const {
    return rela::utils::strCat(kSuitUnicode[suit_], kIndexToCard[value_]);
  }

  bool greaterThan(const Card& other, int trump = kNoTrump) const {
    // Cards values are in reverse order, so less value means larger card.
    return suit_ == other.suit() ? value_ < other.value() : suit_ == trump;
  }

 private:
  int suit_ = kNoSuit;
  int value_ = -1;
};

inline bool operator==(const Card& lhs, const Card& rhs) {
  return lhs.suit() == rhs.suit() && lhs.value() == rhs.value();
}

inline bool operator!=(const Card& lhs, const Card& rhs) {
  return lhs.suit() != rhs.suit() || lhs.value() != rhs.value();
}

}  // namespace bridge
