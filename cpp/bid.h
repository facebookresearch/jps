#pragma once

#include <cctype>
#include <string>

#include "cpp/card.h"
#include "rela/logging.h"
#include "rela/string_util.h"

namespace bridge {

constexpr int kMaxBidLevel = 7;
constexpr int kNumStrains = 5;
constexpr int kNumNormalBids = kMaxBidLevel * kNumStrains;
constexpr int kNumSpecialBids = 4;
constexpr int kNumBids = kNumNormalBids + kNumSpecialBids;

constexpr int kBidNormal = 0;
constexpr int kBidPass = 1;
constexpr int kBidDouble = 2;
constexpr int kBidRedouble = 3;
constexpr int kBidNull = 4;

constexpr uint32_t kBidDoubledMask = 1;
constexpr uint32_t kBidReDoubledMask = 2;

class Bid {
 public:
  Bid() = default;

  Bid(const Bid& bid) = default;

  Bid(int level, int strain)
      : type_(kBidNormal), level_(level), strain_(strain) {}

  Bid(int index) {
    RELA_CHECK_GE(index, 0);
    RELA_CHECK_LT(index, kNumBids);
    if (index >= kNumNormalBids) {
      type_ = index - kNumNormalBids + 1;
    } else {
      type_ = kBidNormal;
      level_ = index / kNumStrains + 1;
      strain_ = index % kNumStrains;
    }
  }

  Bid(const std::string& str);

  Bid& operator=(const Bid& bid) = default;

  int type() const { return type_; }

  int level() const { return level_; }

  int strain() const { return strain_; }

  int index() const {
    if (type_ != kBidNormal) {
      return kNumNormalBids + type_ - 1;
    }
    RELA_CHECK_GE(level_, 1);
    RELA_CHECK_LE(level_, 7);
    return (level_ - 1) * kNumStrains + strain_;
  }

  std::string toString() const;

 private:
  int type_ = kBidNull;
  int level_ = 0;
  int strain_ = kNoSuit;
};

inline bool operator==(const Bid& lhs, const Bid& rhs) {
  return lhs.index() == rhs.index();
}

inline bool operator!=(const Bid& lhs, const Bid& rhs) {
  return lhs.index() != rhs.index();
}

inline bool operator<(const Bid& lhs, const Bid& rhs) {
  return lhs.index() < rhs.index();
}

inline bool operator>(const Bid& lhs, const Bid& rhs) {
  return lhs.index() > rhs.index();
}

constexpr void doubleBid(uint32_t& mask) { mask |= 1; }

constexpr void redoubleBid(uint32_t& mask) { mask ^= 3; }

}  // namespace bridge
