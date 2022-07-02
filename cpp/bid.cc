#include "cpp/bid.h"

namespace bridge {

Bid::Bid(const std::string& str) {
  if (str == "P" || str == "p") {
    type_ = kBidPass;
  } else if (str == "X" || str == "x") {
    type_ = kBidDouble;
  } else if (str == "XX" || str == "xx") {
    type_ = kBidRedouble;
  } else if (str.length() >= 2 && std::isdigit(str[0])) {
    type_ = kBidNormal;
    level_ = str[0] - '0';
    RELA_CHECK_GE(level_, 1);
    RELA_CHECK_LE(level_, 7);
    strain_ = suitToIndex(str[1]);
    RELA_CHECK_NE(strain_, kNoSuit);
  }
}

std::string Bid::toString() const {
  switch (type_) {
    case kBidPass: {
      return "P";
    }
    case kBidDouble: {
      return "X";
    }
    case kBidRedouble: {
      return "XX";
    }
    case kBidNull: {
      return "NA";
    }
    default: { return rela::utils::strCat(level_, kIndexToSuit[strain_]); }
  }
}

}  // namespace bridge
