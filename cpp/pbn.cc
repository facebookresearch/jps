#include "cpp/pbn.h"

#include <algorithm>

#include "cpp/hand.h"
#include "cpp/seat.h"

namespace bridge {

namespace {

bool findStartSeatAndPos(const std::string& pbn, int& seat, size_t& pos) {
  pos = 0;
  for (; pos < pbn.length(); ++pos) {
    if (seatToIndex(pbn[pos]) != kNoSeat || pbn[pos] == '.' ||
        cardToIndex(pbn[pos]) != -1) {
      break;
    }
  }
  if (pos + kPbnLength > pbn.length()) {
    return false;
  }
  seat = seatToIndex(pbn[pos]);
  if (seat == kNoSeat) {
    seat = kNorth;
  } else {
    if (pos + 2 + kPbnLength > pbn.length() || pbn[pos + 1] != ':') {
      return false;
    }
    pos += 2;
  }
  return true;
}

bool parseDealFromPbnImpl(const std::string& pbn, int seat, size_t pos,
                          int* deal) {
  int suit = kSpade;
  std::fill(deal, deal + kDeckSize, -1);
  std::array<int, kNumPlayers> cnt = {0, 0, 0, 0};
  for (size_t i = 0; i < kPbnLength; ++i) {
    const char ch = pbn.at(pos + i);
    if (ch == ' ') {
      seat = nextSeat(seat);
      suit = kSpade;
    } else if (ch == '.') {
      --suit;
    } else {
      const int value = cardToIndex(ch);
      if (value == -1) {
        return false;
      }
      const Card c(suit, value);
      if (deal[c.index()] != -1) {
        return false;
      }
      deal[c.index()] = seat;
      ++cnt[seat];
    }
  }
  return std::all_of(cnt.cbegin(), cnt.cend(),
                     [](int x) { return x == kHandSize; });
}

}  // namespace

bool parseDealFromPbn(const std::string& pbn,
                      std::array<int, kDeckSize>& deal) {
  int seat = kNorth;
  size_t pos = 0;
  if (!findStartSeatAndPos(pbn, seat, pos)) {
    return false;
  }
  return parseDealFromPbnImpl(pbn, seat, pos, deal.data());
}

bool parseDealFromPbn(const std::string& pbn, std::vector<int>& deal) {
  int seat = kNorth;
  size_t pos = 0;
  if (!findStartSeatAndPos(pbn, seat, pos)) {
    return false;
  }
  deal.resize(kDeckSize);
  deal.shrink_to_fit();
  return parseDealFromPbnImpl(pbn, seat, pos, deal.data());
}

}  // namespace bridge
