#pragma once

namespace bridge {

constexpr int kNumPlayers = 4;

constexpr int kNoSeat = -1;
constexpr int kNorth = 0;
constexpr int kEast = 1;
constexpr int kSouth = 2;
constexpr int kWest = 3;

constexpr int kNS = 0;
constexpr int kEW = 1;

constexpr char kIndexToSeat[] = "NESW";

constexpr int seatToIndex(char seat) {
  switch (seat) {
    case 'N': {
      return kNorth;
    }
    case 'E': {
      return kEast;
    }
    case 'S': {
      return kSouth;
    }
    case 'W': {
      return kWest;
    }
    default: { return kNoSeat; }
  }
}

constexpr int partner(int seat) { return (seat + 2) % kNumPlayers; }

constexpr int nextSeat(int seat) { return (seat + 1) % kNumPlayers; }

constexpr int prevSeat(int seat) { return (seat + 3) % kNumPlayers; }

constexpr int relativeSeat(int self, int other) {
  return (other - self + kNumPlayers) % kNumPlayers;
}

constexpr bool isNS(int seat) { return (seat & 1) == kNS; };

constexpr bool isEW(int seat) { return (seat & 1) == kEW; };

constexpr bool isPartner(int a, int b) { return ((a - b) & 1) == 0; }

}  // namespace bridge
