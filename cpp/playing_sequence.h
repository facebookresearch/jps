#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "bridge_common.h"
#include "card.h"
#include "hand.h"

namespace bridge {

class PlayingSequence {
 public:
  PlayingSequence() { clear(); }

  PlayingSequence(int trump) : trump_(trump) { clear(); }

  bool isPlayLegal(const Card& c, const Hand& h) const;

  bool isPlayEnd() const { return numCardsPlayed_ == kDeck; }

  bool isFirstTurn() const { return numCardsPlayed_ % kPlayer == 0; }

  const std::array<int, kDeck>& cardsPlayed() const { return cardsPlayed_; }

  void clear() { std::fill(cardsPlayed_.begin(), cardsPlayed_.end(), 0); }

  int makePlay(const Card& c, int seat);

 private:
  const int trump_ = NO_SUIT;

  std::array<int, kDeck> cardsPlayed_;
  int numCardsPlayed_ = 0;

  int currentSuit_ = NO_SUIT;
  int currentWinningSeat_ = NO_SEAT;
  Card currentWinningCard_;

  int numNSWins_ = 0;
  int numEWWins_ = 0;
};

}  // namespace bridge
