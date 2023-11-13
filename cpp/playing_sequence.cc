#include "playing_sequence.h"

#include "rela/logging.h"

namespace bridge {

bool PlayingSequence::isPlayLegal(const Card& c, const Hand& h) const {
  // Does not have that card.
  if (!h.containsCard(c)) {
    return false;
  }

  // Beginning of the round.
  if (numCardsPlayed_ % kPlayer == 0) {
    return true;
  }

  RELA_CHECK_NE(currentSuit_, NO_SUIT);
  return !h.containsSuit(currentSuit_) || c.suit() == currentSuit_;
}

int PlayingSequence::makePlay(const Card& c, int seat) {
  if (isFirstTurn()) {
    currentSuit_ = c.suit();
    currentWinningSeat_ = seat;
    currentWinningCard_ = c;
  } else if (c.greaterThan(currentWinningCard_, trump_)) {
    currentWinningSeat_ = seat;
    currentWinningCard_ = c;
  }

  ++numCardsPlayed_;
  cardsPlayed_[c.index()] = numCardsPlayed_;

  int next_seat = NEXT_SEAT(seat);
  if (isFirstTurn()) {
    next_seat = currentWinningSeat_;
    currentSuit_ = NO_SUIT;
    if (IS_NS(currentWinningSeat_)) {
      ++numNSWins_;
    } else {
      ++numEWWins_;
    }
  }

  return next_seat;
}

}  // namespace bridge
