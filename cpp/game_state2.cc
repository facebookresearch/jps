#include "cpp/game_state2.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include "cpp/pbn.h"
#include "cpp/score.h"
#include "rela/string_util.h"

namespace bridge {

void GameState2::reset() {
  for (int i = 0; i < kNumPlayers; ++i) {
    playerToSeat_[i] = i;
    seatToPlayer_[i] = i;
  }
  for (auto& hand : hands_) {
    hand.clear();
  }

  vul_ = 0;

  initialStage_ = kStageDealing;
  terminalStage_ = kStageScoring;
  currentStage_ = kStageDealing;

  dealer_ = kNoSeat;
  declarer_ = kNoSeat;
  currentSeat_ = kNoSeat;
  contract_ = Bid();
  std::fill(scores_.begin(), scores_.end(), 0);

  resetBiddingStatus();
  resetPlayingStatus();
  std::fill(ddTable_.begin(), ddTable_.end(), -1);
}

void GameState2::dealFromPbn(const std::string& pbn, int dealer, int vul) {
  RELA_CHECK(parseDealFromPbn(pbn, deal_));
  pbn_ = pbn;
  for (int i = 0; i < kDeckSize; ++i) {
    const Card card(i);
    hands_[deal_[i]].add(card);
  }
  vul_ = vul;
  currentStage_ = kStageBidding;
  dealer_ = dealer;
  currentSeat_ = dealer;
}

void GameState2::resetBiddingStatus() {
  biddingHistory_.clear();
  biddingHistory_.reserve(kMaxBiddingHistory);
  doubled_ = 0;
  highestBidSeat_ = kNoSeat;
  numConsecutivePasses_ = 0;
}

void GameState2::resetPlayingStatus() {
  playingHistory_.clear();
  playingHistory_.reserve(kDeckSize);
  playedCards_ = 0;
  std::fill(currentTrick_.begin(), currentTrick_.end(), Card());
  leadingSuit_ = kNoSuit;
  winningSeat_ = kNoSeat;
  std::fill(winningSeats_.begin(), winningSeats_.end(), -1);
  std::fill(numWins_.begin(), numWins_.end(), 0);
}

bool GameState2::isBidLegal(const Bid& bid) const {
  switch (bid.type()) {
    case kBidPass: {
      // Can always pass.
      return true;
    }
    case kBidDouble: {
      // Not doubled, not redobuled, cannot double your own contract.
      return contract_.type() == kBidNormal && doubled_ == 0 &&
             !isPartner(currentSeat_, highestBidSeat_);
    }
    case kBidRedouble: {
      // Doubled, not redoubled, can only redouble your own contract.
      return doubled_ == kBidDoubledMask &&
             isPartner(currentSeat_, highestBidSeat_);
    }
    case kBidNull: {
      return false;
    }
    default: {
      return contract_.type() == kBidNull || bid > contract_;
    }
  }
}

uint64_t GameState2::legalBiddingActions() const {
  uint64_t mask = 0;
  for (int i = 0; i < kNumBids; ++i) {
    const Bid bid(i);
    if (isBidLegal(bid)) {
      mask |= (1LLU << i);
    }
  }
  return mask;
}

bool GameState2::isPlayLegal(const Card& card) const {
  const Hand& h = hands_.at(currentSeat_);
  if (!h.containsCard(card)) {
    return false;
  }
  if (leadingSuit_ == kNoSuit) {
    return true;
  }
  return !h.containsSuit(leadingSuit_) || card.suit() == leadingSuit_;
}

uint64_t GameState2::legalPlayingActions() const {
  uint64_t mask = 0;
  for (int i = 0; i < kDeckSize; ++i) {
    const Card card(i);
    if (isPlayLegal(card)) {
      mask |= (1LLU << i);
    }
  }
  return mask;
}

void GameState2::biddingStep(int act) {
  const Bid bid(act);
  RELA_CHECK(isBidLegal(bid), "Bid ", bid.toString(), " (action index: ", act,
             ") is illegal for player ", currentPlayer(), " at seat ",
             kIndexToSeat[currentSeat_]);

  switch (bid.type()) {
    case kBidPass: {
      ++numConsecutivePasses_;
      break;
    }
    case kBidDouble: {
      doubleBid(doubled_);
      numConsecutivePasses_ = 0;
      break;
    }
    case kBidRedouble: {
      redoubleBid(doubled_);
      numConsecutivePasses_ = 0;
      break;
    }
    case kBidNormal: {
      contract_ = bid;
      doubled_ = 0;
      highestBidSeat_ = currentSeat_;
      numConsecutivePasses_ = 0;
      break;
    }
    default: {
      break;
    }
  }
  biddingHistory_.push_back(bid);

  if (!isBiddingEnd()) {
    currentSeat_ = nextSeat(currentSeat_);
  } else if (contract_.type() == kBidNull) {
    currentStage_ = kStageScoring;
    currentSeat_ = kNoSeat;
  } else {
    const int biddingLength = biddingHistory_.size();
    // Iterate through records in biddingHistory_ whose seat is the parter of
    // highestBidSeat_.
    for (int i = ((highestBidSeat_ - dealer_ + kNumPlayers) & 1);
         i < biddingLength; i += 2) {
      if (biddingHistory_[i].strain() == contract_.strain()) {
        declarer_ = (dealer_ + i) % kNumPlayers;
        break;
      }
    }
    currentStage_ = kStagePlaying;
    openingLead_ = nextSeat(declarer_);
    currentSeat_ = openingLead_;

    // Get score from ddtable.
    if (currentStage_ >= terminalStage_) {
      const int declarerSide = (declarer_ & 1);
      const int wins = ddTable_[declarer_ * kNumStrains + contract_.strain()];
      const int score = computeDeclarerScore(contract_, wins, doubled_,
                                             ((vul_ >> declarerSide) & 1));
      if (score > 0) {
        scores_[declarerSide] += score;
      } else {
        scores_[declarerSide ^ 1] += -score;
      }
    }
  }
}

void GameState2::playingStep(int act) {
  const Card card(act);
  if (!isPlayLegal(card)) {
    std::cout << "Current state: " << std::endl;
    std::cout << getInfo() << std::endl;
  }
  RELA_CHECK(isPlayLegal(card), "Card ", card.toString(),
             " (action index: ", act, ") is illegal for player ",
             currentPlayer(), " at seat ", kIndexToSeat[currentSeat_]);

  if (leadingSuit_ == kNoSuit) {
    leadingSuit_ = card.suit();
    winningSeat_ = currentSeat_;
    winningCard_ = card;
  } else if (card.greaterThan(winningCard_, contract_.strain())) {
    winningSeat_ = currentSeat_;
    winningCard_ = card;
  }
  playingHistory_.emplace_back(card, currentSeat_);
  hands_[currentSeat_].remove(card);
  playedCards_ |= (1LLU << card.index());
  currentTrick_[currentSeat_] = card;

  if (playingHistory_.size() % kNumPlayers != 0) {
    currentSeat_ = nextSeat(currentSeat_);
  } else {
    ++numWins_[winningSeat_ & 1];
    winningSeats_[playingHistory_.size() / kNumPlayers - 1] = winningSeat_;
    currentSeat_ = winningSeat_;
    leadingSuit_ = kNoSuit;
    winningSeat_ = kNoSeat;
    winningCard_ = Card();
    std::fill(currentTrick_.begin(), currentTrick_.end(), Card());
  }

  if (isPlayingEnd()) {
    const int declarerSide = (declarer_ & 1);
    const int score =
        computeDeclarerScore(contract_, numWins_[declarerSide], doubled_,
                             ((vul_ >> declarerSide) & 1));
    if (score > 0) {
      scores_[declarerSide] += score;
    } else {
      scores_[declarerSide ^ 1] += -score;
    }
    currentStage_ = kStageScoring;
  }
}

std::string GameState2::getInfo() const {
  // Print out all information.
  std::stringstream ss;

  // Print out current cards.
  ss << "Full hands: " << std::endl;
  for (int i = 0; i < kNumPlayers; ++i) {
    ss << kIndexToSeat[i] << ": " << hands_[i].originalHandString()
       << std::endl;
  }

  ss << std::endl;
  ss << "Current hands: " << std::endl;
  for (int i = 0; i < kNumPlayers; ++i) {
    ss << kIndexToSeat[i] << ": " << hands_[i].toString() << std::endl;
  }

  ss << std::endl << "Bidding: " << std::endl;
  for (int i = 0; i < kNumPlayers; ++i) {
    ss << std::setw(2) << kIndexToSeat[i] << " ";
  }
  ss << std::endl;

  int bidIdx = 0;
  for (; bidIdx < dealer_; ++bidIdx) {
    ss << "   ";
  }
  for (const auto& bid : biddingHistory_) {
    ss << std::setw(2) << bid.toString() << " ";
    bidIdx++;
    if (bidIdx >= kNumPlayers) {
      bidIdx = 0;
      ss << std::endl;
    }
  }
  ss << std::endl;
  ss << "Contract: " << contract_.toString()
     << ", Declarer: " << kIndexToSeat[declarer_] << std::endl;

  // Playing sequence so far:
  int count = 0;
  for (const auto& it : playingHistory_) {
    int round = count / kNumPlayers;
    int idx = count % kNumPlayers;

    if (idx == 0) {
      // Round and who to start.
      int lastWinSeat = round > 0 ? winningSeats_[round - 1] : openingLead_;
      ss << std::endl
         << std::setw(2) << round << "[" << kIndexToSeat[lastWinSeat] << "]: ";
    }
    ss << it.first.toString() << " ";
    count++;
    if (idx == kNumPlayers - 1) {
      int winSeat = winningSeats_[round];
      if (((winSeat - declarer_) & 1) == 0) {
        // Declare win the round.
        ss << " *";
      }
    }
  }

  ss << std::endl << std::endl;
  for (int i = 0; i < kNumPlayers; ++i) {
    ss << kIndexToSeat[i] << ": " << numWins_[i & 1] << " ";
  }
  ss << std::endl;

  // DDT table.
  if (ddTable_[0] >= 0) {
    ss << std::endl << "   ";
    for (int i = 0; i < kNumStrains; ++i) {
      ss << std::setw(2) << kIndexToSuit[i] << " ";
    }
    ss << std::endl;
    for (int j = 0; j < kNumPlayers; ++j) {
      ss << kIndexToSeat[j] << ": ";
      for (int i = 0; i < kNumStrains; ++i) {
        ss << std::setw(2) << ddTable_[j * kNumStrains + i] << " ";
      }
      ss << std::endl;
    }
    ss << std::endl;
  }
  return ss.str();
}

}  // namespace bridge
