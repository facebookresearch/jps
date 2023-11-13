#pragma once

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpp/bid.h"
#include "cpp/card.h"
#include "cpp/hand.h"
#include "cpp/seat.h"
#include "nlohmann/json.hpp"
#include "rela/logging.h"
#include "rela/types.h"

namespace bridge {

constexpr int kStageDealing = 0;
constexpr int kStageBidding = 1;
constexpr int kStagePlaying = 2;
constexpr int kStageScoring = 3;

constexpr int kMaxBiddingHistory = 40;

constexpr int kNumVuls = 4;
constexpr int kVulNone = 0;
constexpr int kVulNS = 1;
constexpr int kVulEW = 2;
constexpr int kVulBoth = 3;

constexpr int kNumTricks = 13;

class GameState2 {
 public:
  GameState2() { reset(); }

  GameState2(const GameState2& game) = default;

  void reset();

  const std::array<int, kDeckSize>& deal() const { return deal_; }

  const std::array<Hand, kNumPlayers>& hands() const { return hands_; }

  const Hand& hand(int seat) const { return hands_.at(seat); }

  int playerToSeat(int player) const { return playerToSeat_.at(player); }

  int seatToPlayer(int seat) const { return seatToPlayer_.at(seat); }

  void setPlayerSeat(int offset) {
    for (int i = 0; i < kNumPlayers; ++i) {
      playerToSeat_[i] = (i + offset) % kNumPlayers;
      seatToPlayer_[playerToSeat_[i]] = i;
    }
  }

  void setPlayerSeat(const std::array<int, 4>& seats) {
    RELA_CHECK_EQ(seats.size(), kNumPlayers);
    for (int i = 0; i < kNumPlayers; ++i) {
      RELA_CHECK(seats[i] >= 0 && seats[i] < kNumPlayers);
      playerToSeat_[i] = seats[i];
      seatToPlayer_[seats[i]] = i;
    }
  }

  int initialStage() const { return initialStage_; }

  int terminalStage() const { return terminalStage_; }

  void setGameRange(int initialStage, int terminalStage) {
    initialStage_ = initialStage;
    terminalStage_ = terminalStage;
    currentStage_ = initialStage;
  }

  int currentStage() const { return currentStage_; }

  int vul() const { return vul_; }

  int dealer() const { return dealer_; }

  int declarer() const { return declarer_; }

  bool terminated() const { return currentStage_ >= terminalStage_; }

  int currentSeat() const { return currentSeat_; }

  int currentPlayer() const {
    if (currentStage_ == kStagePlaying && isPartner(declarer_, currentSeat_)) {
      // Dummy hands played by declarer.
      return seatToPlayer_.at(declarer_);
    }
    return seatToPlayer_.at(currentSeat_);
  }

  const Bid& contract() const { return contract_; }

  int score(int side) const { return scores_.at(side); }

  int relativeScore(int side) const {
    return scores_.at(side) - scores_.at(side ^ 1);
  }

  template <typename Container>
  void dealFromDeck(const Container& deck, int dealer, int vul) {
    for (int i = 0; i < kDeckSize; ++i) {
      const int seat = i / kHandSize;
      deal_[deck[i]] = seat;
      hands_[seat].add(Card(deck[i]));
    }
    vul_ = vul;
    currentStage_ = kStageBidding;
    dealer_ = dealer;
    currentSeat_ = dealer;
  }

  void dealFromPbn(const std::string& pbn, int dealer, int vul);

  template <class Container>
  void setDDTable(const Container& ddTable) {
    RELA_CHECK_EQ(ddTable.size(), kNumStrains * kNumPlayers);
    for (int i = 0; i < kNumStrains; ++i) {
      for (int j = 0; j < kNumPlayers; ++j) {
        ddTable_[j * kNumStrains + i] = ddTable[i * kNumPlayers + j];
      }
    }
  }

  uint64_t legalActions() const {
    RELA_CHECK(currentStage() == kStageBidding ||
               currentStage() == kStagePlaying);
    return currentStage() == kStageBidding ? legalBiddingActions()
                                           : legalPlayingActions();
  }

  void step(int act) {
    RELA_CHECK(currentStage() == kStageBidding ||
               currentStage() == kStagePlaying);
    if (currentStage() == kStageBidding) {
      biddingStep(act);
    } else {
      playingStep(act);
    }
  }

  const std::vector<Bid>& biddingHistory() const { return biddingHistory_; }

  uint32_t doubled() const { return doubled_; }

  template <typename Container>
  bool bidFromHistory(const Container& history) {
    RELA_CHECK_EQ(currentStage_, kStageBidding);
    for (std::string b : history) {
      RELA_CHECK_GE(b.size(), 1, "Empty bid!");

      if (b[0] == '(') {
        b = b.substr(1, b.length() - 2);
      }
      const Bid bid(b);
      biddingStep(bid.index());
    }
    return currentStage() == kStagePlaying;
  }

  const std::vector<std::pair<Card, int>>& playingHistory() const {
    return playingHistory_;
  }

  uint64_t playedCards() const { return playedCards_; }

  const std::array<Card, kNumPlayers>& currentTrick() const {
    return currentTrick_;
  }

  const Card& winningCard() const { return winningCard_; }

  const std::array<int, kNumTricks>& winningSeats() const {
    return winningSeats_;
  }

  int numWins(int side) const { return numWins_.at(side); }

  std::string getInfo() const;

 private:
  void resetBiddingStatus();

  void resetPlayingStatus();

  bool isBidLegal(const Bid& bid) const;

  uint64_t legalBiddingActions() const;

  bool isBiddingEnd() const {
    return biddingHistory_.size() >= 4 && numConsecutivePasses_ >= 3;
  }

  bool isPlayLegal(const Card& card) const;

  uint64_t legalPlayingActions() const;

  bool isPlayingEnd() const { return playingHistory_.size() >= kDeckSize; }

  void biddingStep(int act);

  void playingStep(int act);

  std::string pbn_;
  std::array<int, kDeckSize> deal_;
  std::array<Hand, kNumPlayers> hands_;

  int vul_;

  std::array<int, kNumPlayers * kNumStrains> ddTable_;

  std::array<int, kNumPlayers> playerToSeat_;
  std::array<int, kNumPlayers> seatToPlayer_;

  int initialStage_ = kStageDealing;
  int terminalStage_ = kStageScoring;
  int currentStage_ = kStageDealing;

  int dealer_ = kNoSeat;
  int declarer_ = kNoSeat;
  int openingLead_ = kNoSeat;
  int currentSeat_ = kNoSeat;
  Bid contract_;
  std::array<int, 2> scores_ = {0, 0};

  // Bidding status.
  std::vector<Bid> biddingHistory_;
  // Bidding status mask, 0 for normal, 1 for doubled, 2 for redoubled.
  uint32_t doubled_ = 0;
  int highestBidSeat_ = kNoSeat;
  int numConsecutivePasses_ = 0;

  // Playing status.
  std::vector<std::pair<Card, int>> playingHistory_;
  std::array<int, kNumTricks> winningSeats_;

  // Played cards mask.
  uint64_t playedCards_ = 0;
  // Current trick mask.
  std::array<Card, kNumPlayers> currentTrick_;
  int leadingSuit_ = kNoSuit;
  int winningSeat_ = kNoSeat;
  Card winningCard_;
  std::array<int, 2> numWins_ = {0, 0};
};

}  // namespace bridge
