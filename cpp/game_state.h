#pragma once

#include <array>
#include <vector>

#include "auction.h"
#include "bid.h"
#include "bridge_common.h"
#include "card.h"
#include "hand.h"
#include "nlohmann/json.hpp"
#include "playing_sequence.h"
#include "rela/logging.h"
#include "rela/types.h"
#include "util.h"

using json = nlohmann::json;

namespace bridge {

class GameState;

class FeatureExtractor {
 public:
  FeatureExtractor(const GameState& s) : s_(s) {}
  void computeFeature(int currSeat, int currTableIdx, torch::Tensor& s) const;
  static int featureDim() { return _kFeatureDim; }
  static std::vector<Bid> getBidFromFeature(const torch::Tensor& s);
  rela::TensorDict computePartnerInfo(int currSeat) const;

 private:
  const GameState& s_;

  void saveAuctionTo(int seatIdx, int tableIdx, torch::Tensor& s) const;

  static constexpr int _kMaxHistLen = 40;
  static constexpr int _kBidChannels = kAction * _kMaxHistLen;
  static constexpr int _kVulChannels = 2;
  static constexpr int _kChannels =
      kDeck * 3 + _kBidChannels + _kVulChannels + kAction;

  static constexpr int _kMyBidStart = kDeck;
  static constexpr int _kVulStart = _kMyBidStart + _kBidChannels;
  static constexpr int _kAvailStart = _kVulStart + _kVulChannels;
  static constexpr int _kPartnerStart = _kAvailStart + kAction;
  static constexpr int _kOppStart = _kPartnerStart + kDeck;
  static constexpr int _kFeatureDim = _kPartnerStart;
};

class FeatureExtractorBaseline {
 public:
  FeatureExtractorBaseline(const GameState& s) : s_(s) {}
  rela::TensorDict computeBaselineFeature() const;
  rela::TensorDict computeBaselineFeature2(int currSeat, int tableIdx) const;

 private:
  const GameState& s_;
  static constexpr int _kBaselineChannels = 94;
};

class FeatureExtractorOld {
 public:
  FeatureExtractorOld(const GameState& s) : s_(s) {}
  void computeFeature(int currSeat, int currTableIdx, torch::Tensor& s) const;
  static int featureDim() { return _kFeatureDim; }

 private:
  const GameState& s_;
  void saveAuctionTo(int seatIdx, int tableIdx, torch::Tensor& s) const;

  static constexpr int _singleBidChannels = kSpecialBidStart;

  static constexpr int _kMyBidStart = kDeck;
  static constexpr int _kPBidStart = _kMyBidStart + _singleBidChannels;
  static constexpr int _kLeftBidStart = _kMyBidStart + 2 * _singleBidChannels;
  static constexpr int _kRightBidStart = _kMyBidStart + 3 * _singleBidChannels;
  static constexpr int _kDoubledStart = _kMyBidStart + 4 * _singleBidChannels;
  static constexpr int _kVulStart = _kMyBidStart + 5 * _singleBidChannels;
  static constexpr int _kVulChannels = 2;
  static constexpr int _kAvailStart = _kVulStart + _kVulChannels;
  static constexpr int _kFeatureDim = _kAvailStart + kAction;
};

class GameState {
 public:
  GameState(int numTables)
      : featureExtractor_(*this),
        featureExtractorBaseline_(*this),
        featureExtractorOld_(*this) {
    while ((int)auctions_.size() < numTables) {
      Auction auction;
      auctions_.push_back(auction);
      playingSequences_.emplace_back();
    }
    DDTable_.resize(kPlayer * kStrain);
    for (auto& hand : hands_) {
      hand.clear();
    }
  }

  int playerIdx() const { return (tableIdx_ + currentSeat_) % kPlayer; }

  int getCurrentSeat() const { return currentSeat_; }
  int getCurrentTableIdx() const { return tableIdx_; }

  int getDealer() const { return dealer_; }
  int getVul() const { return (int)vul_; }

  void reset(const std::string& pbn, const std::vector<int>& ddTable,
             int dealer, Vulnerability vul) {
    dealer_ = dealer;
    vul_ = vul;

    fillStateFromPBN(pbn);
    fillInDDTable(ddTable);

    swap_ = 0;
    currentSeat_ = dealer_;

    // std::shuffle(std::begin(deal), std::end(deal), rng_);
    std::fill(hcps_.begin(), hcps_.end(), 0);
    // std::cout << "dealing cards" << std::endl;
    for (int i = 0; i < kPlayer; ++i) {
      std::fill(suitStats_[i].begin(), suitStats_[i].end(), 0);
      hands_[i].clear();
      // Fill hands_ and stats from deal
      for (int j = 0; j < kHand; ++j) {
        const Card card(deal_[i * kHand + j]);
        hands_[i].add(card);
        ++suitStats_[i][card.suit()];
        hcps_[i] += HCPMap[card.value()];
      }
    }
    for (size_t i = 0; i < auctions_.size(); i++) {
      auctions_[i] = {};
      // auctions_[i].dealer = dealer_;
      auctions_[i].setDealer(dealer_);
      // auctions_[i].currentSeat = dealer_;
      auctions_[i].setCurrentSeat(dealer_);
    }

    parScore_ = getParScore();
    // int parScore_;
    // getPar(state_, &parScore_);
    // if (parScore_ < 0) {
    //   parScore_ = parScore_ * -1;
    //   // swap sides to make sure par is positive in 1 table setting.
    //   swap_ = 1;
    // }
    // parScore_ = parScore_;
    // std::cout << "done reset" << std::endl;
    reward_ = 0;
    tableIdx_ = 0;
    rawScores_.clear();
    trick2Take_.clear();
  }

  void playingStep(int actionIdx) {
    int pIdx = playerIdx();
    const Card c(actionIdx);
    auto& playingSequence = playingSequences_[tableIdx_];

    const bool isPlayLegal =
        playingSequence.isPlayLegal(c, hands_[currentSeat_]);
    RELA_CHECK(isPlayLegal, "card play ", c.toString(),
               " is illegal for player ", pIdx);
    hands_[currentSeat_].remove(c);
    currentSeat_ = playingSequence.makePlay(c, currentSeat_);
  }

  void biddingStep(int actionIdx) {
    int pIdx = playerIdx();
    Bid bid(actionIdx);
    // std::cout << "in env " << actionIdx << std::endl;
    // bid.fromIdx(actionIdx);
    // std::cout << "Get bid: " << bid.print() << ", raw: " << a << std::endl;
    bool check = isBidLegal(bid);
    RELA_CHECK(check, "Bid ", bid.toString(), " (raw: ", actionIdx,
               ") is illegal for player ", pIdx);
    // if (!check) {
    //   std::cout << "bid " << bid.print() << " (raw: " << actionIdx
    //             << ") is illegal for player " << pIdx << std::endl;
    //   std::cout << printHandAndBidding();
    // }
    // assert(check);
    // std::cout << state_.tableIdx << " " << state_.currentSeat << std::endl;
    // std::cout << "player " << playerIdx << " making bid " <<  bid.print() <<
    // std::endl;
    makeBid(bid);
    currentSeat_ = NEXT_SEAT(currentSeat_);
  }

  bool fillStateFromPBN(const std::string& deal) {
    pbn_ = deal;
    int currentPlayer = SEAT_NORTH;
    int currentSuit = SPADE;
    int count = 0;
    for (int i = 0; i < kPBNLen; i++) {
      char c = pbn_[9 + i];
      if (c == '.') {
        currentSuit = currentSuit - 1;
        continue;
      }
      if (c == ' ') {
        currentSuit = SPADE;
        currentPlayer += 1;
        continue;
      }

      if (ReverseCardMap.find(c) == ReverseCardMap.end()) {
        std::cout << "WARNING \"" << c << "\" (idx: " << i
                  << ", asc: " << (int)c << ") is not in card map" << std::endl;
        std::cout << "deal: " << pbn_ << std::endl;
        return false;
      }

      deal_[count] = currentSuit * kCardsPerSuit + ReverseCardMap.at(c);
      count += 1;
    }
    assert(count == kDeck);
    return true;
  }

  /*
  static std::string convertToPBN(const GameState& state) {
      std::stringstream resultString;
      for (int i = 0; i < kPlayer; i++) {
          for (int j = 0; j < kSuit; j++) {
              for (int k = 0; k < kCardsPerSuit; k++) {
                  if (state.hands[i][j][k] == 1) {
                      resultString << CardMap[k];
                  }
              }
              resultString << ".";
          }
          resultString << " ";
      }
      return resultString.str();
  }*/

  void fillInDDTable(const std::vector<int>& ddTable) {
    assert(ddTable.size() == kStrain * kPlayer);
    int i = 0;
    for (int strain = CLUB; strain < kStrain; strain++) {
      for (int declarer = SEAT_NORTH; declarer < kPlayer; declarer++) {
        DDTable_[declarer * kStrain + strain] = ddTable[i];
        i++;
      }
    }
  }

  std::vector<int> saveDDTable() const {
    std::vector<int> ddTable;
    for (int strain = CLUB; strain < kStrain; strain++) {
      for (int declarer = SEAT_NORTH; declarer < kPlayer; declarer++) {
        ddTable.push_back(DDTable_[declarer * kStrain + strain]);
      }
    }
    return ddTable;
  }

  bool nextTable() {
    int rawNSSeatScore, trick2Take;
    std::tie(rawNSSeatScore, trick2Take) = getRawNSScore(auctions_[tableIdx_]);

    // if (verbose) {
    //  std::cout << "tricks to take is " << tricksToTake << std::endl;
    //}
    // if (swap) {
    //   rawScore = rawScore * -1;
    // }
    rawScores_.push_back(rawNSSeatScore);
    trick2Take_.push_back(trick2Take);

    tableIdx_++;
    currentSeat_ = dealer_;

    if (tableIdx_ == (int)auctions_.size()) {
      if (auctions_.size() == 1) rawScores_.push_back(parScore_);
      reward_ = computeNSReward(rawScores_);

      tableIdx_ = 0;
      // Game has ended.
      return true;
    } else {
      return false;
    }
  }

  float getReward() const { return reward_; }

  float playerRawScore(int playerIdx) const {
    if (tableIdx_ == 1) {
      if (playerIdx % 2 == 0)
        return rawScores_[0];
      else
        return -1 * rawScores_[0];
    } else {
      if (playerIdx % 2 == 0)
        return -1 * rawScores_[1];
      else
        return rawScores_[1];
    }
  }

  int trick2Take(int tableIdx) const {
    // #tricks the declarer takes.
    return trick2Take_[tableIdx];
  }

  int getDeclarer(int tableIdx) const { return auctions_[tableIdx].declarer(); }

  int getContractStrain(int tableIdx) const {
    if (auctions_[tableIdx].lastestBidIdx() == -1) {
      return -1;
    }
    Bid finalContract(auctions_[tableIdx].lastestBidIdx());
    // finalContract.fromIdx(auctions_[tableIdx].lastestBidIdx);
    return finalContract.strain();
  }

  bool biddingTerminal() const {
    return auctions_[tableIdx_].isAuctionEnd() ||
           auctions_[tableIdx_].bidHistory().size() >= 40;
  }

  bool isBidLegal(Bid bid) const {
    const Auction& currentAuction = auctions_[tableIdx_];
    return currentAuction.isBidLegal(bid, currentSeat_);
  }

  void makeBid(Bid bid) { auctions_[tableIdx_].makeBid(bid); }

  void setBidOtherChoices(std::vector<Bid>&& others) {
    auctions_[tableIdx_].setOtherChoices(std::move(others));
  }

  bool playingTerminal() const {
    return playingSequences_[tableIdx_].isPlayEnd();
  }

  std::string printAllHands(int player = -1) const {
    std::stringstream resultString;
    resultString << std::endl
                 << "Seat \u2660   \u2665   \u2666   \u2663   HCP   Actual Hand"
                 << std::endl;
    for (int i = 0; i < kPlayer; i++) {
      if ((player >= 0) && (i != player)) {
        continue;
      }
      resultString << i << "    ";
      for (int j = 0; j < kSuit; j++) {
        resultString << suitStats_[i][kSuit - 1 - j] << "   ";
      }
      resultString << hcps_[i];
      if (hcps_[i] < 10) resultString << " ";
      resultString << "   " << hands_[i].originalHandString() << std::endl;
    }
    return resultString.str();
  }

  std::string printBidding() const {
    std::stringstream ss;

    if (!rawScores_.empty()) {
      assert(auctions_.size() == rawScores_.size());
      assert(auctions_.size() == trick2Take_.size());
    }

    for (int i = 0; i < (int)auctions_.size(); ++i) {
      int dealer = auctions_[i].dealer();
      int declarer = auctions_[i].declarer();
      const auto& currentBidHistory = auctions_[i].bidHistory();
      ss << "Bidd " << i << ": Dealer: " << dealer << " ";
      for (size_t j = 0; j < currentBidHistory.size(); ++j) {
        const Bid& bid = currentBidHistory[j];
        int playerIdx = (dealer + i + j) % kPlayer;
        if (IS_NS(playerIdx)) ss << " " << bid.toString();
        if (IS_EW(playerIdx)) ss << " (" << bid.toString() << ")";
      }
      if (!rawScores_.empty()) {
        ss << " declarer: " << declarer << " trickTaken: " << trick2Take_[i]
           << " rawScore: " << rawScores_[i];
      }
      ss << std::endl;
    }

    return ss.str();
  }

  std::string printHandAndBidding() const {
    std::stringstream ss;

    ss << "Dealer: " << dealer_ << ", Vul: " << vulMap[vul_]
       << " Deal: " << pbn_ << std::endl;
    ss << "Reward: " << reward_ << std::endl;
    ss << "parScore: " << parScore_ << std::endl;

    ss << printAllHands() << std::endl;
    ss << printBidding() << std::endl;

    return ss.str();
  }

  json getJson() const {
    json s;
    s["dealer"] = dealer_;
    s["vul"] = vul_;
    s["vul_str"] = vulMap[vul_];
    s["pbn"] = pbn_;
    s["reward"] = reward_;
    // s["par_score"] = parScore_;
    // s["state_display"] = printAllHands();

    s["bidd"] = json::array();

    for (size_t i = 0; i < auctions_.size(); ++i) {
      int dealer_ = auctions_[i].dealer();
      const auto& currentBidHistory = auctions_[i].bidHistory();
      json biddSeq = json::array();

      for (size_t j = 0; j < currentBidHistory.size(); ++j) {
        const Bid& bid = currentBidHistory[j];
        int playerIdx = (dealer_ + i + j) % kPlayer;
        if (IS_NS(playerIdx)) biddSeq.push_back(bid.toString());
        if (IS_EW(playerIdx)) biddSeq.push_back("(" + bid.toString() + ")");
      }

      s["bidd"][i]["seq"] = biddSeq;

      /*
      const auto& otherChoices = auctions_[i].bidHistoryOtherChoices;

      if (! otherChoices.empty()) {
        json otherSeq = json::array();
        for (size_t j = 0; j < otherChoices.size(); ++j) {
          json candidates = json::array();
          for (size_t k = 0; k < otherChoices[j].size(); ++k) {
            const Bid& bid = otherChoices[j][k];

            json v;
            v["bid"] = bid.print();
            v["prob"] = bid.prob;
            candidates.push_back(v);
          }
          otherSeq.push_back(candidates);
        }

        s["bidd"][i]["otherSeq"] = otherSeq;
      }
      */

      if (rawScores_.size() > i) {
        s["bidd"][i]["trickTaken"] = trick2Take_[i];
        s["bidd"][i]["rawNSScore"] = rawScores_[i];
      }
    }

    // Note that there it is tricky. We want to save as the same order as the
    // loading.
    s["ddt"] = saveDDTable();
    // s["hcp"] = hcps_;

    return s;
  }

  torch::Tensor computeLegalMove2(int currSeat, int tableIdx) const {
    torch::Tensor legalMove = torch::zeros({kAction});

    auto f = legalMove.accessor<float, 1>();
    const Auction& currentAuction = auctions_[tableIdx];

    for (int j = 0; j < kAction; j++) {
      Bid tmpBid(j);
      // tmpBid.fromIdx(j);
      f[j] = currentAuction.isBidLegal(tmpBid, currSeat) ? 1 : 0;
      // std::cout << f[availStart + j];
    }
    return legalMove;
  }

  torch::Tensor computeHandFeature(int currSeat, int tableIdx) const;

  torch::Tensor computePlayedCardsFeature(int tableIdx) const;

  torch::Tensor getCompleteInfoEncoding(int currSeat) const {
    torch::Tensor encoding = torch::zeros({kDeck}, torch::kLong);
    auto f = encoding.accessor<long, 1>();

    // For each card, specify where it is.
    // Order: NESW
    for (size_t k = 0; k < kPlayer; k++) {
      // 0 - self, 1 - left opponent, 2 - partner, 3 - right opponent.
      int relativeEncoding = (k - currSeat + kPlayer) % kPlayer;

      for (int i = 0; i < kSuit; i++) {
        for (int j = 0; j < kCardsPerSuit; j++) {
          if (hands_[k].containsCard({i, j})) {
            // Order: C AKQJ..2, D AKQJ..2, H AKQJ..2, S AKQJ..2
            f[i * kCardsPerSuit + j] = relativeEncoding;
          }
        }
      }
    }

    return encoding;
  }

  torch::Tensor getCompleteInfo() const {
    torch::Tensor encoding = torch::zeros({kPlayer, kDeck});
    auto f = encoding.accessor<float, 2>();

    // For each card, specify where it is.
    // Order: NESW
    for (int k = 0; k < kPlayer; k++) {
      for (int i = 0; i < kSuit; i++) {
        for (int j = 0; j < kCardsPerSuit; j++) {
          f[k][i * kCardsPerSuit + j] =
              static_cast<float>(hands_[k].containsCard({i, j}));
        }
      }
    }

    return encoding;
  }

  void saveHandTo(int seatIdx, torch::Tensor& s) const {
    auto f = s.accessor<float, 1>();

    for (int i = 0; i < kSuit; i++) {
      for (int j = 0; j < kCardsPerSuit; j++) {
        f[i * kCardsPerSuit + j] =
            static_cast<float>(hands_[seatIdx].containsCard({i, j}));
      }
    }
  }

  void saveAvailableBids(int currSeat, int tableIdx, torch::Tensor& s,
                         int offset) const {
    auto f = s.accessor<float, 1>();
    const Auction& currentAuction = auctions_[tableIdx];
    // fill in available bids
    // std::cout << "fill in avail" << std::endl;
    // std::cout << "currentseat" << currentSeat_ << std::endl;
    for (int j = 0; j < kAction; j++) {
      Bid tmpBid(j);
      // tmpBid.fromIdx(j);
      f[offset + j] = currentAuction.isBidLegal(tmpBid, currSeat) ? 1 : 0;
      // std::cout << f[availStart + j];
    }
  }

  torch::Tensor computeAbsAuctionSeqInfo() const {
    // Longest possible bid seq x 2
    torch::Tensor s = torch::zeros({38 * 4 * 2}, torch::kLong);
    s.fill_(-1);
    auto accessor = s.accessor<long, 1>();

    int cnt = 0;

    for (size_t tableIdx = 0; tableIdx < auctions_.size(); ++tableIdx) {
      const Auction& currentAuction = auctions_[tableIdx];
      int bidLen = (int)currentAuction.bidHistory().size();
      for (int i = 0; i < bidLen; i++) {
        int idx = currentAuction.bidHistory().at(i).index();
        accessor[cnt++] = idx;
      }
    }

    return s;
  }

  void computeFeature2(int currSeat, int currTableIdx, torch::Tensor& s) const {
    featureExtractor_.computeFeature(currSeat, currTableIdx, s);
  }

  rela::TensorDict computeBaselineFeature2(int currSeat, int tableIdx) const {
    return featureExtractorBaseline_.computeBaselineFeature2(currSeat,
                                                             tableIdx);
  }

  void computeFeatureOld(int currSeat, int currTableIdx,
                         torch::Tensor& s) const {
    featureExtractorOld_.computeFeature(currSeat, currTableIdx, s);
  }

  rela::TensorDict computePartnerInfo(int currSeat) const {
    return featureExtractor_.computePartnerInfo(currSeat);
  }

  int getParScore() const {
    int par = 0;
    int bestScores[2];
    int bestLevels[2];
    int bestScoresBkp[2];
    int bestLevelsBkp[2];
    for (int declarer = 0; declarer < 2; declarer++) {
      bestScores[declarer] = 0;
      bestLevels[declarer] = -100;
      bestScoresBkp[declarer] = 0;
      bestLevelsBkp[declarer] = -100;
    }
    for (int declarer = 0; declarer < 2; declarer++) {
      bool vul = (IS_NS_VUL(vul_) && IS_NS(declarer)) ||
                 (IS_EW_VUL(vul_) && IS_EW(declarer));
      for (int strain = kStrain - 1; strain >= 0; strain--) {
        int tricksToTake = DDTable_[declarer * kStrain + strain];
        int ctricks = tricksToTake - 6;
        int level = (ctricks - 1) * kStrain + strain;
        if ((level < 0) && (level > bestLevels[declarer])) {
          bestLevels[declarer] = level;
        }
        if (ctricks > 0) {
          int score = getRawScore(strain, ctricks, tricksToTake, 0, vul);
          if (score > bestScores[declarer]) {
            bestScores[declarer] = score;
            bestLevels[declarer] = level;
          } else if (level > bestLevels[declarer]) {
            bestScoresBkp[declarer] = score;
            bestLevelsBkp[declarer] = level;
          }
        }
      }
    }
    int finalLevel = 0;
    // Note this -1 won't be used.
    int finalSide = -1;
    int finalLevelTmp;
    for (int declarer = 0; declarer < 2; declarer++) {
      bool vul = (IS_NS_VUL(vul_) && IS_NS(declarer)) ||
                 (IS_EW_VUL(vul_) && IS_EW(declarer));
      // one side win contract
      if (bestLevels[declarer] > bestLevels[1 - declarer]) {
        par = bestScores[declarer];
        finalSide = declarer;
        finalLevel = bestLevels[declarer];
        // better backup contracts, other
        if (bestLevelsBkp[1 - declarer] > finalLevel) {
          par = bestScoresBkp[1 - declarer];
          finalSide = 1 - declarer;
          finalLevel = bestLevelsBkp[1 - declarer];
        } else {
          // better sacrifices
          int maxSacLevel = bestLevelsBkp[1 - declarer] > 0
                                ? bestLevelsBkp[1 - declarer]
                                : bestLevels[1 - declarer];
          int undertricks = (finalLevel - maxSacLevel) / kStrain + 1;
          // sac assumed to be doubled
          int sacScore = getRawScore(0, -6, undertricks * -1, 1, vul);
          finalLevelTmp = maxSacLevel + kStrain * undertricks;
          if ((sacScore + par > 0) && (finalLevelTmp < kSpecialBidStart)) {
            par = sacScore;
            finalSide = 1 - declarer;
            finalLevel = finalLevelTmp;
          }
        }

        // better backup contracts, self
        if ((finalSide == 1 - declarer) &&
            (bestLevelsBkp[declarer] > finalLevel) &&
            (bestScoresBkp[declarer] > par * -1)) {
          par = bestScoresBkp[declarer];
          finalSide = declarer;
          finalLevel = bestLevelsBkp[declarer];
          // Look for sac again
          int maxSacLevel = bestLevelsBkp[1 - declarer] > 0
                                ? bestLevelsBkp[1 - declarer]
                                : bestLevels[1 - declarer];
          int undertricks = (finalLevel - maxSacLevel) / kStrain + 1;
          int sacScore = getRawScore(0, -6, undertricks * -1, 1, vul);
          finalLevelTmp = maxSacLevel + kStrain * undertricks;
          if ((sacScore + par > 0) && (finalLevelTmp < kSpecialBidStart)) {
            par = sacScore;
            finalSide = 1 - declarer;
            finalLevel = finalLevelTmp;
          }
        }
      }
    }
    return finalSide == 0 ? par : par * -1;
  }

  std::tuple<int, int> getRawNSScore(const Auction& auction) const {
    // penalty for illegal actions
    /*if (auction.illegalPlayer != NO_SEAT) {
      return IS_NS(auction.illegalPlayer) ? -10000 : 10000;
      }*/
    // std::cout << "declarer is " << auction.declarer << std::endl;
    // std::cout << "contract is " << finalContract.print() << std::endl;

    if (auction.declarer() == kNoSeat) {
      return std::make_tuple(0, 0);
    } else {
      int sign = IS_NS(auction.declarer()) ? 1 : -1;
      int tricksToTake =
          DDTable_[auction.declarer() * kStrain + auction.contract().strain()];
      /*
         std::cout << "RAWscore: declarer: " << auction.declarer << ", strain: "
         << finalContract.strain << std::endl;
         std::cout << "RAWscore: DDT: " << std::endl;
         std::cout << printDDTable(DDTable_);
         std::cout << "RAWscore: trickToTake: " << tricksToTake << std::endl;
      */

      int doubled = 0;
      if (auction.isBidDoubled()) doubled = 1;
      if (auction.isBidRedoubled()) doubled = 2;
      bool vul = (IS_NS_VUL(vul_) && IS_NS(auction.declarer())) ||
                 (IS_EW_VUL(vul_) && IS_EW(auction.declarer()));
      int result = sign * getRawScore(auction.contract().strain(),
                                      auction.contract().level(), tricksToTake,
                                      doubled, vul);
      return std::make_tuple(result, tricksToTake);
    }
    //*par = getParScore(state, DDTable);
    // std::cout << "result is " << *result << std::endl;
  }

  friend class FeatureExtractor;
  friend class FeatureExtractorBaseline;
  friend class FeatureExtractorOld;

 private:
  FeatureExtractor featureExtractor_;
  FeatureExtractorBaseline featureExtractorBaseline_;
  FeatureExtractorOld featureExtractorOld_;

  std::array<int, kDeck> deal_;
  int dealer_;
  Vulnerability vul_;
  std::array<Hand, kPlayer> hands_;
  std::array<std::array<int, kSuit>, kPlayer> suitStats_;
  std::array<int, kPlayer> hcps_;
  float reward_;
  std::vector<Auction> auctions_;
  std::vector<PlayingSequence> playingSequences_;
  int swap_ = 0;
  int parScore_;
  int tableIdx_ = 0;
  int currentSeat_;
  std::vector<int> DDTable_;
  std::string pbn_;
  enum Stage { NONE, BIDDING, PLAYING };

  std::vector<int> rawScores_;
  std::vector<int> trick2Take_;
};

}  // namespace bridge
