// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#pragma once

// #include "game_interface.h"
#include "comm_options.h"
#include "rela/env.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <sstream>

namespace simple {

namespace hanabi {

struct Options {
  int numPlayer = 3;
  // { {2, 1}, {1, 1} } means that there are two colors
  // For color0, there is 2x card0 and 1x card1
  // For color1, there is 1x card0 and 1x card1
  std::vector<std::vector<int>> cards = {{2, 1}, {1, 1}};
  int numHold = 1;    // Max number of cards allowed to hold for each player.
  int initHints = 1;  // #hints initially.
  int initLives = 2;  // init lives
  std::vector<int> seeds;  // Seeds to consider.

  Options() {
    seeds = rela::utils::getIncSeq(100);
  }
};

enum ActionType { INVALID = -1, HINT, PLAY, DISCARD };
enum HintType { INVALID_HINT = -1, SUIT, RANK, NUM_TYPES };

struct Card {
  int suit = -1;
  int rank = -1;

  Card() {
  }
  Card(int s, int r)
      : suit(s)
      , rank(r) {
  }

  bool invalid() const {
    return suit < 0 || rank < 0;
  }
  bool valid() const {
    return !invalid();
  }

  void setInvalid() {
    suit = -1;
    rank = -1;
  }

  std::string info(bool selfcard) const {
    if (invalid())
      return "s-r-";
    if (selfcard)
      return "s?r?";
    else
      return "s" + std::to_string(suit) + "r" + std::to_string(rank);
  }

  std::string hintStr(HintType t) const {
    if (t == SUIT)
      return "s" + std::to_string(suit);
    else if (t == RANK)
      return "r" + std::to_string(rank);
    else
      throw std::runtime_error("HintType invalid!");
  }
};

struct Action {
  ActionType type = INVALID;
  int hintPlayer = -1;
  HintType hintType = INVALID_HINT;
  int cardIdx = -1;

  Action() {
  }

  static Action genHintAction(HintType t, int hintPlayer, int cardIdx) {
    Action a;
    a.type = HINT;
    a.hintType = t;
    a.hintPlayer = hintPlayer;
    a.cardIdx = cardIdx;
    return a;
  }

  static Action genPlayAction(int cardIdx) {
    Action a;
    a.type = PLAY;
    a.cardIdx = cardIdx;
    return a;
  }

  static Action genDiscardAction(int cardIdx) {
    Action a;
    a.type = DISCARD;
    a.cardIdx = cardIdx;
    return a;
  }

  std::string info() const {
    switch (type) {
    case HINT:
      return "h" + std::to_string(hintPlayer) + (hintType == SUIT ? "s" : "r") +
             "_" + std::to_string(cardIdx);
    case PLAY:
      return "p_" + std::to_string(cardIdx);
    case DISCARD:
      return "d_" + std::to_string(cardIdx);
    default:
      return "invalid";
    }
  }
};

class ActionParser {
 public:
  ActionParser(const Options& opt)
      : kMaxHintActions(opt.numPlayer * opt.numHold * NUM_TYPES)
      , kMaxPlayActions(opt.numHold)
      , kMaxDiscardActions(opt.numHold)
      , kNumHold(opt.numHold) {
  }

  Action decode(int action) const {
    assert(action >= 0 && action < maxNumAction());
    Action a;
    if (action < kMaxHintActions) {
      a.type = HINT;
      a.hintPlayer = action / (kNumHold * NUM_TYPES) + 1;
      action %= kNumHold * NUM_TYPES;
      a.cardIdx = action / NUM_TYPES;
      a.hintType = static_cast<HintType>(action % NUM_TYPES);
      return a;
    }

    action -= kMaxHintActions;

    if (action < kMaxPlayActions) {
      a.type = PLAY;
      a.cardIdx = action;
      return a;
    }

    action -= kMaxPlayActions;

    a.type = DISCARD;
    a.cardIdx = action;
    return a;
  }

  int encode(const Action& a) const {
    switch (a.type) {
    case HINT:
      return static_cast<int>(a.hintType) + a.cardIdx * NUM_TYPES +
             (a.hintPlayer - 1) * (kNumHold * NUM_TYPES);
    case PLAY:
      return a.cardIdx + kMaxHintActions;
    case DISCARD:
      return a.cardIdx + kMaxHintActions + kMaxPlayActions;
    default:
      return -1;
    }
  }

  int maxNumAction() const {
    return kMaxHintActions + kMaxPlayActions + kMaxDiscardActions;
  }

 private:
  const int kMaxHintActions, kMaxPlayActions, kMaxDiscardActions, kNumHold;
};

// Simple Hanabi
// Each player has hint / play / discard options.
//   action space:
//      hint: (numPlayer - 1) * numHold (you can only hint one card at a time,
//      if hintPoint = 0, you cannot hint).
//
//      play: numHold (pick a card to play.
//      if no card to play, then you have to discard / hint)
//        if the play is wrong, lose one life, if you lose all lives, the game
//        ends with reward -1 if the play finish the sequence, you get a reward
//        of len(cards) = seqLen.
//
//      discard: numHold (pick a card to discard, you get another card and gain
//      1 hint point)
class SimpleHanabi : public rela::Env {
 public:
  SimpleHanabi(const Options& opt)
      : options_(opt)
      , parser_(opt) {
  }

  bool reset() override {
    allCards_.clear();
    publicActions_.clear();
    return true;
  }

  std::string info() const override {
    std::stringstream ss;
    ss << "InfoSet: " << infoSet() << ", player: " << playerIdx()
       << ", terminal: " << terminated();
    return ss.str();
  }

  int maxNumAction() const override {
    return parser_.maxNumAction();
  }

  std::vector<rela::LegalAction> legalActions() const override {
    if (terminated())
      return {};

    if (allCards_.empty()) {
      // Before random. We just consider a few possible cases.
      return rela::utils::intSeq2intStrSeq(
          rela::utils::getIncSeq(options_.seeds.size()));
    }

    std::vector<Action> actions;
    if (hints_ > 0) {
      for (int i = 0; i < options_.numPlayer; ++i) {
        // Cannot hint self.
        if (i == currPlayer_ - 1)
          continue;
        for (int j = 0; j < options_.numHold; ++j) {
          if (hands_[i][j].valid()) {
            actions.emplace_back(Action::genHintAction(SUIT, i + 1, j));
            actions.emplace_back(Action::genHintAction(RANK, i + 1, j));
          }
        }
      }
    }

    for (int j = 0; j < options_.numHold; ++j) {
      if (hands_[currPlayer_ - 1][j].valid()) {
        actions.emplace_back(Action::genPlayAction(j));
        actions.emplace_back(Action::genDiscardAction(j));
      }
    }

    std::vector<rela::LegalAction> res;
    for (const auto& act : actions) {
      res.emplace_back(parser_.encode(act), act.info());
    }
    return res;
  }

  int playerIdx() const override {
    // natural's turn
    if (allCards_.empty())
      return 0;
    else
      return currPlayer_;
  }

  std::vector<int> partnerIndices(int playerIdx) const override {
    if (playerIdx == 0)
      return {};
    return rela::utils::getIncSeq(options_.numPlayer, 1);
  }

  std::string _info(bool partial) const {
    if (allCards_.empty()) {
      return "s";
    }

    // Current status
    std::string s;

    for (int i = 0; i < options_.numPlayer; ++i) {
      // Skip the current player (who cannot see the card).
      s += "p" + std::to_string(i + 1) + ":";

      bool selfcard = (partial && i == currPlayer_ - 1);
      for (int j = 0; j < options_.numHold; ++j) {
        s += hands_[i][j].info(selfcard);
        s += ",";
      }
    }
    // All public actions + current status.
    s += "||";
    for (const auto& pubAct : publicActions_) {
      s += pubAct + "-";
    }
    // Finally the current top
    s += "||";
    for (size_t suit = 0; suit < currTops_.size(); ++suit) {
      s += "s" + std::to_string(suit) + "t" + std::to_string(currTops_[suit]) +
           ",";
    }

    s += "hints" + std::to_string(hints_) + ",lives" + std::to_string(lives_) +
         ",used" + std::to_string(cardUsed_);
    return s;
  }

  std::string infoSet() const override {
    return _info(true);
  }

  std::string completeCompactDesc() const override {
    return _info(false);
  }

  std::unique_ptr<rela::Env> clone() const override {
    return std::make_unique<SimpleHanabi>(*this);
  }

  void step(int action) override {
    if (allCards_.empty()) {
      // Initialize the game.
      for (int suit = 0; suit < (int)options_.cards.size(); ++suit) {
        for (int rank = 0; rank < (int)options_.cards[suit].size(); ++rank) {
          for (int k = 0; k < options_.cards[suit][rank]; ++k) {
            allCards_.emplace_back(suit, rank);
          }
        }
      }

      std::mt19937 rng(options_.seeds[action]);

      // Shuffle the card.
      std::shuffle(allCards_.begin(), allCards_.end(), rng);
      cardIdx_ = 0;

      // Deal the card.
      hands_.resize(options_.numPlayer);
      for (int i = 0; i < options_.numPlayer; ++i) {
        hands_[i].resize(options_.numHold);
        for (int j = 0; j < options_.numHold; ++j) {
          hands_[i][j] = allCards_[cardIdx_++];
        }
      }
      cardUsed_ = 0;
      currPlayer_ = 1;
      hints_ = options_.initHints;
      lives_ = options_.initLives;

      currTops_ = std::vector<int>(options_.cards.size(), -1);
      return;
    }

    assert(currPlayer_ >= 1 && currPlayer_ <= options_.numPlayer);
    Action a = parser_.decode(action);

    // During the game play.
    std::string pubAct = "p" + std::to_string(currPlayer_) + "_";
    if (a.type == HINT) {
      assert(hints_ > 0);
      // Hints.
      auto hintCard = hands_[a.hintPlayer - 1][a.cardIdx];
      // Note that the public information is the index not the card itself (the
      // player being hinted doesn't know the card).
      pubAct += a.info() + "(" + hintCard.hintStr(a.hintType) + ")";
      publicActions_.push_back(pubAct);
      hints_--;
      _nextPlayer();
      return;
    }

    auto& card = hands_[currPlayer_ - 1][a.cardIdx];
    assert(card.valid());

    // The card is revealed after the action. so we need to put them to pubAct.
    pubAct += a.info() + "(" + card.info(false) + ")";
    publicActions_.push_back(pubAct);

    if (a.type == PLAY) {
      if (card.rank == currTops_[card.suit] + 1) {
        currTops_[card.suit]++;
      } else {
        lives_--;
      }
    } else {
      // Discard card.
      hints_++;
    }

    cardUsed_++;
    // Take another card from the pool
    if (cardIdx_ < (int)allCards_.size()) {
      card = allCards_[cardIdx_++];
    } else {
      // Missing card.
      card.setInvalid();
    }
    _nextPlayer();
  }

  bool terminated() const override {
    if (allCards_.empty())
      return false;

    if (lives_ == 0)
      return true;
    if (cardUsed_ == (int)allCards_.size())
      return true;

    for (int suit = 0; suit < (int)options_.cards.size(); ++suit) {
      if (currTops_[suit] < (int)options_.cards[suit].size() - 1) {
        return false;
      }
    }
    return true;
  }

  bool subgameEnd() const override {
    return terminated();
  }

  float playerReward(int playerIdx) const override {
    if (!terminated() || playerIdx == 0)
      return 0.0f;

    // Failed.
    if (lives_ == 0)
      return -1.0f;

    int score = 0;
    for (const auto& v : currTops_) {
      score += v;
    }
    return score + lives_;
  }

  float playerRawScore(int idx) const override {
    return playerReward(idx);
  }

  std::string action2str(int action) const override {
    if (allCards_.empty()) {
      return "seed=" + std::to_string(options_.seeds[action]);
    }
    Action a = parser_.decode(action);
    return a.info();
  }

  // Include chance.
  rela::EnvSpec spec() const override {
    rela::EnvSpec s;

    // = 0 means that no feature is available.
    s.featureSize = 0;
    // Not usable.
    s.maxActionRound = -1;

    s.maxNumActions.push_back(options_.seeds.size());
    s.players.push_back(rela::PlayerGroup::GRP_NATURE);

    for (int i = 1; i <= options_.numPlayer; ++i) {
      s.maxNumActions.push_back(maxNumAction());
      s.players.push_back(rela::PlayerGroup::GRP_1);
    }

    return s;
  }

  rela::TensorDict feature() const override {
    throw std::runtime_error("Not implemented yet!");
  }

 private:
  int currPlayer_;
  int hints_;
  int lives_;
  std::vector<int> currTops_;

  std::vector<Card> allCards_;
  int cardIdx_;
  int cardUsed_;

  std::vector<std::vector<Card>> hands_;
  std::vector<std::string> publicActions_;

  const Options options_;
  ActionParser parser_;

  void _nextPlayer() {
    currPlayer_++;
    if (currPlayer_ > options_.numPlayer) {
      currPlayer_ = 1;
    }
  }
};

}  // namespace hanabi

}  // namespace simple
