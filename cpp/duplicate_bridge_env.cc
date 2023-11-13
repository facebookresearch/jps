#include "cpp/duplicate_bridge_env.h"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "cpp/bid.h"
#include "cpp/hand.h"
#include "cpp/score.h"
#include "nlohmann/json.hpp"
#include "rela/types.h"
#include "torch/types.h"

namespace bridge {

DuplicateBridgeEnv::DuplicateBridgeEnv(
    const std::unordered_map<std::string, std::string>& params) {
  extractParams(params, "thread_idx", &thread_, /*mandatory=*/true);
  extractParams(params, "save_feature_history", &option_.saveFeatureHistory,
                /*mandatory=*/false);
  extractParams(params, "fixed_vul", &option_.fixedVul, /*mandatory=*/false);
  extractParams(params, "fixed_dealer", &option_.fixedDealer,
                /*mandatory=*/false);
  extractParams(params, "eval_mode", &evalMode_, /*mandatory=*/false);
  extractParams(params, "verbose", &verbose_, /*mandatory=*/false);

  resetTables();
  int seed;
  if (extractParams(params, "seed", &seed, /*mandatory=*/false)) {
    rng_.seed(seed);
  } else {
    std::random_device r;
    rng_.seed(r() + thread_);
  }
}

DuplicateBridgeEnv::DuplicateBridgeEnv(
    std::shared_ptr<DBInterface> database,
    const std::unordered_map<std::string, std::string>& params)
    : database_(database) {
  extractParams(params, "thread_idx", &thread_, /*mandatory=*/true);
  extractParams(params, "sampler", &option_.sampler, /*mandatory=*/false);
  extractParams(params, "save_feature_history", &option_.saveFeatureHistory,
                /*mandatory=*/false);
  extractParams(params, "fixed_vul", &option_.fixedVul, /*mandatory=*/false);
  extractParams(params, "fixed_dealer", &option_.fixedDealer,
                /*mandatory=*/false);

  if (!extractParams(params, "train_bidding", &option_.trainBidding,
                     /*mandatory=*/false)) {
    option_.trainBidding = false;
  }
  if (!extractParams(params, "train_playing", &option_.trainPlaying,
                     /*mandatory=*/false)) {
    option_.trainPlaying = false;
  }

  extractParams(params, "eval_mode", &evalMode_, /*mandatory=*/false);
  extractParams(params, "verbose", &verbose_, /*mandatory=*/false);

  handle_ = database_->getData(thread_);
  RELA_CHECK_NOTNULL(handle_);

  resetTables();

  int seed;
  if (extractParams(params, "seed", &seed, /*mandatory=*/false)) {
    rng_.seed(seed);
  } else {
    std::random_device r;
    rng_.seed(r() + thread_);
  }
}

bool DuplicateBridgeEnv::resetWithoutDatabase() {
  std::array<int, kDeckSize> deck;
  for (int i = 0; i < kDeckSize; ++i) {
    deck[i] = i;
  }
  std::shuffle(deck.begin(), deck.end(), rng_);

  int dealer = option_.fixedDealer;
  if (dealer == -1) {
    std::uniform_int_distribution<int> distrib(0, kNumPlayers - 1);
    dealer = distrib(rng_);
  }
  int vul = option_.fixedVul;
  if (vul == -1) {
    std::uniform_int_distribution<int> distrib(0, kNumVuls - 1);
    vul = distrib(rng_);
  }

  games_[0]->reset();
  games_[1]->reset();
  resetSeats();
  games_[0]->dealFromDeck(deck, dealer, vul);
  games_[1]->dealFromDeck(deck, dealer, vul);
  subgameEnd_ = false;
  terminated_ = false;

  gameIndex_ = 0;

  return true;
}

bool DuplicateBridgeEnv::resetWithDatabase() {
  RELA_CHECK_NOTNULL(database_.get());
  RELA_CHECK_NOTNULL(handle_.get());
  RELA_CHECK_GT(handle_->size, 0);

  int index = dataIndex_;
  if (option_.sampler == "uniform") {
    std::uniform_int_distribution<int> distrib(0, handle_->size - 1);
    index = distrib(rng_);
  } else if (option_.sampler == "seq") {
    if (dataIndex_ >= handle_->size) {
      return false;
    }
    ++dataIndex_;
  } else {
    return false;
  }

  const auto j = json::parse(handle_->data.at(index));

  int dealer = option_.fixedDealer;
  if (j.find("dealer") != j.end()) {
    dealer = j["dealer"];
  }
  if (dealer == -1) {
    std::uniform_int_distribution<int> distrib(0, kNumPlayers - 1);
    dealer = distrib(rng_);
  }

  int vul = option_.fixedVul;
  if (j.find("vul") != j.end()) {
    vul = j["vul"];
  }
  if (vul == -1) {
    std::uniform_int_distribution<int> distrib(0, kNumVuls - 1);
    vul = distrib(rng_);
  }

  games_[0]->reset();
  games_[1]->reset();
  resetSeats();

  subgameEnd_ = false;
  terminated_ = false;

  const auto pbnIt = j.find("pbn");
  RELA_CHECK(pbnIt != j.end(), "pbn not found in record");
  games_[0]->dealFromPbn(pbnIt.value(), dealer, vul);
  games_[1]->dealFromPbn(pbnIt.value(), dealer, vul);

  const auto ddtIt = j.find("ddt");
  if (ddtIt != j.end()) {
    games_[0]->setDDTable(j["ddt"]);
    games_[1]->setDDTable(j["ddt"]);
  }

  if (option_.trainBidding && !option_.trainPlaying) {
    setGameRange(kStageBidding, kStagePlaying);
  }

  if (!option_.trainBidding && option_.trainPlaying) {
    const auto it = j.find("bidd");
    if (it == j.end()) {
      // No bidding results..
      std::cout << "[" << thread_ << ":" << index
                << "] No bidding info! evalMode_: " << evalMode_ << std::endl;
      std::cout << j << std::endl;
    }
    RELA_CHECK(it != j.end(), "Bidding sequence not found in record.");

    bool bidd0Succ = games_[0]->bidFromHistory(it.value()[0]["seq"]);
    bool bidd1Succ = games_[1]->bidFromHistory(it.value()[1]["seq"]);

    if (!bidd0Succ || !bidd1Succ) {
      // All pass or incomplete bidding history.
      if (games_[0]->currentStage() == kStageBidding ||
          games_[1]->currentStage() == kStageBidding) {
        // Incomplete bidding history.
        std::cout << "Dump incomplete bidding case:" << std::endl;
        std::cout << j << std::endl;
      }
      return reset();
    }
  }

  gameIndex_ = 0;

  return true;
}

void DuplicateBridgeEnv::step(int act) {
  GameState2* game = games_.at(gameIndex_).get();

  if (game->currentStage() == kStagePlaying) {
    RELA_CHECK_GE(act, kNumBids);
    act -= kNumBids;
  }

  game->step(act);
  subgameEnd_ = false;
  terminated_ = false;
  if (game->terminated()) {
    ++gameIndex_;
    subgameEnd_ = true;
    if (gameIndex_ > 1) {
      terminated_ = true;
    }
  }

  if (terminated_ && evalMode_ && verbose_) {
    // Print the game out to check whether it is right.
    std::cout << "Table 0: " << std::endl;
    std::cout << games_[0]->getInfo() << std::endl;
    std::cout << "Table 1: " << std::endl;
    std::cout << games_[1]->getInfo() << std::endl;
  }
}

float DuplicateBridgeEnv::playerReward(int player) const {
  const int side = (player & 1);
  const int score1 = games_[0]->relativeScore(side);
  const int score2 = games_[1]->relativeScore(side);
  const float reward = computeNormalizedScore(score1, score2);
  return reward;
}

float DuplicateBridgeEnv::playerRawScore(int player) const {
  if (!subgameEnd()) {
    return 0;
  }
  const int index = terminated() ? 1 : 0;
  const int side = (games_.at(index)->playerToSeat(player) & 1);
  return games_.at(index)->relativeScore(side);
}

rela::TensorDict DuplicateBridgeEnv::feature() const {
  return {{"vul", vulTensor()},
          {"stage", gameStageTensor()},
          {"bid", biddingSequenceTensor()},
          {"contract", contractTensor()},
          {"doubled", doubledTensor()},
          {"declarer", declarerTensor()},
          {"s", situationTensor()},
          {"trick", currentTrickTensor()},
          {"play", playingSequenceTensor()},
          {"legal_move", legalActionsTensor()}};
}

torch::Tensor DuplicateBridgeEnv::vulTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const int vul = game->vul();
  const int selfSide = (game->currentSeat() & 1);
  const int oppoSide = (selfSide ^ 1);
  return torch::tensor({static_cast<float>((vul >> selfSide) & 1),
                        static_cast<float>((vul >> oppoSide) & 1)},
                       torch::kFloat);
}

torch::Tensor DuplicateBridgeEnv::gameStageTensor() const {
  const int64_t stage =
      games_.at(gameIndex_)->currentStage() == kStageBidding ? 0 : 1;
  return torch::tensor({stage}, torch::kInt64);
}

torch::Tensor DuplicateBridgeEnv::biddingSequenceTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const auto& biddingSeq = game->biddingHistory();
  const int seqLen = biddingSeq.size();
  const int dealer = game->dealer();
  const int self = game->currentSeat();
  const Bid kNull;
  torch::Tensor result = torch::empty({kMaxBiddingHistory, 2}, torch::kInt64);
  auto data = result.accessor<int64_t, 2>();
  if (seqLen <= kMaxBiddingHistory) {
    for (int i = 0; i < seqLen; ++i) {
      data[i][0] = biddingSeq[i].index();
      data[i][1] = relativeSeat(self, (dealer + i) % kNumPlayers);
    }
    for (int i = seqLen; i < kMaxBiddingHistory; ++i) {
      data[i][0] = kNull.index();
      data[i][1] = kNoSeat;
    }
  } else {
    const int offset = seqLen - kMaxBiddingHistory;
    for (int i = 0; i < kMaxBiddingHistory; ++i) {
      const int p = i + offset;
      data[i][0] = biddingSeq[p].index();
      data[i][1] = relativeSeat(self, (dealer + p) % kNumPlayers);
    }
  }
  return result;
}

torch::Tensor DuplicateBridgeEnv::contractTensor() const {
  const Bid& contract = games_.at(gameIndex_)->contract();
  return torch::tensor({static_cast<int64_t>(contract.index())}, torch::kInt64);
}

torch::Tensor DuplicateBridgeEnv::doubledTensor() const {
  const uint32_t doubled = games_.at(gameIndex_)->doubled();
  return torch::tensor(
      {static_cast<float>(doubled & 1), static_cast<float>((doubled >> 1) & 1)},
      torch::kFloat);
}

torch::Tensor DuplicateBridgeEnv::declarerTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const int64_t declarer =
      game->currentStage() == kStageBidding
          ? kNoSeat
          : relativeSeat(game->currentSeat(), game->declarer());
  return torch::tensor({declarer}, torch::kInt64);
}

torch::Tensor DuplicateBridgeEnv::situationTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const auto& hands = game->hands();

  torch::Tensor result = torch::zeros({kNumPlayers, kDeckSize}, torch::kFloat);
  auto data = result.accessor<float, 2>();

  // self hand.
  for (int i = 0; i < kDeckSize; ++i) {
    if (hands.at(game->currentSeat()).containsCard(Card(i))) {
      data[0][i] = 1.0f;
    }
  }

  if (game->currentStage() == kStagePlaying) {
    const int dummySeat = partner(game->declarer());
    // If current seat is dummpy, treat declarer as dummpy.
    const int dummyPos =
        dummySeat == game->currentSeat()
            ? relativeSeat(game->currentSeat(), game->declarer())
            : relativeSeat(game->currentSeat(), dummySeat);

    if (!game->playingHistory().empty()) {
      // dummy hand.
      for (int i = 0; i < kDeckSize; ++i) {
        if (hands.at(dummySeat).containsCard(Card(i))) {
          data[dummyPos][i] = 1.0f;
        }
      }
    }
  }

  return result;
}

torch::Tensor DuplicateBridgeEnv::currentTrickTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const auto& currentTrick = game->currentTrick();
  torch::Tensor result = torch::empty({kNumPlayers}, torch::kInt64);
  auto data = result.accessor<int64_t, 1>();
  for (int i = 0; i < kNumPlayers; ++i) {
    const int pos = relativeSeat(game->currentSeat(), i);
    const Card& card = currentTrick.at(i);
    data[pos] = card.suit() == kNoSuit ? kDeckSize : card.index();
  }
  return result;
}

torch::Tensor DuplicateBridgeEnv::playingSequenceTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const auto& playSeq = game->playingHistory();
  const int seqLen = playSeq.size();
  const int self = game->currentSeat();
  torch::Tensor result = torch::empty({kDeckSize, 2}, torch::kInt64);
  auto data = result.accessor<int64_t, 2>();
  for (int i = 0; i < seqLen; ++i) {
    const auto& it = playSeq[i];
    data[i][0] = it.first.index();
    data[i][1] = relativeSeat(self, it.second);
  }
  for (int i = seqLen; i < kDeckSize; ++i) {
    data[i][0] = kDeckSize;
    data[i][1] = kNoSeat;
  }
  return result;
}

// torch::Tensor DuplicateBridgeEnv::winningSeatsTensor() const {
//   const GameState2* game = games_.at(gameIndex_).get();
//   const auto& winningSeats = game->winningSeats();
//   torch::Tensor result = torch::zeros({kNumTricks}, torch::kInt64);
//   auto data = result.accessor<int64_t, 1>();
//   for (int i = 0; i < kNumTricks; ++i) {
//     data[i] = winningSeats[i] == kNoSeat
//                   ? kNoSeat
//                   : relativeSeat(game->currentSeat(), winningSeats[i]);
//   }
//   return result;
// }

torch::Tensor DuplicateBridgeEnv::legalActionsTensor() const {
  const GameState2* game = games_.at(gameIndex_).get();
  const uint64_t legalActionsMask = game->legalActions();

  const int size = game->currentStage() == kStageBidding ? kNumBids : kDeckSize;
  const int offset = game->currentStage() == kStageBidding ? 0 : kNumBids;

  torch::Tensor result = torch::zeros({maxNumAction()}, torch::kFloat);
  auto data = result.accessor<float, 1>();
  for (int i = 0; i < size; ++i) {
    if ((legalActionsMask >> i) & 1) {
      data[offset + i] = 1.0f;
    }
  }

  return result;
}

}  // namespace bridge
