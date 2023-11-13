#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cpp/game_option.h"
#include "cpp/game_state2.h"
#include "cpp/seat.h"
#include "cpp/util_func.h"
#include "rela/env.h"
#include "rela/logging.h"
#include "rela/types.h"
#include "torch/torch.h"

namespace bridge {

constexpr int kNumActions = kNumBids + kDeckSize;

class DuplicateBridgeEnv : public rela::Env {
 public:
  DuplicateBridgeEnv() { resetTables(); }

  DuplicateBridgeEnv(const DuplicateBridgeEnv& env)
      : database_(env.database_),
        handle_(env.handle_),
        option_(env.option_),
        rng_(env.rng_),
        thread_(env.thread_),
        dataIndex_(env.dataIndex_),
        gameIndex_(env.gameIndex_) {
    games_[0] = std::make_unique<GameState2>(*env.games_[0]);
    games_[1] = std::make_unique<GameState2>(*env.games_[1]);
  }

  DuplicateBridgeEnv(
      const std::unordered_map<std::string, std::string>& params);

  DuplicateBridgeEnv(
      std::shared_ptr<DBInterface> database,
      const std::unordered_map<std::string, std::string>& params);

  bool subgameEnd() const override { return subgameEnd_; }

  bool terminated() const override { return terminated_; }

  bool reset() override {
    return database_ == nullptr ? resetWithoutDatabase() : resetWithDatabase();
  }

  int playerIdx() const override {
    return games_.at(gameIndex_)->currentPlayer();
  }

  std::vector<int> partnerIndices(int player) const override {
    return {partner(player)};
  }

  int maxNumAction() const override { return kNumActions; }

  rela::EnvSpec spec() const override {
    return {-1,
            -1,
            {kNumActions, kNumActions, kNumActions, kNumActions},
            {rela::PlayerGroup::GRP_1, rela::PlayerGroup::GRP_2,
             rela::PlayerGroup::GRP_1, rela::PlayerGroup::GRP_2}};
  }

  rela::TensorDict feature() const override;

  void step(int act) override;

  float playerReward(int player) const override;

  float playerRawScore(int player) const override;

  void setGameRange(int initialStage, int terminalStage) {
    games_[0]->setGameRange(initialStage, terminalStage);
    games_[1]->setGameRange(initialStage, terminalStage);
  }

  int currentStage() const { return games_.at(gameIndex_)->currentStage(); }

  int getIdx() const { return handle_->offset + dataIndex_; }

 private:
  void resetTables() {
    games_[0] = std::make_unique<GameState2>();
    games_[1] = std::make_unique<GameState2>();
    resetSeats();
  }

  void resetSeats() {
    games_[0]->setPlayerSeat(0);
    games_[1]->setPlayerSeat(1);
  }

  bool resetWithoutDatabase();

  bool resetWithDatabase();

  // Vul mask tensor.
  // size = (2,), dtype = float.
  // 0 for self, 1 for opposite.
  torch::Tensor vulTensor() const;

  // Game stage tensor.
  // size = (1,), dtype = long.
  // 0-bidding, 1-playing
  torch::Tensor gameStageTensor() const;

  // Bidding sequence start from relative position of dealer, pre-pad with
  // kBidNull.
  // size = (40, 2), dtype = long.
  // The 1st col is the bid index.
  // The 2nd col is the player's relative seat.
  torch::Tensor biddingSequenceTensor() const;

  // Contract index tensor.
  // size = (1,), dtype = long.
  torch::Tensor contractTensor() const;

  // Doubled status of the bidding.
  // size = (2,), dtype = float.
  // 00 for no double, 01 for doubled, 10 for re-doubled.
  torch::Tensor doubledTensor() const;

  // Declarer relative postion tensor.
  // size = (1,), dtype = long.
  // Relative seat: 0-self, 1-left, 2-partner, 3-right
  torch::Tensor declarerTensor() const;

  // Mask tensor for current situation.
  // size = (4, 52), dtype = float.
  // Position for 1st dim: (self, left, partner, right)
  torch::Tensor situationTensor() const;

  // Current trick tensor.
  // size = (4,), dtype = long.
  // Position for 1st dim: (self, left, partner, right)
  torch::Tensor currentTrickTensor() const;

  // Playing sequence tensor.
  // size = (52, 2), dtype = long.
  // The 1st col is the card index, 52 is the padding index for embedding.
  // The 2nd col is the player's relative seat.
  torch::Tensor playingSequenceTensor() const;

  // Relative winning seat for each trick tensor.
  // size = (13,), dtype = long.
  // Relative seat: 0-self, 1-left, 2-partner, 3-right
  // Pad -1 for unfinished tricks.
  // torch::Tensor winningSeatsTensor() const;

  // Mask for legal actions.
  // if bidding, size = (39,) else if playing, size = (52,), dtype = float.
  torch::Tensor legalActionsTensor() const;

  std::shared_ptr<DBInterface> database_ = nullptr;
  std::shared_ptr<DBInterface::Handle> handle_ = nullptr;

  std::array<std::unique_ptr<GameState2>, 2> games_;
  GameOption option_;

  std::mt19937 rng_;

  int thread_;
  int dataIndex_ = 0;
  int gameIndex_ = 0;

  bool subgameEnd_ = false;
  bool terminated_ = false;

  bool verbose_ = false;
  bool evalMode_ = false;
};

}  // namespace bridge
