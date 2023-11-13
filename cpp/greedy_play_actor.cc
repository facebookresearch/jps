#include "cpp/greedy_play_actor.h"

#include "cpp/bid.h"
#include "cpp/card.h"
#include "cpp/game_state2.h"
#include "cpp/seat.h"
#include "rela/logging.h"
#include "rela/types.h"

namespace bridge {

namespace {

rela::TensorDict convertModelInput(const rela::TensorDict& obs) {
  constexpr int kBidOffset = kDeckSize;
  constexpr int kVulOffset = kBidOffset + 5 * kNumNormalBids;
  constexpr int kAvaOffset = kVulOffset + 2;
  constexpr int kFeatureSize = kAvaOffset + kNumBids;

  const torch::Tensor& vul = obs.at("vul");
  const torch::Tensor& hand = obs.at("s");
  const torch::Tensor& bid = obs.at("bid");
  torch::Tensor s = torch::zeros({kFeatureSize}, hand.options());
  const auto kBiddingMoveRange = torch::arange(0, kNumBids);
  const torch::Tensor legalMove =
      obs.at("legal_move").index({kBiddingMoveRange});
  auto vAcc = vul.accessor<float, 1>();
  auto hAcc = hand.accessor<float, 2>();
  auto bAcc = bid.accessor<int64_t, 2>();
  auto sAcc = s.accessor<float, 1>();
  auto mAcc = legalMove.accessor<float, 1>();

  // Self hand.
  for (int i = 0; i < kDeckSize; ++i) {
    sAcc[i] = hAcc[0][i];
  }

  // Bidding information.
  int curBid = -1;
  for (int i = 0; i < kMaxBiddingHistory; ++i) {
    const int bid = bAcc[i][0];
    int seat = bAcc[i][1];
    if (seat == kNoSeat) {
      break;
    }
    if (seat == 1 || seat == 2) {
      // Old feature order is [self, partner, left, right].
      seat ^= 3;
    }
    if (bid < kNumNormalBids) {
      curBid = bid;
      sAcc[kBidOffset + seat * kNumNormalBids + bid] = 1.0f;
    } else if (bid > kNumNormalBids) {
      sAcc[kBidOffset + 4 * kNumNormalBids + curBid] = 1.0f;
    }
  }

  // Vul information.
  sAcc[kVulOffset + 0] = vAcc[0];
  sAcc[kVulOffset + 1] = vAcc[1];

  // Available bids.
  for (int i = 0; i < kNumBids; ++i) {
    sAcc[kAvaOffset + i] = mAcc[i];
  }

  return {{"s", s.unsqueeze(0)}, {"legal_move", legalMove.unsqueeze(0)}};
}

rela::TensorDict convertModelOutput(const rela::TensorDict& output,
                                    const torch::Tensor& legalMove) {
  const torch::Tensor piModel = output.at("pi").squeeze();
  auto piModelAcc = piModel.accessor<float, 1>();
  torch::Tensor pi = torch::zeros_like(legalMove);
  auto piAcc = pi.accessor<float, 1>();
  for (int64_t i = 0; i < piModel.numel(); ++i) {
    piAcc[i] = piModelAcc[i];
  }
  return {{"a", output.at("a").squeeze(0)}, {"pi", pi}};
}

}  // namespace

rela::TensorDictFuture GreedyPlayActor::act(rela::TensorDict& obs) {
  obs_ = obs;
  return [this]() { return this->actImpl(); };
}

rela::TensorDict GreedyPlayActor::actImpl() const {
  const int64_t stage = obs_.at("stage").item<int64_t>() + 1;
  return stage == kStageBidding ? biddingAct() : playingAct();
}

rela::TensorDict GreedyPlayActor::biddingAct() const {
  if (models_ != nullptr) {
    const auto modelOutput =
        models_->callDirect("act", convertModelInput(obs_));
    return convertModelOutput(modelOutput, obs_.at("legal_move"));
  }

  const auto& mov = obs_.at("legal_move");
  const Bid contract(obs_.at("contract").item<int64_t>());

  torch::Tensor act;
  torch::Tensor pi = torch::zeros_like(mov);
  auto piAcc = pi.accessor<float, 1>();
  if (contract.type() == kBidNull) {
    for (int i = 0; i < 5; ++i) {
      piAcc[i] = 0.2f;
    }
    act = pi.multinomial(1, /*replacement=*/true);
  } else {
    const Bid bid("P");
    piAcc[bid.index()] = 1.0f;
    act = torch::tensor({static_cast<int64_t>(bid.index())}, torch::kInt64);
  }

  return {{"a", act}, {"pi", pi}};
}

rela::TensorDict GreedyPlayActor::playingAct() const {
  const auto& mov = obs_.at("legal_move");
  auto movAcc = mov.accessor<float, 1>();
  const Bid contract(obs_.at("contract").item<int64_t>());

  torch::Tensor act;
  torch::Tensor pi;

  // Greedy when playing.
  const int strain = contract.strain();
  const auto& play = obs_.at("play");
  auto playAcc = play.accessor<int64_t, 2>();

  // Find win card.
  Card winCard;
  int l = 0;
  for (; l < kDeckSize && playAcc[l][1] != kNoSeat; ++l)
    ;
  int p = l / kNumPlayers * kNumPlayers;
  for (int i = p; i < l; ++i) {
    const Card curCard(playAcc[i][0]);
    if (winCard.suit() == kNoSuit || curCard.greaterThan(winCard, strain)) {
      winCard = curCard;
    }
  }

  Card card;
  if (winCard.suit() != kNoSuit) {
    for (int i = 0; i < kDeckSize; ++i) {
      if (movAcc[i + kNumBids] == 0) {
        continue;
      }
      Card cur(i);
      if (cur.greaterThan(winCard, strain) &&
          (card.suit() == kNoSuit || card.greaterThan(cur, strain))) {
        card = cur;
      }
    }
  }
  if (card.suit() == kNoSuit) {
    pi = mov / mov.sum(-1, /*keepdim=*/true);
    act = pi.multinomial(1, /*replacement=*/true);
  } else {
    pi = torch::zeros_like(mov);
    pi[card.index() + kNumBids] = 1.0;
    act = torch::tensor({static_cast<int64_t>(card.index() + kNumBids)},
                        torch::kInt64);
  }

  return {{"a", act}, {"pi", pi}};
}

}  // namespace bridge
