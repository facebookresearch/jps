#include "game_state.h"

namespace bridge {

torch::Tensor GameState::computeHandFeature(int currSeat,
                                            int /* tableIdx */) const {
  torch::Tensor result = torch::empty({kDeck}, torch::kInt64);
  auto data = result.accessor<int64_t, 1>();
  for (int i = 0; i < kSuit; ++i) {
    for (int j = 0; j < kCardsPerSuit; ++j) {
      data[i * kCardsPerSuit + j] = hands_[currSeat].containsCard({i, j});
    }
  }
  return result;
}

torch::Tensor GameState::computePlayedCardsFeature(int tableIdx) const {
  const auto& cardsPlayed = playingSequences_[tableIdx].cardsPlayed();
  torch::Tensor result = torch::empty({kDeck}, torch::kInt64);
  auto data = result.accessor<int64_t, 1>();
  for (int i = 0; i < kDeck; ++i) {
    data[i] = (cardsPlayed[i] > 0);
  }
  return result;
}

void FeatureExtractor::computeFeature(int currSeat, int currTableIdx,
                                      torch::Tensor& s) const {
  s_.saveHandTo(currSeat, s);
  saveAuctionTo(currSeat, currTableIdx, s);
  s_.saveAvailableBids(currSeat, currTableIdx, s, _kAvailStart);
}

rela::TensorDict FeatureExtractor::computePartnerInfo(int currSeat) const {
  torch::Tensor s = torch::zeros({kSuit * kCardsPerSuit});
  s_.saveHandTo((currSeat + 2) % kPlayer, s);

  torch::Tensor p_hand_hcp = torch::zeros({kSuit * kCardsPerSuit});
  auto f = p_hand_hcp.accessor<float, 1>();

  // FIXME: no access to s_?
  for (size_t i = 0; i < kSuit; i++) {
    for (const auto& c : {'A', 'K', 'Q', 'J'}) {
      auto it = ReverseCardMap.find(c);
      assert(it != ReverseCardMap.end());
      f[i * kCardsPerSuit + it->second] = 1.0;
    }
  }
  return {{"p_hand", s}, {"p_hand_hcp", p_hand_hcp}};
}

void FeatureExtractor::saveAuctionTo(int seatIdx, int tableIdx,
                                     torch::Tensor& s) const {
  auto f = s.accessor<float, 1>();

  // From kMyBidStart
  for (int i = _kMyBidStart; i < _kVulStart; i++) {
    f[i] = 0;
  }
  // std::cout << "auction size : " << auction.bidHistory.size() << std::endl;
  assert(tableIdx <= 1 && seatIdx >= 0 && seatIdx < kPlayer);
  const Auction& currentAuction = s_.auctions_[tableIdx];
  int bidLen = (int)currentAuction.bidHistory().size();

  // Note that the old version assumes we save the feature for the player who
  // just bid.
  // Here we don't assume it is true anymore.

  // int lastIdx = -1;
  int relativeDealer = (s_.dealer_ - seatIdx + kPlayer) % kPlayer;

  for (int i = 0; i < bidLen; i++) {
    int idx = currentAuction.bidHistory().at(i).index();

    int relativePlayer = i + relativeDealer;
    if (relativePlayer >= _kMaxHistLen) {
      break;
    }
    f[_kMyBidStart + relativePlayer * kAction + idx] = 1;
  }

  int seatIsNS = seatIdx % 2 == 0;
  bool isOurVul =
      (IS_NS_VUL(s_.vul_) && seatIsNS) || (IS_EW_VUL(s_.vul_) && !seatIsNS);
  bool isOppVul =
      (IS_NS_VUL(s_.vul_) && !seatIsNS) || (IS_EW_VUL(s_.vul_) && seatIsNS);

  f[_kVulStart] = isOurVul;
  f[_kVulStart + 1] = isOppVul;
}

std::vector<Bid> FeatureExtractor::getBidFromFeature(const torch::Tensor& s) {
  auto f = s.accessor<float, 1>();
  std::vector<Bid> bids;

  int offset = _kMyBidStart;
  bool bidStarted = false;

  for (int i = 0; i < _kMaxHistLen; ++i) {
    int idx = -1;
    for (int j = 0; j < kAction; ++j) {
      if (f[offset + j] > 0.5) {
        idx = j;
        break;
      }
    }

    if (idx == -1) {
      if (!bidStarted && i < 4) {
        bids.push_back(Bid());
        offset += kAction;
        continue;
      } else {
        break;
      }
    }

    bids.push_back(Bid(idx));
    bidStarted = true;

    offset += kAction;
  }

  return bids;
}

rela::TensorDict FeatureExtractorBaseline::computeBaselineFeature() const {
  torch::Tensor s = torch::zeros({kPlayer, _kBaselineChannels});
  torch::Tensor convert = torch::zeros({kPlayer});
  auto stateAccessor = s.accessor<float, 2>();
  auto convertAccessor = convert.accessor<float, 1>();
  for (int p = 0; p < kPlayer; p++) {
    auto f = stateAccessor[(p + s_.tableIdx_) % kPlayer];
    for (int i = 0; i < kSuit; i++) {
      for (int j = 0; j < kCardsPerSuit; j++) {
        f[i * kCardsPerSuit + j] = static_cast<float>(
            s_.hands_[p].containsCard({kSuit - 1 - i, kCardsPerSuit - 1 - j}));
      }
    }
    // PASS + 35 normal bids + bid count
    for (int i = kDeck; i < _kBaselineChannels; i++) {
      f[i] = 0;
    }
    for (int i = _kBaselineChannels - 6; i < _kBaselineChannels - 1; i++) {
      f[i] = 1;
    }
    auto currentAuction = s_.auctions_[s_.tableIdx_];
    // std::cout << "auction size : " << currentAuction.bidHistory.size() <<
    // std::endl;

    int bidLen = (int)currentAuction.bidHistory().size();
    for (int i = 0; i < bidLen; i++) {
      int idx = currentAuction.bidHistory().at(i).index();
      if ((bidLen - i) % 2 == 0) {
        f[_kBaselineChannels - 1] += 1;
        if (f[_kBaselineChannels - 1] > 3) {
          convertAccessor[(p + s_.tableIdx_) % kPlayer] = 1;
          break;
        }
        if (idx == kSpecialBidStart) {
          f[kDeck] = 1;
          if ((f[_kBaselineChannels - 1] > 1)) {
            convertAccessor[(p + s_.tableIdx_) % kPlayer] = 1;
            break;
          }
        } else {
          f[kDeck + idx + 1] = 1;
        }
      }
    }
  }
  return {{"s", s}, {"convert", convert}};
}

rela::TensorDict FeatureExtractorBaseline::computeBaselineFeature2(
    int currSeat, int tableIdx) const {
  torch::Tensor s = torch::zeros({_kBaselineChannels});
  torch::Tensor convert = torch::zeros({1});

  auto f = s.accessor<float, 1>();
  auto convertAccessor = convert.accessor<float, 1>();

  for (int i = 0; i < kSuit; i++) {
    for (int j = 0; j < kCardsPerSuit; j++) {
      f[i * kCardsPerSuit + j] =
          static_cast<float>(s_.hands_[currSeat].containsCard(
              {kSuit - 1 - i, kCardsPerSuit - 1 - j}));
    }
  }
  // PASS + 35 normal bids + bid count
  for (int i = kDeck; i < _kBaselineChannels; i++) {
    f[i] = 0;
  }
  for (int i = _kBaselineChannels - 6; i < _kBaselineChannels - 1; i++) {
    f[i] = 1;
  }
  auto currentAuction = s_.auctions_[tableIdx];
  // std::cout << "auction size : " << currentAuction.bidHistory.size() <<
  // std::endl;

  int bidLen = (int)currentAuction.bidHistory().size();
  for (int i = 0; i < bidLen; i++) {
    int idx = currentAuction.bidHistory().at(i).index();
    if ((bidLen - i) % 2 == 0) {
      f[_kBaselineChannels - 1] += 1;
      if (f[_kBaselineChannels - 1] > 3) {
        convertAccessor[0] = 1;
        break;
      }
      if (idx == kSpecialBidStart) {
        f[kDeck] = 1;
        if ((f[_kBaselineChannels - 1] > 1)) {
          convertAccessor[0] = 1;
          break;
        }
      } else {
        f[kDeck + idx + 1] = 1;
      }
    }
  }
  return {{"s", s}, {"convert", convert}};
}

void FeatureExtractorOld::computeFeature(int currSeat, int currTableIdx,
                                         torch::Tensor& s) const {
  s_.saveHandTo(currSeat, s);
  saveAuctionTo(currSeat, currTableIdx, s);
  s_.saveAvailableBids(currSeat, currTableIdx, s, _kAvailStart);
}

void FeatureExtractorOld::saveAuctionTo(int seatIdx, int tableIdx,
                                        torch::Tensor& s) const {
  auto f = s.accessor<float, 1>();

  // From kMyBidStart
  for (int i = _kMyBidStart; i < _kVulStart; i++) {
    f[i] = 0;
  }
  // std::cout << "auction size : " << auction.bidHistory.size() << std::endl;
  assert(tableIdx <= 1 && seatIdx >= 0 && seatIdx < kPlayer);
  const Auction& currentAuction = s_.auctions_[tableIdx];
  int bidLen = (int)currentAuction.bidHistory().size();

  // Note that the old version assumes we save the feature for the player who
  // just bid. Here we don't assume it is true anymore.

  int lastIdx = -1;
  for (int i = 0; i < bidLen; i++) {
    int idx = currentAuction.bidHistory()[i].index();
    int relativeSeat = (s_.dealer_ + i - seatIdx + kPlayer) % kPlayer;

    if (idx < kSpecialBidStart) {
      lastIdx = idx;
      switch (relativeSeat) {
        case 0:
          // std::cout << "assigning " << idx << " to my" << std::endl;
          f[_kMyBidStart + idx] = 1;
          break;
        case 2:
          // std::cout << "assigning " << idx << " to p" << std::endl;
          f[_kPBidStart + idx] = 1;
          break;
        case 1:
          f[_kRightBidStart + idx] = 1;
          break;
        case 3:
          f[_kLeftBidStart + idx] = 1;
          break;
        default:
          // should never happen.
          throw std::runtime_error("relativeSeat " +
                                   std::to_string(relativeSeat) + " is OOB!");
      }
    } else if (idx > kSpecialBidStart) {
      // std::cout << "assigning double to " << lastIdx << std::endl;
      f[_kDoubledStart + lastIdx] = 1;
    }
  }

  // FIXME: Note that everything is relative, but Vul is absolute,
  // which render it useless in the training. We need to fix.
  f[_kVulStart] = IS_NS_VUL(s_.vul_) ? 1 : 0;
  f[_kVulStart + 1] = IS_EW_VUL(s_.vul_) ? 1 : 0;
}

}  // namespace bridge
