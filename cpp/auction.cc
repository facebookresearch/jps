#include "cpp/auction.h"

#include "rela/logging.h"

namespace bridge {

void Auction::makeBid(const Bid& currentBid) {
  RELA_CHECK_NE(currentBid.type(), kBidNull);
  switch (currentBid.type()) {
    case kBidPass: {
      ++lastConsecutivePasses_;
      break;
    }
    case kBidDouble: {
      isBidDoubled_ = true;
      lastConsecutivePasses_ = 0;
      break;
    }
    case kBidRedouble: {
      isBidDoubled_ = false;
      isBidRedoubled_ = true;
      lastConsecutivePasses_ = 0;
      break;
    }
    case kBidNormal: {
      isBidDoubled_ = false;
      isBidRedoubled_ = false;
      lastConsecutivePasses_ = 0;
      lastestBidIdx_ = currentBid.index();
      highestBidPlayer_ = (dealer_ + bidHistory_.size()) % kNumPlayers;
      break;
    }
  }
  bidHistory_.push_back(currentBid);
  if (isAuctionEnd() && lastestBidIdx_ >= 0) {
    contract_ = Bid(lastestBidIdx_);
    for (size_t i = 0; i < bidHistory_.size(); ++i) {
      int player = (dealer_ + i) % kNumPlayers;
      if (isPartner(player, highestBidPlayer_) &&
          bidHistory_[i].strain() == contract_.strain()) {
        declarer_ = player;
        break;
      }
    }
    RELA_CHECK_NE(declarer_, kNoSeat);
  }

  // bool isNS = IS_NS(highestBidPlayer);
  //
  // // Find who is the declarer.
  // for (size_t i = 0; i < bidHistory.size(); ++i) {
  //   int player = (dealer + i) % kPlayer;
  //   if (IS_NS(player) == isNS && bidHistory[i].strain == contract.strain)
  //   {
  //     declarer = player;
  //     break;
  //   }
  // }
  // assert(declarer != NO_SEAT);
}

}  // namespace bridge
