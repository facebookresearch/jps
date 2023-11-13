#pragma once

#include <vector>

#include "cpp/bid.h"
#include "cpp/seat.h"

namespace bridge {

class Auction {
 public:
  Auction() = default;

  int dealer() const { return dealer_; }

  void setDealer(int dealer) { dealer_ = dealer; }

  int currentSeat() const { return currentSeat_; }

  void setCurrentSeat(int currentSeat) { currentSeat_ = currentSeat; }

  int declarer() const { return declarer_; }

  int lastestBidIdx() const { return lastestBidIdx_; }

  const std::vector<Bid>& bidHistory() const { return bidHistory_; }

  bool isBidDoubled() const { return isBidDoubled_; }

  bool isBidRedoubled() const { return isBidRedoubled_; }

  const Bid& contract() const { return contract_; }

  bool isBidLegal(const Bid& currentBid, int currentSeat) const {
    switch (currentBid.type()) {
      case kBidPass: {
        // Can always pass.
        return true;
      }
      case kBidDouble: {
        // Not doubled, not redobuled, cannot double your own contract.
        return highestBidPlayer_ != kNoSeat && !isBidDoubled_ &&
               !isBidRedoubled_ && !isPartner(currentSeat, highestBidPlayer_);
      }
      case kBidRedouble: {
        // Doubled, not redoubled, can only redouble your own contract.
        return isBidDoubled_ && !isBidRedoubled_ &&
               isPartner(currentSeat, highestBidPlayer_);
      }
      case kBidNull: {
        return false;
      }
      default: { return currentBid.index() > lastestBidIdx_; }
    }
  }

  bool isAuctionEnd() const {
    return (bidHistory_.size() >= 4 && lastConsecutivePasses_ >= 3) ||
           illegalPlayer_ != kNoSeat;
  }

  void makeBid(const Bid& currentBid);

  void setOtherChoices(std::vector<Bid>&& otherChoices) {
    bidHistoryOtherChoices_.emplace_back(std::move(otherChoices));
  }

 private:
  int dealer_ = kNoSeat;
  int highestBidPlayer_ = kNoSeat;
  int currentSeat_ = kNoSeat;

  int declarer_ = kNoSeat;
  Bid contract_;

  std::vector<Bid> bidHistory_;
  std::vector<std::vector<Bid>> bidHistoryOtherChoices_;

  int lastestBidIdx_ = -1;
  int lastConsecutivePasses_ = 0;
  bool isBidDoubled_ = false;
  bool isBidRedoubled_ = false;
  int illegalPlayer_ = kNoSeat;
};

}  // namespace bridge
