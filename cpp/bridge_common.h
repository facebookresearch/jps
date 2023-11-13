#pragma once

#include <cassert>
#include <iostream>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <vector>

constexpr int HCPMap[] = {4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr char CardMap[] = {'A', 'K', 'Q', 'J', 'T', '9', '8',
                            '7', '6', '5', '4', '3', '2'};
const std::map<char, int> ReverseCardMap = {
    {'A', 0}, {'K', 1}, {'Q', 2}, {'J', 3},  {'T', 4},  {'9', 5}, {'8', 6},
    {'7', 7}, {'6', 8}, {'5', 9}, {'4', 10}, {'3', 11}, {'2', 12}};
constexpr char SuitMap[] = {'C', 'D', 'H', 'S', 'N'};
const std::vector<std::string> SuitUnicode = {"\u2660", "\u2665", "\u2666",
                                              "\u2663"};
const int NO_SEAT = -1;
const int SEAT_NORTH = 0;
const int SEAT_EAST = 1;
const int SEAT_SOUTH = 2;
const int SEAT_WEST = 3;
const int kPlayer = 4;

constexpr char playerMap[] = {'N', 'E', 'S', 'W'};

const int NO_SUIT = -1;
const int CLUB = 0;
const int DIAMOND = 1;
const int HEART = 2;
const int SPADE = 3;
const int NT = 4;
const int kStrain = 5;

const int kSuit = 4;
const int kCardsPerSuit = 13;
const int kHand = 13;
const int kDeck = kSuit * kCardsPerSuit;
const int kLevel = 7;
const int kSpecialAction = 4;
const int kSpecialBidStart = kStrain * kLevel;
const int kAction = kSpecialBidStart + kSpecialAction;
const int kPBNLen = 67;

const std::vector<std::string> vulMap = {"None", "NS", "EW", "Both"};
enum Vulnerability { VUL_NONE, VUL_NS, VUL_EW, VUL_BOTH, NUM_VULNERABILITY };
enum BidType { BID_NORMAL, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_NOOP };

#define PARTNER(a) (((a) + 2) % kPlayer)
#define NEXT_SEAT(a) (((a) + 1) % kPlayer)
#define PREV_SEAT(a) (((a) + 3) % kPlayer)

#define IS_NS(a) ((a) % 2 == 0)
#define IS_EW(a) ((a) % 2 == 1)
#define IS_NS_VUL(a) ((a) == VUL_NS || (a) == VUL_BOTH)
#define IS_EW_VUL(a) ((a) == VUL_EW || (a) == VUL_BOTH)

// struct Bid {
//   int level = 0;
//   int strain = -1;
//   BidType bidType = BID_NORMAL;
//   float prob = -1.0;
//   bool converted = false;
//   int converted_bid = 0;
//
//   int toIdx() const {
//     switch (bidType) {
//     case BID_PASS:
//       return kSpecialBidStart;
//     case BID_DOUBLE:
//       return kSpecialBidStart + 1;
//     case BID_REDOUBLE:
//       return kSpecialBidStart + 2;
//     case BID_NOOP:
//       return kSpecialBidStart + 3;
//     default:
//       assert(level >= 1 && level <= 7);
//       return (level - 1) * kStrain + strain;
//     }
//   }
//
//   void setPass() {
//     bidType = BID_PASS;
//   }
//
//   void fromIdx(int idx) {
//     if (!(idx >= 0 && idx < kSpecialBidStart + 4))
//       std::cout << idx << std::endl;
//     assert(idx >= 0 && idx < kSpecialBidStart + 4);
//     switch (idx) {
//     case kSpecialBidStart:
//       bidType = BID_PASS;
//       break;
//     case kSpecialBidStart + 1:
//       bidType = BID_DOUBLE;
//       break;
//     case kSpecialBidStart + 2:
//       bidType = BID_REDOUBLE;
//       break;
//     case kSpecialBidStart + 3:
//       bidType = BID_NOOP;
//       break;
//     default:
//       bidType = BID_NORMAL;
//       level = idx / kStrain + 1;
//       strain = idx % kStrain;
//     };
//   }
//
//   bool fromStr(const std::string& s) {
//     if (s == "P" || s == "p") {
//       bidType = BID_PASS;
//       return true;
//     } else if (s == "X" || s == "x") {
//       bidType = BID_DOUBLE;
//       return true;
//     } else if (s == "XX" || s == "xx") {
//       bidType = BID_REDOUBLE;
//       return true;
//     } else {
//       level = s[0] - '0';
//       if (level < 1 || level > 7)
//         return false;
//       for (int i = 0; i < kStrain; i++) {
//         if (std::toupper(s[1]) == SuitMap[i]) {
//           strain = i;
//           return true;
//         }
//       }
//       return false;
//     }
//   }
//
//   std::string print() const {
//     switch (bidType) {
//     case BID_PASS:
//       return "P";
//     case BID_DOUBLE:
//       return "X";
//     case BID_REDOUBLE:
//       return "XX";
//     case BID_NOOP:
//       return "NA";
//     default:
//       std::stringstream ss;
//       ss << level << SuitMap[strain];
//       return ss.str();
//     }
//   }
// };

// struct Auction {
//   int dealer = NO_SEAT;
//   int highestBidPlayer = NO_SEAT;
//   int currentSeat = NO_SEAT;
//
//   int declarer = NO_SEAT;
//   Bid contract;
//
//   std::vector<Bid> bidHistory;
//   std::vector<std::vector<Bid>> bidHistoryOtherChoices;
//
//   int lastestBidIdx = -1;
//   int lastConsecutivePasses = 0;
//   bool isBidDoubled = false;
//   bool isBidRedoubled = false;
//   int illegalPlayer = NO_SEAT;
//   bool isBidLegal(const Bid& currentBid, int currentSeat) const {
//     switch (currentBid.bidType) {
//     // can always pass
//     case BID_PASS:
//       return true;
//     // not doubled or redoubled, cannot double your own contract
//     case BID_DOUBLE:
//       // std::cout <<  highestBidPlayer << " " << currentSeat << std::endl;
//       // std::cout << isBidDoubled << isBidRedoubled << std::endl;
//       return ((highestBidPlayer != NO_SEAT) && !isBidDoubled &&
//       !isBidRedoubled &&
//               ((currentSeat - highestBidPlayer) % 2 != 0));
//     // doubled, not redoubled, can only redouble your own contract
//     case BID_REDOUBLE:
//       return (isBidDoubled && !isBidRedoubled &&
//               (currentSeat - highestBidPlayer) % 2 == 0);
//     case BID_NOOP:
//       return false;
//     // can only bid a higher contract
//     default:
//       return currentBid.toIdx() > lastestBidIdx;
//     };
//   }
//
//   bool isAuctionEnd() const {
//     return (bidHistory.size() >= 4 && lastConsecutivePasses >= 3) ||
//            illegalPlayer != NO_SEAT;
//   }
//
//   void makeBid(const Bid& currentBid) {
//     switch (currentBid.bidType) {
//     case BID_PASS:
//       lastConsecutivePasses += 1;
//       break;
//     case BID_DOUBLE:
//       isBidDoubled = true;
//       lastConsecutivePasses = 0;
//       break;
//     case BID_REDOUBLE:
//       isBidDoubled = false;
//       isBidRedoubled = true;
//       lastConsecutivePasses = 0;
//       break;
//     case BID_NORMAL:
//       isBidDoubled = false;
//       isBidRedoubled = false;
//       lastConsecutivePasses = 0;
//       lastestBidIdx = currentBid.toIdx();
//       highestBidPlayer = (dealer + bidHistory.size()) % kPlayer;
//       // std::cout << "dealer is " << dealer << std::endl;
//       // std::cout << "marking highestBidPlayer" << highestBidPlayer <<
//       std::endl; break;
//     case BID_NOOP:
//       assert(false);
//     };
//     bidHistory.push_back(currentBid);
//
//     if (isAuctionEnd() && lastestBidIdx >= 0) {
//       contract.fromIdx(lastestBidIdx);
//       bool isNS = IS_NS(highestBidPlayer);
//
//       // Find who is the declarer.
//       for (size_t i = 0; i < bidHistory.size(); ++i) {
//         int player = (dealer + i) % kPlayer;
//         if (IS_NS(player) == isNS && bidHistory[i].strain == contract.strain)
//         {
//           declarer = player;
//           break;
//         }
//       }
//       assert(declarer != NO_SEAT);
//     }
//   }
//
//   void setOtherChoices(std::vector<Bid>&& otherChoices) {
//     bidHistoryOtherChoices.push_back(std::move(otherChoices));
//   }
// };

// struct PlayingSequence {
//   int trump = NO_SUIT;
//   int currentSeat = NO_SEAT;
//   int currentWinningSeat = NO_SEAT;
//   Card currentWinningCard;
//   int numCardsPlayed = 0;
//   int currentSuit = NO_SUIT;
//   int nsWinners = 0;
//   int ewWinners = 0;
//   std::vector<short> cardsPlayed;
//
//   PlayingSequence() {
//     cardsPlayed.resize(kDeck);
//   }
//
//   bool isPlayLegal(const Card& c, const Hand& h) const {
//     // already played
//     if (cardsPlayed[c.toIdx()] > 0)
//       return false;
//     // does not have that card
//     if (h[c.suit][c.value] == 0)
//       return false;
//     // beginning of round
//     if (numCardsPlayed % 4 == 0)
//       return true;
//     // checking if player has current suit
//     bool hasSuit = false;
//     for (int i = 0; i < kCardsPerSuit; i++) {
//       if (h[currentSuit][i] == 1 &&
//           cardsPlayed[currentSuit * kCardsPerSuit + i] == 0) {
//         hasSuit = true;
//         break;
//       }
//     }
//     // must follow suit
//     if (hasSuit && c.suit != currentSuit)
//       return false;
//     return true;
//   }
//
//   int makePlay(const Card& c, int seat) {
//     if (numCardsPlayed % 4 == 0) {
//       currentSuit = c.suit;
//       currentWinningSeat = seat;
//       currentWinningCard = c;
//     } else {
//       if (c.compareTo(currentWinningCard, trump)) {
//         currentWinningSeat = seat;
//         currentWinningCard = c;
//       }
//     }
//     numCardsPlayed += 1;
//     cardsPlayed[c.toIdx()] = numCardsPlayed;
//     int nextSeat = NEXT_SEAT(seat);
//     if (numCardsPlayed % 4 == 0) {
//       nextSeat = currentWinningSeat;
//       if (IS_NS(currentWinningSeat)) {
//         nsWinners += 1;
//       } else {
//         ewWinners += 1;
//       }
//     }
//     return nextSeat;
//   }
//
//   bool isPlayEnd() const {
//     return numCardsPlayed == kDeck;
//   }
// };
