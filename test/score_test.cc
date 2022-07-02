#include "cpp/score.h"

#include <algorithm>

#include "cpp/bid.h"
#include "cpp/card.h"
#include "gtest/gtest.h"

namespace bridge {
namespace {

TEST(ScoreTest, ContractMadeNotVulTest) {
  // Undoubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 30 * i;
    const int bonus1 =
        std::max(0, i - 5) * 500 + (contractPoints1 >= 100 ? 300 : 50);

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 20 * i;
    const int bonus2 =
        std::max(0, i - 5) * 500 + (contractPoints2 >= 100 ? 300 : 50);

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 30 * i + 10;
    const int bonus3 =
        std::max(0, i - 5) * 500 + (contractPoints3 >= 100 ? 300 : 50);

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 30 * numOvertricks;
      const int overtrickPoints2 = 20 * numOvertricks;
      const int overtrickPoints3 = 30 * numOvertricks;
      EXPECT_EQ(
          computeDeclarerScore(contract1, j, /*doubled=*/0, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract2, j, /*doubled=*/0, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract3, j, /*doubled=*/0, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract4, j, /*doubled=*/0, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract5, j, /*doubled=*/0, /*vul=*/false),
          contractPoints3 + overtrickPoints3 + bonus3);
    }
  }

  // Doubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 60 * i;
    const int bonus1 =
        50 + std::max(0, i - 5) * 500 + (contractPoints1 >= 100 ? 300 : 50);

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 40 * i;
    const int bonus2 =
        50 + std::max(0, i - 5) * 500 + (contractPoints2 >= 100 ? 300 : 50);

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 60 * i + 20;
    const int bonus3 =
        50 + std::max(0, i - 5) * 500 + (contractPoints3 >= 100 ? 300 : 50);

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 100 * numOvertricks;
      const int overtrickPoints2 = 100 * numOvertricks;
      const int overtrickPoints3 = 100 * numOvertricks;
      EXPECT_EQ(
          computeDeclarerScore(contract1, j, /*doubled=*/1, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract2, j, /*doubled=*/1, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract3, j, /*doubled=*/1, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract4, j, /*doubled=*/1, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract5, j, /*doubled=*/1, /*vul=*/false),
          contractPoints3 + overtrickPoints3 + bonus3);
    }
  }

  // Redoubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 120 * i;
    const int bonus1 =
        100 + std::max(0, i - 5) * 500 + (contractPoints1 >= 100 ? 300 : 50);

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 80 * i;
    const int bonus2 =
        100 + std::max(0, i - 5) * 500 + (contractPoints2 >= 100 ? 300 : 50);

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 120 * i + 40;
    const int bonus3 =
        100 + std::max(0, i - 5) * 500 + (contractPoints3 >= 100 ? 300 : 50);

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 200 * numOvertricks;
      const int overtrickPoints2 = 200 * numOvertricks;
      const int overtrickPoints3 = 200 * numOvertricks;
      EXPECT_EQ(
          computeDeclarerScore(contract1, j, /*doubled=*/2, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract2, j, /*doubled=*/2, /*vul=*/false),
          contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(
          computeDeclarerScore(contract3, j, /*doubled=*/2, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract4, j, /*doubled=*/2, /*vul=*/false),
          contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(
          computeDeclarerScore(contract5, j, /*doubled=*/2, /*vul=*/false),
          contractPoints3 + overtrickPoints3 + bonus3);
    }
  }
}

TEST(ScoreTest, ContractMadeVulTest) {
  // Undoubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 30 * i;
    const int bonus1 =
        (contractPoints1 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 20 * i;
    const int bonus2 =
        (contractPoints2 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 30 * i + 10;
    const int bonus3 =
        (contractPoints3 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 30 * numOvertricks;
      const int overtrickPoints2 = 20 * numOvertricks;
      const int overtrickPoints3 = 30 * numOvertricks;
      EXPECT_EQ(computeDeclarerScore(contract1, j, /*doubled=*/0, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract2, j, /*doubled=*/0, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract3, j, /*doubled=*/0, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract4, j, /*doubled=*/0, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract5, j, /*doubled=*/0, /*vul=*/true),
                contractPoints3 + overtrickPoints3 + bonus3);
    }
  }

  // Doubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 60 * i;
    const int bonus1 =
        50 + (contractPoints1 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 40 * i;
    const int bonus2 =
        50 + (contractPoints2 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 60 * i + 20;
    const int bonus3 =
        50 + (contractPoints3 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 200 * numOvertricks;
      const int overtrickPoints2 = 200 * numOvertricks;
      const int overtrickPoints3 = 200 * numOvertricks;
      EXPECT_EQ(computeDeclarerScore(contract1, j, /*doubled=*/1, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract2, j, /*doubled=*/1, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract3, j, /*doubled=*/1, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract4, j, /*doubled=*/1, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract5, j, /*doubled=*/1, /*vul=*/true),
                contractPoints3 + overtrickPoints3 + bonus3);
    }
  }

  // Redoubled.
  for (int i = 1; i <= 7; ++i) {
    // Major suits.
    const Bid contract1(i, suitToIndex('S'));
    const Bid contract2(i, suitToIndex('H'));
    const int contractPoints1 = 120 * i;
    const int bonus1 =
        100 + (contractPoints1 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // Minor suits.
    const Bid contract3(i, suitToIndex('D'));
    const Bid contract4(i, suitToIndex('C'));
    const int contractPoints2 = 80 * i;
    const int bonus2 =
        100 + (contractPoints2 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    // No Trump.
    const Bid contract5(i, suitToIndex('N'));
    const int contractPoints3 = 120 * i + 40;
    const int bonus3 =
        100 + (contractPoints3 >= 100 ? 500 : 50) + std::max(0, i - 5) * 750;

    for (int j = i + 6; j <= 13; ++j) {
      const int numOvertricks = j - i - 6;
      const int overtrickPoints1 = 400 * numOvertricks;
      const int overtrickPoints2 = 400 * numOvertricks;
      const int overtrickPoints3 = 400 * numOvertricks;
      EXPECT_EQ(computeDeclarerScore(contract1, j, /*doubled=*/2, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract2, j, /*doubled=*/2, /*vul=*/true),
                contractPoints1 + overtrickPoints1 + bonus1);
      EXPECT_EQ(computeDeclarerScore(contract3, j, /*doubled=*/2, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract4, j, /*doubled=*/2, /*vul=*/true),
                contractPoints2 + overtrickPoints2 + bonus2);
      EXPECT_EQ(computeDeclarerScore(contract5, j, /*doubled=*/2, /*vul=*/true),
                contractPoints3 + overtrickPoints3 + bonus3);
    }
  }
}

TEST(ScoreTest, ContractDefeatedNotVulTest) {
  // Undoubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = 50 * numUndertricks;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/0, /*vul=*/false),
            -penalty);
      }
    }
  }

  constexpr int kPenalty[] = {0, 100, 300, 500, 800};

  // Doubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = (numUndertricks <= 4)
                                ? kPenalty[numUndertricks]
                                : 300 * (numUndertricks - 4) + 800;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/1, /*vul=*/false),
            -penalty);
      }
    }
  }

  // Redoubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = (numUndertricks <= 4)
                                ? (kPenalty[numUndertricks] * 2)
                                : 600 * (numUndertricks - 4) + 1600;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/2, /*vul=*/false),
            -penalty);
      }
    }
  }
}

TEST(ScoreTest, ContractDefeatedVulTest) {
  // Undoubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = 100 * numUndertricks;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/0, /*vul=*/true),
            -penalty);
      }
    }
  }

  constexpr int kPenalty[] = {0, 200, 500, 800, 1100};

  // Doubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = (numUndertricks <= 4)
                                ? kPenalty[numUndertricks]
                                : 300 * (numUndertricks - 4) + 1100;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/1, /*vul=*/true),
            -penalty);
      }
    }
  }

  // Redoubled.
  for (int i = 1; i <= 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      const Bid contract(i, j);
      for (int k = 0; k < i + 6; ++k) {
        const int numUndertricks = i + 6 - k;
        const int penalty = (numUndertricks <= 4)
                                ? kPenalty[numUndertricks] * 2
                                : 600 * (numUndertricks - 4) + 2200;
        EXPECT_EQ(
            computeDeclarerScore(contract, k, /*doubled=*/2, /*vul=*/true),
            -penalty);
      }
    }
  }
}

}  // namespace
}  // namespace bridge
