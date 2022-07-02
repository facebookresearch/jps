#include "cpp/score.h"

#include <algorithm>
#include <cmath>

namespace bridge {

namespace {

constexpr int kScoreTable[] = {15,   45,   85,   125,  165,  215,  265,  315,
                               365,  425,  495,  595,  745,  895,  1095, 1295,
                               1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995};
constexpr int kScoreTableSize = sizeof(kScoreTable) / sizeof(int);

int computeMadePoints(const Bid& contract, int numOvertricks, uint32_t doubled,
                      bool vul) {
  const int pointsPerTrick =
      (contract.strain() == kClub || contract.strain() == kDiamond) ? 20 : 30;
  int contractPoints = pointsPerTrick * contract.level();
  if (contract.strain() == kNoTrump) {
    contractPoints += 10;
  }
  contractPoints <<= doubled;

  // Compute overtrick points.
  const int pointsPerOvertrick =
      doubled == 0 ? pointsPerTrick : (vul ? 200 * doubled : 100 * doubled);
  const int overtrickPoints = pointsPerOvertrick * numOvertricks;

  int bonus = 0;

  // Compute slam bonus.
  if (contract.level() == 6) {
    bonus += vul ? 750 : 500;
  } else if (contract.level() == 7) {
    bonus += vul ? 1500 : 1000;
  }

  // Compute doubled or redoubled bonus.
  bonus += 50 * doubled;

  // Compute game or part-game bonus.
  if (contractPoints < 100) {
    bonus += 50;
  } else {
    bonus += vul ? 500 : 300;
  }

  return contractPoints + overtrickPoints + bonus;
}

int computeDefeatedPenalty(int numUndertricks, uint32_t doubled, bool vul) {
  if (doubled == 0) {
    return (vul ? 100 : 50) * numUndertricks;
  }
  if (vul) {
    return (300 * numUndertricks - 100) * doubled;
  }
  if (numUndertricks == 1) {
    return 100 * doubled;
  }
  if (numUndertricks == 2) {
    return 300 * doubled;
  }
  return (300 * numUndertricks - 400) * doubled;
}

}  // namespace

int computeDeclarerScore(const Bid& contract, int numTricks, uint32_t doubled,
                         bool vul) {
  const int target = contract.level() + 6;
  if (numTricks >= target) {
    return computeMadePoints(contract, numTricks - target, doubled, vul);
  } else {
    return -computeDefeatedPenalty(target - numTricks, doubled, vul);
  }
}

float computeNormalizedScore(int score1, int score2) {
  const int score = score1 - score2;
  const int sign = score == 0 ? 0 : (score > 0 ? 1 : -1);
  const int absScore = std::abs(score);
  const int p =
      std::upper_bound(kScoreTable, kScoreTable + kScoreTableSize, absScore) -
      kScoreTable;
  return static_cast<float>(sign * p) / static_cast<float>(kScoreTableSize);
}

}  // namespace bridge
