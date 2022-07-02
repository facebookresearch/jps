#pragma once

#include "cpp/bid.h"

namespace bridge {

// Compute score for declarer side.
// Postive for making the contract made points and negative for contract
// defeated penalty.
int computeDeclarerScore(const Bid& contract, int numTricks, uint32_t doubled,
                         bool vul);

// Compute normalized score whoose range in [-1, 1] given the raw score for two
// tables.
float computeNormalizedScore(int score1, int score2);

}  // namespace bridge
