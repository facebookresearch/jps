#pragma once

#include <array>
#include <string>
#include <vector>

#include "cpp/card.h"

namespace bridge {

constexpr int kPbnLength = 67;

// Accept the following three formats of pbn string:
// 1. [Deal "N:.63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4
//     KQT2.AT.J6542.85"]
// 2. N:.63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4 KQT2.AT.J6542.85
// 3. .63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4 KQT2.AT.J6542.85
// Result deal is represented by the seat for each card.

bool parseDealFromPbn(const std::string& pbn, std::array<int, kDeckSize>& deal);

bool parseDealFromPbn(const std::string& pbn, std::vector<int>& deal);

}  // namespace bridge
