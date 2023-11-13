#pragma once

#include "bridge_common.h"
// #include "game_state.h"
#include <rela/types.h>
#include <rela/utils.h>
#include <iomanip>

namespace bridge {
static int getRawScore(int ctrump, int ctricks, int tricks, int doubled,
                       bool vul) {
  int target = ctricks + 6;
  int overtricks = tricks - target;
  int perOvertrick = 0;
  int perUndertrick = 0;
  int res = 0;
  int score = 0;
  int perTrick = 0;
  int baseScore = 0;
  int bonus = 0;
  int overtricksScore = 0;

  perTrick = ctrump == CLUB || ctrump == DIAMOND ? 20 : 30;
  if (overtricks >= 0) {
    baseScore = perTrick * ctricks;
    bonus = 0;

    if (ctrump == NT) {
      baseScore += 10;
    }
    if (doubled == 1) {
      baseScore *= 2;
      bonus += 50;
    }
    if (doubled == 2) {
      baseScore *= 4;
      bonus += 100;
    }
    if (baseScore >= 100) {
      bonus += vul ? 500 : 300;
    } else {
      bonus += 50;
    }
    if (ctricks == 6) {
      bonus += vul ? 750 : 500;
    } else if (ctricks == 7) {
      bonus += vul ? 1500 : 1000;
    }

    if (doubled == 0) {
      perOvertrick = perTrick;
    } else {
      perOvertrick = vul ? 200 * doubled : 100 * doubled;
    }
    overtricksScore = overtricks * perOvertrick;
    res = baseScore + overtricksScore + bonus;
  } else {
    if (doubled == 0) {
      perUndertrick = vul ? 100 : 50;
      res = overtricks * perUndertrick;
    } else {
      if (overtricks == -1) {
        score = vul ? -200 : -100;
      } else if (overtricks == -2) {
        score = vul ? -500 : -300;
      } else {
        score = 300 * overtricks;
        score += vul ? 100 : 400;
      }
      if (doubled == 2) {
        score *= 2;
      }
      res = score;
    }
  }
  return res;
}

/*
static void fillInDDTableHeuristic(const GameState& state, int DDTable[]) {
    int ddsPredict[] = {15, 18, 21, 24, 26, 28, 30, 32, 34, 36, 39, 42, 45};
    for (int declarer = 0; declarer < 2; declarer++) {
      for (int strain = 0; strain < kStrain; strain++) {
        int z = 7;
        if (strain != NT) {
          z = state.suitStats[declarer][strain] +
              state.suitStats[PARTNER(declarer)][strain];
        } else {
          int zz = 5;
          for (int i = 0; i < kSuit; i++) {
            int tmp = state.suitStats[declarer][i] +
                state.suitStats[PARTNER(declarer)][i];
            if (tmp < zz) zz = tmp;
          }
          z = z + (zz - 5);
        }
        z = z + z / 2 + state.hcps[declarer] + state.hcps[PARTNER(declarer)];
        int tricksToTake = -1;
        for (int i = 0; i < 13; ++i)
        {
          if (ddsPredict[i] > z) {
              tricksToTake = i;
              break;
          }
        }
        if (tricksToTake == -1) tricksToTake = 13;
        DDTable[declarer * kStrain + strain] = tricksToTake;
        //std::cout << tricksToTake << " ";
      }
      //std::cout << std::endl;
    }
}*/

static float computeNSReward(const std::vector<int>& rawScores) {
  int raw = rawScores[0] - rawScores[1];
  // if (rawScores[1] < 0) {
  //  if ((rawScores[0] <= 0) && (raw > 0)) raw = 0;
  //  if (rawScores[0] > 0) raw = rawScores[0];
  //}
  int sign = raw < 0 ? -1 : 1;
  int impTable[] = {15,   45,   85,   125,  165,  215,  265,  315,
                    365,  425,  495,  595,  745,  895,  1095, 1295,
                    1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995};
  int absScore = sign * raw;
  assert(absScore >= 0);
  int imp = -1;
  for (int i = 0; i < 24; ++i) {
    if (impTable[i] > absScore) {
      imp = i;
      break;
    }
  }
  if (imp == -1) imp = 24;

  return sign * imp / 24.0;
}

static std::string printDDTable(const std::vector<int>& DDtable) {
  std::stringstream ss;

  ss << "  ";
  for (const char c : SuitMap) {
    ss << std::setw(5) << c;
  }
  ss << std::endl;

  int cnt = 0;
  for (int j = 0; j < kPlayer; ++j) {
    ss << playerMap[j] << " ";
    for (int i = 0; i < kStrain; ++i) {
      ss << std::setw(5) << DDtable[cnt++];
    }
    ss << std::endl;
  }
  return ss.str();
}

static std::string visualizeState(rela::TensorDict& obs) {
  auto s = rela::utils::get(obs, "baseline_s");
  auto f = s.accessor<float, 1>();
  Hand hand;
  // for (size_t i = 0; i < kSuit; i++) {
  //   for (size_t j = 0; j < kCardsPerSuit; j++) {
  //     int idx = i * kCardsPerSuit + j;
  //     if (f[idx] != 0.0) {
  //       hand.add(Card(idx));
  //     }
  //   }
  // }
  for (int i = 0; i < kSuit * kCardsPerSuit; i++) {
    if (f[i] != 0.0) {
      hand.add(Card(i));
    }
  }
  return hand.toString();
}

}  // namespace bridge
