// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "common.h"
#include "rela/env.h"

namespace tabular_utils {

inline std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

inline std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

inline std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
  return ltrim(rtrim(str, chars), chars);
}

inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::string cur;
  for (char ch : s) {
    if (ch != delim) {
      cur.push_back(ch);
    } else {
      result.emplace_back(std::move(cur));
    }
  }

  // The last one.
  result.emplace_back(std::move(cur));
  return result;
}

// A function to load policies drawn in the table form.
inline tabular::Policies loadPolicy(rela::Env &env, const std::string& filename) {
  std::ifstream iFile(filename);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(iFile, line)) {
    lines.push_back(line);
  } 

  // loop backwards to check the last eval iter
  const std::string kStart = "eval iter"; 
  const std::string kRegret = "regret:";

  std::vector<std::pair<int, int>> regrets;

  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string& l = lines[i];

    if (l.size() >= kStart.size() && l.substr(0, kStart.size()) == kStart) {
      auto pos = l.find(kRegret);
      assert(pos != std::string::npos);
      pos += kRegret.size();
      auto j = l.find("(", pos);
      if (j == std::string::npos) {
        j = l.length();
      }

      int regret = std::stoi(l.substr(pos, j - pos));
      regrets.emplace_back(regret, i);
    }
  }
  
  // Find lowest regret.
  auto it = std::min_element(regrets.begin() + 1, regrets.end());
  std::cout << "Found min regret: " << it->first << std::endl;

  it --;
  int start = it->second + 3;

  tabular::Policies policy;

  int j = 0;
  int N = -1;

  std::cout << "Start parsing at line: " << start << std::endl;

  // start parsing.
  while (lines[start][0] == '|') {
    auto items = split(lines[start], '|');
    /*
    for (const auto& item : items) {
      std::cout << "\"" << item << "\"" << " ";
    }
    std::cout << std::endl;
    */
    if (j == 0) {
      N = int(items.size()) - 3;
      std::cout << "N: " << N << std::endl;
    }
    assert(items.size() == N + 3);
    for (int i = 2; i < (int)items.size() - 1; ++i) {
      auto bids = split(trim(items[i]), ' ');
      int c1 = j;
      int c2 = i - 2;

      // std::cout << "c1=" << c1 << ",c2=" << c2 << ": " << bids << std::endl;

      env.reset();
      // First action.
      env.step(c1 + c2 * N); 

      // Going to setup a bunch of policies.
      for (size_t k = 0; k < bids.size(); ++k) {
        int action = env.str2action(bids[k]);
        
        auto legalActions = env.legalActions();
        std::vector<float> pi(legalActions.size(), 0.0);
        int legalActionLoc = -1;
        for (size_t kk = 0; kk < legalActions.size(); ++kk) {
          if (legalActions[kk].first == action) {
            pi[kk] = 1.0;
            legalActionLoc = kk;
            break;
          }
        }
        assert(legalActionLoc != -1);

        auto key = env.infoSet();
        auto res = policy.emplace(key, pi);
        if (! res.second) {
          // There is already a policy. Make sure they are the same.
          const auto& prevPi = res.first->second;
          int prevAction = std::max_element(prevPi.begin(), prevPi.end()) - prevPi.begin();
          if (prevAction != legalActionLoc) {
            std::cout << "Error!! Same infoSet [" << key << "], different action [" 
                      << legalActionLoc << ", " << prevAction << "], legalActions: " << legalActions << std::endl;
            assert(false);
          }
        }

        env.step(action);
      }
      assert(env.terminated());
    }
    start ++;
    j ++;
  } 

  std::cout << "#Policy: " << policy.size() << std::endl;
  return policy;
} 

}  // tabular_utils
