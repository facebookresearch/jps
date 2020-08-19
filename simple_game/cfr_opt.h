// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 
// 
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <sstream>

#include "rela/env.h"
#include "utils.h"
#include "common.h"

namespace tabular {

namespace cfr {

class InfoSet {
 public:
  InfoSet(std::string key, int player, bool isChancePlayer, int num_action, int seed, float explore_factor = 0.01) 
     : key_(key), player_(player), isChancePlayer_(isChancePlayer), numAction_(num_action), reachPr_(0.0f)
     , sumRegret_(num_action, 0)
     , sumStrategy_(num_action, 0), strategy_(num_action)
     , exploreFactor_(explore_factor), rng_(seed ^ std::hash<std::string>{}(key)), gen_(0, explore_factor) {
       uniform(strategy_);
      // std::cout << "New InfoSet, key: " << key_ << ", num_action = " << numAction_ << ", player: " << player_ << std::endl;
  }

  const std::vector<float> &getStrategy() const { return strategy_; }
  int getPlayer() const { return player_; }

  void purifyStrategy() {
    if (isChancePlayer_)
      return;

    // compute argmax.
    float maxProb = -1.0f;
    int maxAction = 0;
    for (int i = 0; i < numAction_; ++i) {
      if (maxProb < strategy_[i]) {
        maxProb = strategy_[i];
        maxAction = i;
      }
    }

    assert(maxAction >= 0);
    std::fill(strategy_.begin(), strategy_.end(), 0.0f);
    strategy_[maxAction] = 1.0f;
  }

  void randomizePolicy() {
    if (isChancePlayer_)
      return;

    std::uniform_real_distribution<float> gen;
    for (int i = 0; i < numAction_; ++i) {
      strategy_[i] = gen(rng_);
    }
    normalize(strategy_);
  }

  std::string info() const {
    std::stringstream ss;
    auto strategy = computeAvgStrategy();
    ss << "\"" << key_ << "\"[" << player_ << "]: policy: " << printVector(strategy) << ", regret: " << printVector(sumRegret_);
    return ss.str();
  }

  void update(const std::vector<float> &reachProb, const std::vector<float> &regret) {
    if (isChancePlayer_) {
      return;
    }

    // std::cout << key_ << ", Reach: " << printVector(reachProb) << ", Regret: " << printVector(regret) << std::endl;

    if (regret.size() != sumRegret_.size()) {
      std::cout << "Info key: " << key_ << ", num_action = " << numAction_ << std::endl;
      std::cout << "regret.size() = " << regret.size() << ", sumRegret.size() = " << sumRegret_.size() << std::endl;
    }
    assert(regret.size() == sumRegret_.size());
    float prod = 1.0;
    for (int i = 0; i < (int)reachProb.size(); ++i) {
      if (i == player_) continue;
      prod *= reachProb[i];
    }

    reachPr_ += reachProb[player_];
    addMulti(sumRegret_, regret, prod);
  }

  void computeStrategy() {
    // Chance Player won't optimize its strategy.
    if (isChancePlayer_) {
      reachPr_ = 0;
      return;
    }

    addMulti(sumStrategy_, strategy_, reachPr_);
    reachPr_ = 0;

    strategy_ = sumRegret_;
    relu(strategy_);
    if (normalize(strategy_)) return;

    // Exploration
    uniform(strategy_);
    if (exploreFactor_ > 0) {
      for (int i = 0; i < numAction_; ++i) {
        strategy_[i] += gen_(rng_); 
      }
    }
      
    if (exploreFactor_ > 0) {
      relu(strategy_);
      normalize(strategy_);
    }
  }

  void setStrategy(const std::vector<float> &strategy) {
    assert(strategy.size() == strategy_.size());
    strategy_ = strategy;
  }

  std::vector<float> computeAvgStrategy() const {
    auto strategy = sumStrategy_;
    if (normalize(strategy)) {
      for (auto &v : strategy) {
        if (v < 0.001) v = 0;
      }
      normalize(strategy);
    } else {
      uniform(strategy);
    }
    return strategy;
  }

 private:
  std::string key_;
  int player_;
  bool isChancePlayer_;
  int numAction_;
  float reachPr_;
  std::vector<float> sumRegret_;
  std::vector<float> sumStrategy_;
  std::vector<float> strategy_;
  float exploreFactor_;

  mutable std::mt19937 rng_;
  std::normal_distribution<float> gen_;
};

class InfoSets {
 public:
  InfoSets(int seed) : seed_(seed) {
  }

  std::shared_ptr<InfoSet> getInfoSet(const rela::Env &g) {
    auto key = g.infoSet();
    assert(key != "");

    int playerId = g.playerIdx();
    int numAction = g.legalActions().size();
    bool isChancePlayer = g.spec().players[playerId] == rela::PlayerGroup::GRP_NATURE;

    auto it = infoSet_.find(key);
    if (it != infoSet_.end()) return it->second; 

    auto v = std::make_shared<InfoSet>(key, playerId, isChancePlayer, numAction, seed_); 
    infoSet_[key] = v; 
    return v;
  }

  void computeStrategy() {
    for (auto &kv : infoSet_) {
      kv.second->computeStrategy();
    }
  }

  void randomizePolicy() {
    for (auto& kv : infoSet_) {
      kv.second->randomizePolicy();
    }
  }

  void purifyStrategies() {
    for (auto& kv : infoSet_) {
      kv.second->purifyStrategy();
    }
  }

  template <typename Func>
  void setStrategies(Func f) {
    for (auto &kv : infoSet_) {
      auto s = f(kv.first);
      if (s.empty()) continue;
      kv.second->setStrategy(s);
    }
  }

  tabular::Policies getAvgStrategies() const {
    tabular::Policies strategies;
    
    for (const auto &kv : infoSet_) {
      strategies[kv.first] = kv.second->computeAvgStrategy();
    }

    return strategies;
  }

  tabular::Policies getStrategies() const {
    tabular::Policies strategies;
    
    for (const auto &kv : infoSet_) {
      strategies[kv.first] = kv.second->getStrategy();
    }

    return strategies;
  }

  void printAvgStrategy() const {
    std::unordered_map<int, std::string> s;

    for (const auto &kv : infoSet_) {
      std::string &ss = s[kv.second->getPlayer()];

      auto strategy = kv.second->computeAvgStrategy();
      ss += kv.first + ": " + printVector(strategy);
      ss += "\n"; 
    }

    for (const auto &kv : s) {
      std::cout << "Player " << kv.first << " strategy: " << std::endl;
      std::cout << kv.second << std::endl;
    }
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<InfoSet>> infoSet_;
  int seed_;
};


// Game tree. 
class Node {
  public:
    void buildTree(InfoSets &infos, const rela::Env &g) {
      // std::cout << "State: " << g.info() << std::endl;
      auto spec = g.spec();
      int numPlayer = spec.players.size();

      if (g.terminated()) {
        u_.resize(numPlayer, 0.0f);
        for (int i = 0; i < numPlayer; ++i) {
          u_[i] = g.playerReward(i);
        }
        return;
      }

      legalActions_ = g.legalActions();
      int numAction = (int)legalActions_.size();

      info_ = infos.getInfoSet(g);
      children_.resize(numAction);
      u_.resize(numPlayer);

      // Preallocate memory;
      regret_.resize(numAction);
      nextReachPr_.resize(numPlayer);

      for (int i = 0; i < numAction; ++i) {
        std::unique_ptr<rela::Env> g_next = g.clone();
        assert(g_next != nullptr);
        g_next->step(legalActions_[i]);
        children_[i].buildTree(infos, *g_next);
      }
    }

    void cfr(const std::vector<float> &reachPr) {
      // std::cout << "State: " << g.info() << std::endl;
      if (children_.empty()) return;

      const auto& strategy = info_->getStrategy();
      assert(strategy.size() == children_.size());

      int numAction = (int)strategy.size();
      int playerIdx = info_->getPlayer();

      std::fill(u_.begin(), u_.end(), 0.0f);

      nextReachPr_ = reachPr;

      for (int i = 0; i < numAction; ++i) {
        nextReachPr_[playerIdx] *= strategy[i];
        children_[i].cfr(nextReachPr_);
        // Recover.
        nextReachPr_[playerIdx] = reachPr[playerIdx];
        addMulti(u_, children_[i].u(), strategy[i]);
      }

      // Compute regret for playerIdx. 
      for (int i = 0; i < numAction; ++i) {
        regret_[i] = children_[i].u()[playerIdx] - u_[playerIdx];
      }
      info_->update(reachPr, regret_);
    }

    void evaluate() {
      if (children_.empty()) return;

      const auto& strategy = info_->getStrategy();
      assert(strategy.size() == children_.size());

      int numAction = (int)strategy.size();
      std::fill(u_.begin(), u_.end(), 0.0f);
      for (int i = 0; i < numAction; ++i) {
        children_[i].evaluate();
        addMulti(u_, children_[i].u(), strategy[i]);
      }
    }

    std::string printTree(int indent) const {
      if (children_.empty()) return "";

      std::stringstream ss;
      for (int k = 0; k < indent; ++k) {
        ss << " ";
      }
      ss << info_->info() << ", u: " << u_[info_->getPlayer()] << ", reach: " << printVector(nextReachPr_) << std::endl;
      for (const auto &n : children_) {
        auto m = n.printTree(indent + 2);
        if (m != "") {
          ss << m << std::endl;
        }
      }
      return ss.str();
    }

    const std::vector<float> &u() const { return u_; }

  private:
    std::shared_ptr<InfoSet> info_;
    std::vector<int> legalActions_;
    std::vector<Node> children_;
    std::vector<float> u_;

    // Preallocated space.
    std::vector<float> regret_, nextReachPr_;
};

class CFRSolver {
 public:
  CFRSolver(int seed, bool verbose) : infos_(seed), verbose_(verbose) {
  }

  void init(const rela::Env &g) {
    spec_ = g.spec();
    numPlayer_ = spec_.players.size();
    root_.buildTree(infos_, g);
  }

  std::vector<float> run(int num_iteration) {
    std::vector<float> reachPr(numPlayer_, 1.0);

    infos_.randomizePolicy();
    std::vector<float> u(numPlayer_, 0);

    for (int k = 0; k < num_iteration; ++k) {
      root_.cfr(reachPr);

      addMulti(u, root_.u());
      infos_.computeStrategy();

      if (verbose_) {
        std::cout << "Iteration: " << k << std::endl;
        for (int i = 0; i < numPlayer_; ++i) {
          std::cout << "Player " << i << ", type: " << (int)spec_.players[i] << ", expected value: " << u[i] / (k + 1) << std::endl;
        }

        infos_.printAvgStrategy();
      }
    }

    multiply(u, 1.0f / num_iteration);
    return u;
  }

  std::vector<float> evaluate() {
    root_.evaluate();
    return root_.u();
  }

  std::string printTree() const {
    return root_.printTree(0);
  }

  const InfoSets &getInfos() const { return infos_; }
  InfoSets &getInfos() { return infos_; }

 private:
  InfoSets infos_;
  bool verbose_;
  Node root_;
  
  rela::EnvSpec spec_;
  int numPlayer_;
};

} // namespace cfr

} // namespace tabular
