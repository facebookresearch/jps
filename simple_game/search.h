// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 
// 
#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <chrono>
#include <stack>
#include "rela/env.h"
#include "utils.h"
#include "common.h"

namespace tabular {

namespace search {

class State;
class InfoSet;

using InfoSets = std::vector<std::shared_ptr<InfoSet>>;
using States = std::vector<std::shared_ptr<State>>;

inline bool _isDeltaStrategy(const std::vector<float>& strategy, int action) {
  return std::abs(strategy[action] - 1.0f) < 1e-8; 
}

class InfoSet {
 public:
  InfoSet(std::string key,
          int player,
          bool isChance,
          int num_action,
          const Options& options)
      : key_(key)
      , player_(player)
      , isChance_(isChance)
      , numAction_(num_action)
      , strategy_(num_action)
      , rng_(options.seed ^ std::hash<std::string>{}(key))
      , options_(options)
      , q_(num_action, 0) {
    uniform(strategy_);
    succs_.resize(numAction_);

    if (options_.verbose) {
      std::cout << "New InfoSet, key: " << key_ << ", num_action = " << numAction_ 
                << ", player: " << player_ << std::endl;
    }
  }

  void setDepth(int depth) {
    // This implicitly assumes that depth is public information (since all actions are public).
    depth_ = depth;
  }

  int depth() const { return depth_; }

  void setPhi(int phi) {
    phi_ = phi;
  }

  int phi() const { return phi_; }

  void perturbChance(float sigma) {
    if (!isChance_)
      return;

    uniform(strategy_);
    if (sigma == 0)
      return;

    std::uniform_real_distribution<float> gen;
    for (int i = 0; i < numAction_; ++i) {
      strategy_[i] += gen(rng_) * sigma;
    }
    normalize(strategy_);
  }

  void randomizePolicy() {
    if (isChance_)
      return;

    std::uniform_real_distribution<float> gen;
    for (int i = 0; i < numAction_; ++i) {
      strategy_[i] = gen(rng_);
    }
    normalize(strategy_);
  }

  void perturbPolicy(float sigma) {
    if (isChance_)
      return;

    std::uniform_real_distribution<float> gen;
    for (int i = 0; i < numAction_; ++i) {
      strategy_[i] += gen(rng_) * sigma;
    }
    normalize(strategy_);
  }

  const std::string &key() const {
    return key_;
  }
  const std::vector<float>& strategy() const {
    return strategy_;
  }
  int getPlayer() const {
    return player_;
  }
  int numAction() const {
    return numAction_;
  }
  bool isChance() const {
    return isChance_;
  }
  const std::vector<int> &legalActions() const;

  virtual std::string info() const {
    std::stringstream ss;
    ss << "\"" << key_ << "\"[" << player_
       << "]: policy: " << printVector(strategy_);
    return ss.str();
  }

  // The function will be called once for each h \in I to accumulate information
  // for policy computation.
  void update(const State& completeInfoState);
  float u() const {
    return u_;
  }
  const std::vector<float>& q() const {
    return q_;
  }
  float totalReach() const {
    return totalReach_;
  }

  float residue(int action) const {
    return -u_ + strategy_[action] * q_[action];
  }

  void resetStats() {
    // Clear reachability.
    u_ = 0.0f;
    totalReach_ = 0.0f;
    std::fill(q_.begin(), q_.end(), 0.0f);
  }

  void setStrategy(const std::vector<float>& strategy) {
    if (strategy.size() != strategy_.size()) {
      std::cout << "InfoSet: " << key_ << std::endl;
      std::cout << "current len of strategy: " << strategy_.size()
                << ", len of input strategy: " << strategy.size() << std::endl;
      std::cout << "current strategy: " << strategy_ << std::endl;
      std::cout << "input strategy: " << strategy << std::endl;
    }
    assert(strategy.size() == strategy_.size());
    strategy_ = strategy;
  }

  void setDeltaStrategy(int action) {
    std::fill(strategy_.begin(), strategy_.end(), 0.0f);
    strategy_[action] = 1.0f;
  }

  bool isDeltaStrategy(int action) const {
    return _isDeltaStrategy(strategy_, action);
  }

  const States& states() const {
    return states_;
  }
  const InfoSets& succ(int action) const {
    return succs_[action];
  }
  const InfoSets& allSucc() const {
    return allSucc_;
  }

  void addDownStream(int action, std::shared_ptr<InfoSet> next) {
    auto& succ = succs_[action];

    auto checkExist = [&](const InfoSets &infoSets) {
      // check if the information set has been added before.
      for (const auto& info : infoSets) {
        if (info->key() == next->key()) {
          return true;
        }
      }
      return false;
    }; 

    if (!checkExist(succ)) succ.push_back(next);

    // For nature private action, we might get duplicate.
    if (!checkExist(allSucc_)) allSucc_.push_back(next);
  }

  void addState(std::shared_ptr<State> node) {
    // Assume each node is only visited once.
    // Do not check duplicate.
    states_.push_back(node);
  }

 private:
  std::string key_;
  int player_;
  bool isChance_;
  const int numAction_;
  std::vector<float> strategy_;
  mutable std::mt19937 rng_;
  const Options options_;

  int depth_;

  float u_ = 0.0f;
  std::vector<float> q_;
  float totalReach_ = 0.0f;

  int phi_ = -1;

  // Complete information state.
  States states_;
  // succ(I, a) is a collection of information set.
  std::vector<InfoSets> succs_;
  InfoSets allSucc_;
};

class Result;
class ResultAgg;
class Manager;

using Entry = std::pair<std::string, int>;

struct Result {
  std::vector<Entry> actions;
  float value = 0.0f;
  std::string comment;

  Result(const Entry& e, float v)
      : value(v) {
    actions.push_back(e);
  }

  Result(const std::vector<Entry>& e, float v, std::string c = "")
      : actions(e), value(v), comment(c) {
  }

  Result(float v = 0.0f)
      : value(v) {
  }

  std::string prefix(const Manager &) const;
  std::string info(const Manager &m) const;

  Result& attach(const Entry& prefix, float edge) {
    actions.push_back(prefix);
    value += edge;
    // std::cout << "After Result::addPrefix: " << actions << std::endl;
    return *this;
  }

  std::string key() const {
    std::string s;
    for (const auto& a : actions) {
      if (!s.empty())
        s += " ";
      s += a.first + "/a" + std::to_string(a.second);
    }
    return s;
  }
};

struct ResultAgg {
  std::unordered_map<std::string, Result> results;

  ResultAgg() {
  }

  ResultAgg(float v) {
    Result r(v);
    results[r.key()] = r;
  }

  ResultAgg(const Entry& e, float v) {
    Result r(e, v);
    results[r.key()] = r;
  }

  ResultAgg& addBias(float edge) {
    for (auto& r : results) {
      r.second.value += edge;
    }
    return *this;
  }

  ResultAgg& attach(const Entry& prefix, float edge) {
    std::unordered_map<std::string, Result> newResults;

    for (auto& r : results) {
      const Result& r2 = r.second.attach(prefix, edge);
      newResults[r2.key()] = r2;
    }
    results = newResults;
    return *this;
  }

  ResultAgg& append(const Result& r) {
    results[r.key()] = r;
    return *this;
  }

  ResultAgg& append(const ResultAgg& r) {
    results.insert(r.results.begin(), r.results.end());
    return *this;
  }

  std::string info(const Manager &, bool sortByValue = true) const;

  Result getBest() const {
    Result best(-std::numeric_limits<float>::max());
    for (const auto& rr : results) {
      // std::cout << rr.second.info() << std::endl;
      if (best.value < rr.second.value) {
        best = rr.second;
      }
    }

    return best;
  }
};

struct Analysis {
  ResultAgg terms;
  ResultAgg reachability;

  void compareReach(const Analysis& a1) const {
    const auto& reach = reachability;
    const auto& reach1 = a1.reachability;

    // Compare the difference of the two reachability.
    int checkedReach = 0;
    int checkedFailedReach = 0;

    std::cout << "Reachability discrepency: gt.size(): " << reach.results.size() 
              << ", est.size(): " << reach1.results.size() << std::endl;
    for (const auto &r : reach.results) {
      auto it = reach1.results.find(r.first);
      if (it == reach1.results.end()) {
        std::cout << "Error! " << r.first << " does not have a match.. " << std::endl;
        continue;
      }
      assert(it != reach1.results.end());

      float gt = r.second.value;
      float est = it->second.value;

      if (std::abs(gt - est) >= 1e-4) {
        std::cout << r.first << ", gt = " << gt << ", est = " << est << ", comment: " << it->second.comment;
        std::cout << std::endl; 
        checkedFailedReach ++;
      }
      checkedReach ++;
    }
    std::cout << "Reach check: err: " << checkedFailedReach << "/" << checkedReach << std::endl;
  }
};

class Manager;

// Game tree.
class State {
 public:
  State(int depth, const std::string& key)
      : depth_(depth), key_(key) {
  }

  void buildTree(std::shared_ptr<State> own, Manager&, const rela::Env& g, bool keepEnvInState = false);

  void propagate(float reach) {
    // if (infos.getOptions().verbose) {
    // std::cout << "In State" << std::endl;
    // }
    if (hasAlterReach_) {
      std::cout << "Error! alterReach should not be set. alterReach: " << alterReach_ 
                << ", totalReach: " << totalReach_ << std::endl;
    }
    assert(!hasAlterReach_);

    totalReach_ = reach;

    if (children_.empty())
      return;

    const auto& pi = info_->strategy();
    assert(pi.size() == children_.size());

    int numAction = (int)pi.size();

    std::fill(u_.begin(), u_.end(), 0.0f);
    for (int i = 0; i < numAction; ++i) {
      children_[i]->propagate(reach * pi[i]);
      addMulti(u_, children_[i]->u(), pi[i]);
    }

    // Update infoSet related information.
    info_->update(*this);
  }

  void setAlterReach(float alterReach) {
    hasAlterReach_ = true;
    alterReach_ = alterReach;
  }

  void clearAlterReach() {
    hasAlterReach_ = false;
    alterReach_ = totalReach_;
  }

  bool hasAlterReach() const { return hasAlterReach_; }

  std::string printTree(int indent) const {
    if (children_.empty())
      return "";

    std::stringstream ss;
    for (int k = 0; k < indent; ++k) {
      ss << " ";
    }
    ss << info_->info() << ", u: " << u()[info_->getPlayer()]
       << ", totalReach: " << totalReach() << std::endl;
    for (const auto& n : children_) {
      auto m = n->printTree(indent + 2);
      if (m != "") {
        ss << m << std::endl;
      }
    }
    return ss.str();
  }

  int numAction() const {
    return (int)children_.size();
  }
  int numPlayer() const {
    return (int)u_.size();
  }

  const std::vector<float>& u() const {
    return u_;
  }
  float totalReach() const {
    return totalReach_;
  }

  float alterReach() const {
    assert(hasAlterReach_);
    return alterReach_;
  }

  const State& child(int i) const {
    return *children_[i];
  }
  State& child(int i) {
    return *children_[i];
  }
  const State* parent() const {
    return parent_.get();
  }
  int parentActionIdx() const { return parentActionIdx_; }

  std::shared_ptr<State> childSharedPtr(int i) const {
    return children_[i];
  }
  const InfoSet& infoSet() const {
    return *info_;
  }
  std::shared_ptr<InfoSet> infoSetSharedPtr() const {
    return info_;
  }

  const std::string& key() const {
    return key_;
  }
  const rela::Env* env() const {
    return env_.get();
  }
  const std::vector<int> &legalActions() const { return legalActions_; }

 private:
  const int depth_;
  const std::string key_;

  // In some cases we might want to keep an Env for debugging purpose.
  std::unique_ptr<rela::Env> env_;
  std::vector<int> legalActions_;

  std::shared_ptr<InfoSet> info_;
  std::shared_ptr<State> parent_;
  int parentActionIdx_ = -1;

  States children_;

  Options options_;

  std::vector<float> u_;
  float totalReach_ = 1.0f;

  bool hasAlterReach_ = false;
  float alterReach_ = 1.0f;
};

class Manager {
 public:
  Manager(const Options& options)
      : options_(options) {
  }

  std::shared_ptr<InfoSet> getInfoSet(const rela::Env& g);

  std::shared_ptr<State> getState(const rela::Env& g) const {
    auto it = states_.find(g.completeCompactDesc());
    assert(it != states_.end());
    return it->second;
  }

  void resetStats();

  void randomizePolicy() {
    for (auto& kv : infoSet_) {
      kv.second->randomizePolicy();
    }
  }

  void perturbPolicy(float sigma) {
    for (auto& kv : infoSet_) {
      kv.second->perturbPolicy(sigma);
    }
  }

  void perturbChance(float sigma) {
    for (auto& kv : infoSet_) {
      kv.second->perturbChance(sigma);
    }
  }

  template <typename Func>
  void setStrategies(Func f) {
    for (auto& kv : infoSet_) {
      auto s = f(kv.first);
      if (s.empty())
        continue;
      kv.second->setStrategy(s);
    }
  }

  void addState(std::shared_ptr<State> s) {
    // Also save it to complete state table.
    states_[s->key()] = s;

    auto info = s->infoSetSharedPtr();
    while ((int)infoSetByDepth_.size() <= info->depth()) {
      infoSetByDepth_.emplace_back();
    }
    infoSetByDepth_[info->depth()].push_back(info);
  }

  void printInfoSetTree() const {
    for (const auto& kv : infoSet_) {
      const auto& infoSet = *kv.second;
      std::cout << "InfoSetKey: " << kv.first 
                << ", #states: " << infoSet.states().size() << std::endl;
      assert(kv.first == infoSet.key());

      std::cout << "  States: ";
      for (const auto& s : infoSet.states()) {
        std::cout << s->key() << ", ";
      }
      std::cout << std::endl;
       
      for (int a = 0; a < infoSet.numAction(); a++) {
        std::cout << "  a=" << a << ": ";
        for (const auto& n : infoSet.succ(a)) {
          std::cout << n->key() << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  void printStrategy() const {
    std::unordered_map<int, std::stringstream> s;

    for (const auto& kv : infoSet_) {
      const auto& infoSet = *kv.second;
      if (infoSet.numAction() > 0 && infoSet.totalReach() > 0) {
        auto& ss = s[infoSet.getPlayer()];
        ss << kv.first << ", reach: " << infoSet.totalReach() << std::endl;
        const auto& legalActions = infoSet.legalActions();
        const auto& strategy = infoSet.strategy();

        for (int i = 0; i < (int)legalActions.size(); ++i) { 
          ss << "  " << legalActions[i] << ": " << strategy[i] << std::endl;
        }
      }
    }

    for (const auto& kv : s) {
      std::cout << "Player " << kv.first << " strategy " << std::endl;
      std::cout << kv.second.str() << std::endl;
    }
  }

  const Options& getOptions() const {
    return options_;
  }
  InfoSet& operator[](const std::string& key) {
    auto it = infoSet_.find(key);
    assert(it != infoSet_.end());
    assert(it->second != nullptr);
    return *it->second;
  }

  const InfoSet& operator[](const std::string& key) const {
    auto it = infoSet_.find(key);
    assert(it != infoSet_.end());
    assert(it->second != nullptr);
    return *it->second;
  }

  InfoSet* infoSet(const std::string& key) {
    auto it = infoSet_.find(key);
    if (it == infoSet_.end())
      return nullptr;
    if (it->second == nullptr)
      return nullptr;
    return it->second.get();
  }

  const InfoSet* infoSet(const std::string& key) const {
    auto it = infoSet_.find(key);
    if (it == infoSet_.end())
      return nullptr;
    if (it->second == nullptr)
      return nullptr;
    return it->second.get();
  }

  int maxDepth() const {
    return infoSetByDepth_.size() - 1;
  }
  
  const InfoSets &getInfoSetsByDepth(int depth) const {
    return infoSetByDepth_[depth];
  }

  std::vector<std::string> allInfoSetKeys() const {
    std::vector<std::string> keys;
    keys.reserve(infoSet_.size());

    for (const auto& k2v : infoSet_) {
      keys.push_back(k2v.first);
    }

    return keys;
  }

  int numInfoSets() const {
    return infoSet_.size();
  }
  int numActionableInfoSets() const {
    return numActionableInfoSets_;
  }
  int numStates() const {
    return states_.size();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<InfoSet>> infoSet_;
  std::unordered_map<std::string, std::shared_ptr<State>> states_;
  std::vector<InfoSets> infoSetByDepth_;
  const Options options_;

  int numActionableInfoSets_ = 0;
};

inline InfoSets combineInfoSets(const InfoSets& set1, const InfoSets& set2) {
  InfoSets result = set1;
  for (const auto &infoSet2 : set2) {
    bool hasOne = false;
    for (const auto &infoSet1 : set1) {
      if (infoSet1->key() == infoSet2->key()) {
        hasOne = true;
        break;
      }
    } 
    if (!hasOne) {
      result.push_back(infoSet2);
    }
  }
  return result;
}

class Solver {
 public:
  Solver(const tabular::Options& options)
      : manager_(options)
      , options_(options)
      , rng_(options.seed) {
  }

  void init(const rela::Env& g, bool keepEnvInState = false) {
    spec_ = g.spec();
    auto players = g.spec().players;
    numPlayer_ = players.size();
    root_ = std::make_shared<State>(0, g.completeCompactDesc());
    std::cout << "Start Building tree .." << std::endl;
    root_->buildTree(root_, manager_, g, keepEnvInState);
  }

  void loadPolicies(const tabular::Policies& policies) {
    int numLoadedInfoSets = 0;
    for (const auto& kv : policies) {
      auto* infoSet = manager_.infoSet(kv.first);
      if (infoSet == nullptr) {
        std::cout << "Invalid infoSet: \"" << infoSet << "\"" << std::endl;
        continue;
      }
      numLoadedInfoSets ++;
      infoSet->setStrategy(kv.second);
    }
    if (numLoadedInfoSets < manager_.numActionableInfoSets()) {
      std::cout << "Warning! #loaded policies [" << numLoadedInfoSets << "] < " 
                << "#actionable policies " << manager_.numActionableInfoSets() << std::endl;
    }
  }

  void loadPolicies(const std::string& filename) {
    std::cout << "Opening " << filename << std::endl;
    std::ifstream iFile(filename);
    if (iFile.is_open()) {
      while (!iFile.eof()) {
        std::string infoSetKey;
        iFile >> infoSetKey;
        auto* infoSet = manager_.infoSet(infoSetKey);

        if (infoSet == nullptr) {
          std::cout << "Invalid infoSet: \"" << infoSet << "\"" << std::endl;
          continue;
        }

        std::vector<float> pi(infoSet->numAction());
        for (int i = 0; i < infoSet->numAction(); ++i) {
          iFile >> pi[i];
        }
        infoSet->setStrategy(pi);
        std::cout << "loaded info: \"" << infoSetKey << "\", pi: " << pi
                  << std::endl;
      }
    } else {
      std::cout << "Failed to open " << filename << std::endl;
    }
  }

  std::vector<float> runSearch(int playerIdx,
                               int numSamples,
                               int num_iteration) {
    assert(root_->infoSet().isChance());
    float lastBest = 0.0f;

    // Initialize sampledDepth. 
    if (options_.maxDepth > 0) {
      // Random shuffle an order of [1, manager_.maxDepth() - options_.maxDepth].
      sampledDepth_ = rela::utils::getIncSeq(manager_.maxDepth() - options_.maxDepth, 1);
      std::shuffle(sampledDepth_.begin(), sampledDepth_.end(), rng_);
    }

    for (int k = 0; k < num_iteration; ++k) {
      if (options_.perturbChance > 0) {
        manager_.perturbChance(options_.perturbChance);
      }

      if (options_.perturbPolicy > 0) {
        manager_.perturbPolicy(options_.perturbPolicy);
      }

      evaluate();
      const auto& u = root_->u();

      float baseScore = u[playerIdx];
      if (k > 0 && std::abs(baseScore - lastBest) >= 1e-6 && options_.perturbChance == 0 && options_.perturbPolicy == 0) {
        std::cout << "Potential err! lastBest [" << lastBest << "]" 
                  << " != baseScore [" << baseScore << "]" << std::endl;
      }

      // Which infoSets we want to use?
      InfoSets infoSets;
      if (options_.maxDepth == 0) {
        infoSets = root_->infoSet().allSucc();
      } else {
        // If we run out of sampledDepth_, we stop.
        if (sampledDepth_.empty()) break;
        int depth = sampledDepth_.back(); 
        sampledDepth_.pop_back();
        std::cout << "[" << k << "]: sampled depth = " << depth
                  << ", maxDepth: " << manager_.maxDepth() 
                  << ", maxAllowedSearchDepth: " << options_.maxDepth << std::endl; 
        infoSets = manager_.getInfoSetsByDepth(depth);
      }

      /*
      if (numSamples > 0) {
        if (options_.firstRandomInfoSetKey != "" && k == 0) {
          keys.clear();
          keys.push_back(options_.firstRandomInfoSetKey);
        } else {
          // Random pick one.
          std::random_shuffle(keys.begin(), keys.end());
          keys.erase(keys.begin() + 1, keys.end());
        }
      }
      We could also label each states and its descendents to be active, if we
      want to do sample-based approach.
      */
      /*
      for (const auto& key : keys) {
        auto samples = manager_.drawSamples(key, numSamples);
        auto res2 = manager_.enumPoliciesSamples(samples, playerIdx);
        resultSampling.combine(res2);
      }
      */
      Analysis analysis;
      auto start = std::chrono::high_resolution_clock::now();
      auto resultSampling = _search2({}, infoSets, playerIdx, options_.computeReach ? &analysis : nullptr);
      resultSampling.addBias(baseScore);
      auto stop = std::chrono::high_resolution_clock::now();
      float searchTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6; 

      if (options_.verbose) {
        std::cout << "candidates from search: " << std::endl
                  << resultSampling.info(manager_) << std::endl;
      }

      auto best = resultSampling.getBest();

      bool improved = false;
      if (best.value - lastBest > 1e-4) {
        improved = true;
        if (options_.showBetter) {
          manager_.printStrategy();
        }
        if (options_.maxDepth > 0) {
          // Random shuffle an order of [1, manager_.maxDepth() - options_.maxDepth].
          sampledDepth_ = rela::utils::getIncSeq(manager_.maxDepth() - options_.maxDepth, 1);
          std::shuffle(sampledDepth_.begin(), sampledDepth_.end(), rng_);
        }
      }

      std::cout << "[" << k << ":search]: time: " << searchTime;
      if (improved) std::cout << " result(*): "; 
      else std::cout << " result: ";
      std::cout << best.info(manager_) << std::endl;
      lastBest = best.value;

      if (options_.gtCompute) {
        Analysis analysisGt;

        auto start = std::chrono::high_resolution_clock::now();
        auto resultBruteForce = _bruteforceSearchJointInfoSet({}, infoSets, playerIdx, options_.computeReach ? &analysisGt : nullptr);
        auto stop = std::chrono::high_resolution_clock::now();
        float bruteForceTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6;

        auto bestBruteForce = resultBruteForce.getBest();
        std::cout << "[" << k << ":brute ]: time: " << bruteForceTime << " result: " << bestBruteForce.info(manager_)
                  << std::endl;

        if (std::abs(bestBruteForce.value - best.value) >= 1e-4) {
          std::cout << "Warning! search value [" << best.value 
                    << "] != bruteForce value [" << bestBruteForce.value << "]" << std::endl;
        }

        if (options_.verbose) {
          std::cout << "Search terms: " << std::endl;
          std::cout << analysis.terms.info(manager_, false) << std::endl;
        }

        if (options_.gtOverride) {
          std::cout << "Overriding search with bruteForce!" << std::endl;
          best = bestBruteForce;
          lastBest = best.value;
        }

        if (options_.verbose) {
          std::cout << "candidates from bruteForce: " << std::endl 
                    << resultBruteForce.info(manager_) << std::endl;
        }

        if (options_.computeReach) {
          // Compare the difference of the two reachability.
          analysisGt.compareReach(analysis);
        }
      }

      if (numSamples == 0 && options_.perturbChance == 0 && best.value == baseScore && options_.maxDepth == 0)
        break;

      // Change the policy based on best policy.
      // Loop over its involved infos and change their actions.
      for (const auto& infoAction : best.actions) {
        manager_[infoAction.first].setDeltaStrategy(infoAction.second);
      }
    }

    manager_.perturbChance(0);
    evaluate();
    return root_->u();
  }

  void evaluate() {
    manager_.resetStats();
    root_->propagate(1.0);
  }

  std::string printTree() const {
    return root_->printTree(0);
  }

  const Manager& manager() const {
    return manager_;
  }
  Manager& manager() {
    return manager_;
  }

  std::shared_ptr<State> root() const {
    return root_;
  }

 private:
  Manager manager_;
  std::shared_ptr<State> root_;
  const tabular::Options options_;
  rela::EnvSpec spec_;
  int numPlayer_;

  mutable std::mt19937 rng_;
  std::vector<int> sampledDepth_;

  std::string printPrefix(const std::vector<Entry> &prefix) const {
    std::stringstream ss;
    for (const auto& pre : prefix) {
      const auto &infoSet = manager_[pre.first];
      ss << "(" << pre.first << ", " << infoSet.legalActions()[pre.second] << ") "; 
    }
    return ss.str();
  }

  ResultAgg _search2(const std::vector<Entry> &prefix, const InfoSets& infoSets, 
      int playerIdx, Analysis *analysis) const {
    // From seed, iteratively add new infosets until we reach terminal.
    //
    // Preprocessing.
    std::vector<std::vector<float>> f(infoSets.size());

    for (int k = 0; k < (int)infoSets.size(); ++k) {
      const auto& info = infoSets[k];
      f[k].resize(info->numAction(), 0);
      // if the policy didn't change, what would be the j1 term?
      // Note this is dependent on upstream policies so we have to compute it
      // here (otherwise we could precompute it)
      if (options_.verbose) {
        std::cout << "=== " << printPrefix(prefix) << ", " << info->key() << " === " << std::endl;
      }

      for (const auto& s : info->states()) {
        float alterReach;
        std::string comment;

        const State *ss = s.get();
        const State *p;
        float prob = 1.0f;
        int hop = 0;
        while (true) {
          p = ss->parent();
          if (p == nullptr || p->infoSet().phi() >= 0) break;
          prob *= p->infoSet().strategy()[ss->parentActionIdx()];
          ss = p;
          hop ++;
        }
        
        // Policy of active nodes is skipped since it is aways 1.
        if (p == nullptr) {
            // s doesn't connect to any active infoSet so the reach of s doesn't change.
            alterReach = s->totalReach();
            comment = "traceBackOriginal";
        } else {
          bool inActivePath = (p->infoSet().phi() == ss->parentActionIdx());
          if (inActivePath) { 
            alterReach = p->alterReach() * prob;
          } else {
            alterReach = 0;
          }
          comment = "traceBackHop-" + std::to_string(hop);
        }

        s->setAlterReach(alterReach);

        if (alterReach > 0) {
          for (int a = 0; a < info->numAction(); ++a) {
            float adv = s->child(a).u()[playerIdx] - s->u()[playerIdx];
            f[k][a] += adv * alterReach;
          }
        }

        if (analysis != nullptr) {
          auto prefix2 = prefix;
          prefix2.push_back(std::make_pair(s->key(), -1));
          analysis->reachability.append(Result(prefix2, alterReach, comment));
        }

        if (options_.verbose) {
          std::cout << "  " << s->key() << ", alterReach: " << alterReach
                    << ", reach: " << s->totalReach() << ", u: " << s->u()[playerIdx];
        }
      }

      if (options_.verbose) {
        std::cout << "J2[" << info->key() << "]: reach: " << info->totalReach()
                  << ", u: " << info->u() << ", q: " << info->q()
                  << ", pi: " << info->strategy() << std::endl;
      }
    }

    if (options_.verbose) {
      std::cout << "_search2() summary: " << printPrefix(prefix) << " #infoSets(): " << infoSets.size() << std::endl;
      for (int k = 0; k < (int)infoSets.size(); ++k) {
        const auto& info = infoSets[k];
        std::cout << "  " << info->key() << ", #state: " << info->states().size() << std::endl;
      }
    }

    ResultAgg result;

    if (options_.maxDepth <= 0 || options_.maxDepth > (int)prefix.size()) {
      for (int k = 0; k < (int)infoSets.size(); ++k) {
        const auto& info = infoSets[k];

        for (int a = 0; a < info->numAction(); ++a) {
          auto currPrefix = std::make_pair(info->key(), a);

          if (options_.skipSameDeltaPolicy && info->isDeltaStrategy(a)) continue;

          if (options_.use2ndOrder) {
            for (int k2 = 0; k2 < k; ++k2) {
              const auto& info2 = infoSets[k2];

              for (int b = 0; b < info2->numAction(); ++b) {
                auto currPrefix2 = std::make_pair(info2->key(), b);

                if (options_.skipSameDeltaPolicy && info2->isDeltaStrategy(b)) continue;

                float edge = f[k2][b] + f[k][a];

                auto prefix2 = prefix;
                prefix2.push_back(currPrefix);
                prefix2.push_back(currPrefix2);

                info->setPhi(a);
                info2->setPhi(b);

                auto nextInfoSets = combineInfoSets(info->succ(a), info2->succ(b));

                if (analysis != nullptr) {
                  auto prefix3 = prefix2;
                  prefix3.push_back(std::make_pair("edge2", -1));
                  analysis->terms.append(Result(prefix3, edge));
                }

                ResultAgg res = _search2(prefix2, nextInfoSets, playerIdx, analysis);
                result.append(res.attach(currPrefix, edge).attach(currPrefix2, 0));
                info->setPhi(-1);
                info2->setPhi(-1);
              }
            }
          }

          // What if we only improve one strategy?
          if (!options_.skipSingleInfoSetOpt) {
            float edge = f[k][a];

            auto prefix2 = prefix;
            prefix2.push_back(currPrefix);

            if (analysis != nullptr) {
              auto prefix3 = prefix2;
              prefix3.push_back(std::make_pair("edge1", -1));
              analysis->terms.append(Result(prefix3, edge)); 
            }

            info->setPhi(a);
            ResultAgg res = _search2(prefix2, info->succ(a), playerIdx, analysis);
            result.append(res.attach(currPrefix, edge));
            info->setPhi(-1);
          }
        }
      }
    }
    // Finally if no phi was set, what would be the performance?
    result.append(Result(0));
    for (int k = 0; k < (int)infoSets.size(); ++k) {
      const auto& info = infoSets[k];
      for (const auto& s : info->states()) {
        s->clearAlterReach();
      }
    }

    return result;
  }

  ResultAgg _search(const std::vector<Entry> &prefix, const InfoSets& infoSets, 
      int playerIdx, Analysis *analysis) const {
    // From seed, iteratively add new infosets until we reach terminal.
    //
    // Preprocessing.
    std::vector<float> j1s(infoSets.size(), 0.0f);
    std::vector<float> j3s(infoSets.size(), 0.0f);

    for (int k = 0; k < (int)infoSets.size(); ++k) {
      const auto& info = infoSets[k];
      // if the policy didn't change, what would be the j1 term?
      // Note this is dependent on upstream policies so we have to compute it
      // here (otherwise we could precompute it)
      if (options_.verbose) {
        std::cout << "=== " << printPrefix(prefix) << ", " << info->key() << " === " << std::endl;
      }

      for (const auto& s : info->states()) {
        // For each state, compute J1, which is purely due to analysis change.
        // Trace analysis.
        float alterReach;
        std::string comment;

        const State *ss = s.get();
        const State *p;
        float prob = 1.0f;
        int hop = 0;
        while (true) {
          p = ss->parent();
          if (p == nullptr || p->infoSet().phi() >= 0) break;
          prob *= p->infoSet().strategy()[ss->parentActionIdx()];
          ss = p;
          hop ++;
        }
        
        // Policy of active nodes is skipped since it is aways 1.
        if (p == nullptr) {
            // s doesn't connect to any active infoSet so the reach of s doesn't change.
            alterReach = s->totalReach();
            comment = "traceBackOriginal";
        } else {
          bool inActivePath = (p->infoSet().phi() == ss->parentActionIdx());
          if (inActivePath) { 
            alterReach = p->alterReach() * prob;
          } else {
            alterReach = 0;
          }
          float term = (alterReach - s->totalReach()) * s->u()[playerIdx];
          if (hop == 0 && inActivePath) {
            // Then s is an immediate descent of active infoSets
            j1s[k] += term; 
          } else {
            // Then s connect to some active infoSet so its reachability has changed. 
            // Note that s has been accounted by one infoSet I' aside to the active infoSet, 
            //      assuming v(s) won't change once it leaves I'. 
            // Now s comes back and we want to make corrections. 
            j3s[k] += term;
          }
          comment = "traceBackHop-" + std::to_string(hop);
        }
          
        s->setAlterReach(alterReach);

        if (analysis != nullptr) {
          auto prefix2 = prefix;
          prefix2.push_back(std::make_pair(s->key(), -1));
          analysis->reachability.append(Result(prefix2, alterReach, comment));
        }

        if (options_.verbose) {
          std::cout << "  " << s->key() << ", alterReach: " << alterReach
                    << ", reach: " << s->totalReach() << ", u: " << s->u()[playerIdx] 
                    << ", j1: " << j1s[k] << ", j3: " << j3s[k] << std::endl;
        }
      }

      if (options_.verbose) {
        std::cout << "J2[" << info->key() << "]: reach: " << info->totalReach()
                  << ", u: " << info->u() << ", q: " << info->q()
                  << ", pi: " << info->strategy() << std::endl;
      }
    }

    float sumJ1 = std::accumulate(j1s.begin(), j1s.end(), 0.0f);

    if (options_.verbose) {
      std::cout << "_search() summary: " << printPrefix(prefix) << " #infoSets(): " << infoSets.size() << std::endl;
      for (int k = 0; k < (int)infoSets.size(); ++k) {
        const auto& info = infoSets[k];
        std::cout << "  " << info->key() << ", #state: " << info->states().size() << std::endl;
      }
    }

    ResultAgg result;

    if (options_.maxDepth <= 0 || options_.maxDepth > (int)prefix.size()) {
      for (int k = 0; k < (int)infoSets.size(); ++k) {
        const auto& info = infoSets[k];

        for (int a = 0; a < info->numAction(); ++a) {
          auto currPrefix = std::make_pair(info->key(), a);

          if (options_.skipSameDeltaPolicy && info->isDeltaStrategy(a)) continue;

          if (options_.use2ndOrder) {
            for (int k2 = 0; k2 < k; ++k2) {
              const auto& info2 = infoSets[k2];

              for (int b = 0; b < info2->numAction(); ++b) {
                auto currPrefix2 = std::make_pair(info2->key(), b);

                if (options_.skipSameDeltaPolicy && info2->isDeltaStrategy(b)) continue;

                float edge = sumJ1 - j1s[k] - j3s[k] - j1s[k2] - j3s[k2] + info->residue(a) + info2->residue(b);

                auto prefix2 = prefix;
                prefix2.push_back(currPrefix);
                prefix2.push_back(currPrefix2);

                info->setPhi(a);
                info2->setPhi(b);

                auto nextInfoSets = combineInfoSets(info->succ(a), info2->succ(b));

                if (analysis != nullptr) {
                  auto prefix3 = prefix2;
                  prefix3.push_back(std::make_pair("edge2", -1));
                  analysis->terms.append(Result(prefix3, edge));
                }

                ResultAgg res = _search(prefix2, nextInfoSets, playerIdx, analysis);
                result.append(res.attach(currPrefix, edge).attach(currPrefix2, 0));
                info->setPhi(-1);
                info2->setPhi(-1);
              }
            }
          }

          // What if we only improve one strategy?
          if (!options_.skipSingleInfoSetOpt) {
            float edge = sumJ1 - j1s[k] - j3s[k] + info->residue(a);

            auto prefix2 = prefix;
            prefix2.push_back(currPrefix);

            if (analysis != nullptr) {
              auto prefix3 = prefix2;
              prefix3.push_back(std::make_pair("edge1", -1));
              analysis->terms.append(Result(prefix3, edge)); 
            }

            info->setPhi(a);
            ResultAgg res = _search(prefix2, info->succ(a), playerIdx, analysis);
            result.append(res.attach(currPrefix, edge));
            info->setPhi(-1);
          }
        }
      }
    }
    // Finally if no phi was set, what would be the performance?
    result.append(Result(sumJ1));
    if (analysis != nullptr) {
      auto prefix3 = prefix;
      prefix3.push_back(std::make_pair("sumJ1", -1));
      analysis->terms.append(Result(prefix3, sumJ1));
    }

    for (int k = 0; k < (int)infoSets.size(); ++k) {
      const auto& info = infoSets[k];
      for (const auto& s : info->states()) {
        s->clearAlterReach();
      }
    }

    return result;
  }

  void dumpReachability(const std::vector<Entry>& prefix, const InfoSets& infoSets, Analysis *analysis) {
    evaluate();
    for (const auto& infoSet : infoSets) {
      // Dump all the reachability first.
      for (const auto& s : infoSet->states()) {
        auto prefix2 = prefix;
        prefix2.push_back(std::make_pair(s->key(), -1));
        analysis->reachability.append(Result(prefix2, s->totalReach()));
      }
    }
  }

  ResultAgg _bruteforceSearchJointInfoSet(const std::vector<Entry>& prefix, const InfoSets& infoSets, int playerIdx, 
      Analysis *analysis) {
    // choose possible actions and set the policy accordingly
    if (analysis != nullptr) {
      dumpReachability(prefix, infoSets, analysis);
    }

    ResultAgg res;

    if (options_.maxDepth <= 0 || options_.maxDepth > (int)prefix.size()) {
      for (const auto& infoSet : infoSets) {
        assert(!infoSet->isChanceNode());

        auto strategy = infoSet->strategy();

        for (int a = 0; a < infoSet->numAction(); ++a) {
          auto currAction = std::make_pair(infoSet->key(), a);
          // Set to delta strategy.
          if (options_.skipSameDeltaPolicy && _isDeltaStrategy(strategy, a)) continue; 

          infoSet->setDeltaStrategy(a);

          if (options_.use2ndOrder) {
            for (const auto& infoSet2 : infoSets) {
              if (infoSet2->key() == infoSet->key()) break;

              auto strategy2 = infoSet2->strategy();

              for (int b = 0; b < infoSet2->numAction(); ++b) {
                auto currAction2 = std::make_pair(infoSet2->key(), b);

                if (options_.skipSameDeltaPolicy && _isDeltaStrategy(strategy2, b)) continue;

                // Set to delta strategy.
                infoSet2->setDeltaStrategy(b);

                auto prefix2 = prefix;
                prefix2.push_back(currAction);
                prefix2.push_back(currAction2);

                auto nextInfoSets = combineInfoSets(infoSet->succ(a), infoSet2->succ(b));

                // recurse its children.
                auto thisRes = _bruteforceSearchJointInfoSet(prefix2, nextInfoSets, playerIdx, analysis);
                // auto thisRes = _bruteforceSearch(prefix2, nextInfoSets, playerIdx, analysis);
                res.append(thisRes.attach(currAction, 0).attach(currAction2, 0));
              }
              infoSet2->setStrategy(strategy2);
            }
          }

          if (!options_.skipSingleInfoSetOpt) {
            auto prefix2 = prefix;
            prefix2.push_back(currAction);

            // recurse its children.
            auto thisRes = _bruteforceSearchJointInfoSet(prefix2, infoSet->succ(a), playerIdx, analysis);
            // auto thisRes = _bruteforceSearch(prefix2, nextInfoSets, playerIdx, analysis);
            res.append(thisRes.attach(currAction, 0));
          }
        }

        // Resume old strategy. 
        infoSet->setStrategy(strategy);
      }
    }

    // Evaluate current policy.
    evaluate();
    res.append(Result(root_->u()[playerIdx]));
    return res;
  }

  ResultAgg _bruteforceSearch(const std::vector<Entry>& prefix, const InfoSets& infoSets, int playerIdx, 
      Analysis *analysis) {
    // choose possible actions and set the policy accordingly
    if (analysis != nullptr) {
      dumpReachability(prefix, infoSets, analysis);
    }

    ResultAgg res;
    for (const auto& infoSet : infoSets) {
      assert(!infoSet->isChanceNode());

      auto strategy = infoSet->strategy();
      for (int a = 0; a < infoSet->numAction(); ++a) {
        // Set to delta strategy.
        auto currAction = std::make_pair(infoSet->key(), a);
        infoSet->setDeltaStrategy(a);

        auto prefix2 = prefix;
        prefix2.push_back(currAction);

        // recurse its children.
        auto thisRes = _bruteforceSearch(prefix2, infoSet->succ(a), playerIdx, analysis);
        res.append(thisRes.attach(currAction, 0));
      }
      // Resume old strategy. 
      infoSet->setStrategy(strategy);
    }

    // Evaluate current policy.
    evaluate();
    res.append(Result(root_->u()[playerIdx]));
    return res;
  }
};

}

}  //namespace tabular
