// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include "search.h"

namespace tabular {

namespace search {

void _addInfoSetIfNonExists(InfoSets& infoSets, std::shared_ptr<InfoSet> next) {
  // check if the information set has been added before.
  for (const auto& info : infoSets) {
    if (info->key() == next->key()) {
      return;
    }
  }
  infoSets.push_back(next);
}

void _addInfoSet(InfoSetsWithStats& infoSetsWithStats,
                 std::shared_ptr<InfoSet> next) {
  // check if the information set has been added before.
  for (auto& infoStat : infoSetsWithStats) {
    if (infoStat.info->key() == next->key()) {
      infoStat.count++;
      return;
    }
  }
  infoSetsWithStats.emplace_back(next);
}

InfoSetsWithStats _getInfoSetsStats(const InfoSets& infoSets) {
  InfoSetsWithStats res;
  for (const auto& info : infoSets) {
    res.emplace_back(info);
  }
  return res;
}

InfoSets _getInfoSets(const InfoSetsWithStats& infoSetsWithStats) {
  InfoSets res;
  for (const auto& kv : infoSetsWithStats) {
    res.emplace_back(kv.info);
  }
  return res;
}

std::string Result::info(const Manager& m) const {
  std::stringstream ss;
  ss << std::setprecision(10) << value << ", seq: ";
  ss << prefix(m);
  if (comment != "") {
    ss << ", comment: " << comment;
  }
  return ss.str();
}

std::string Result::prefix(const Manager& m) const {
  std::stringstream ss;
  for (int i = (int)actions.size() - 1; i >= 0; --i) {
    const auto& infoSetKey = actions[i].first;
    const auto* infoSet = m.infoSet(infoSetKey);
    if (infoSet != nullptr) {
      const auto& legalActions = m[infoSetKey].legalActions();
      ss << "(" << infoSetKey << ", " << legalActions[actions[i].second]
         << ") ";
    } else {
      ss << "[" << infoSetKey << "] ";
    }
  }
  ss << "#";
  return ss.str();
}

std::string ResultAgg::info(const Manager& m, bool sortByValue) const {
  std::stringstream ss;

  auto compByValue = [](const Result& r1, const Result& r2) {
    return r1.value > r2.value;
  };

  auto compByPrefix = [&m](const Result& r1, const Result& r2) {
    return r1.prefix(m) < r2.prefix(m);
  };

  std::vector<Result> results2;
  for (const auto& rr : results) {
    results2.push_back(rr.second);
  }

  if (sortByValue) {
    std::sort(results2.begin(), results2.end(), compByValue);
  } else {
    std::sort(results2.begin(), results2.end(), compByPrefix);
  }

  for (const auto& rr : results2) {
    ss << rr.info(m) << std::endl;
  }

  if (!results2.empty() && sortByValue) {
    ss << "Best over " << results2.size() << ": " << results2[0].info(m)
       << std::endl;
  }
  return ss.str();
}

const std::vector<rela::LegalAction>& InfoSet::legalActions() const {
  assert(!states_.empty());
  return states_[0]->legalActions();
}

void InfoSet::update(const State& s) {
  if (isChance_) {
    return;
  }

  // Compute regret for playerIdx.
  const auto& u = s.u();

  for (int i = 0; i < numAction_; ++i) {
    q_[i] += s.child(i).u()[player_] * s.totalReach();
  }

  u_ += u[player_] * s.totalReach();
  totalReach_ += s.totalReach();
}

void State::buildTree(std::shared_ptr<State> own,
                      Manager& manager,
                      const rela::Env& g,
                      bool keepEnvInState) {
  if (manager.getOptions().verbose == VERBOSE) {
    std::cout << "State: " << g.info() << std::endl;
  }

  options_ = manager.getOptions();

  auto spec = g.spec();
  int numPlayer = spec.players.size();

  info_ = manager.getInfoSet(g);
  info_->setDepth(depth_);
  info_->addState(own);
  manager.addState(own);

  if (keepEnvInState) {
    env_ = g.clone();
  }

  u_.resize(numPlayer, 0);
  if (g.terminated()) {
    for (int i = 0; i < numPlayer; ++i) {
      u_[i] = g.playerReward(i);
    }
    return;
  }

  legalActions_ = g.legalActions();
  int numAction = (int)legalActions_.size();

  for (int i = 0; i < numAction; ++i) {
    std::unique_ptr<rela::Env> g_next = g.clone();
    assert(g_next != nullptr);
    g_next->step(legalActions_[i].first);

    children_.emplace_back(
        std::make_shared<State>(depth_ + 1, g_next->completeCompactDesc()));
    children_[i]->buildTree(children_[i], manager, *g_next, keepEnvInState);
    children_[i]->parent_ = own;
    children_[i]->parentActionIdx_ = i;
    info_->addDownStream(i, children_[i]->info_);
  }
}

void Manager::resetStats() {
  for (auto& kv : infoSet_) {
    kv.second->resetStats();
  }
}

std::shared_ptr<InfoSet> Manager::getInfoSet(const rela::Env& g) {
  auto key = g.infoSet();
  assert(key != "");

  int playerId = g.playerIdx();
  int numAction = g.legalActions().size();

  bool isChancePlayer =
      g.spec().players[playerId] == rela::PlayerGroup::GRP_NATURE;

  auto it = infoSet_.find(key);
  if (it != infoSet_.end())
    return it->second;

  auto v = std::make_shared<InfoSet>(
      key, playerId, isChancePlayer, numAction, options_);
  infoSet_[key] = v;

  if (numAction > 0) {
    numActionableInfoSets_++;
  }
  return v;
}

}  // namespace search

}  // namespace tabular
