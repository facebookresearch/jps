#pragma once

#include "rela/a2c_actor.h"
#include "rela/clock.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/thread_loop.h"
#include "rela/utils.h"

#include "rela/env_actor_base.h"
#include "rela/rollouts.h"
#include "rela/search_actor_options.h"

#include "simple_game/comm.h"

using Rollout = rela::rollout::Rollout;

class SearchRolloutCtrl : public rela::rollout::Ctrl {
 public:
  using Action = rela::rollout::Action;

  SearchRolloutCtrl(std::vector<std::shared_ptr<rela::Actor2>> actors)
      : actors_(std::move(actors)) {}

  void initialize(bool skipRollout, int requestPlayerIdx, int requestPartnerIdx,
                  rela::TensorDict rootObs) {
    rootObs_ = rootObs;

    skipRollout_ = skipRollout;
    requestPlayerIdx_ = requestPlayerIdx;
    requestPartnerIdx_ = requestPartnerIdx;
  }

  const rela::TensorDict reply() const { return reply_; }
  bool verbose() const { return verbose_; }
  void setVerbose(bool verbose) { verbose_ = verbose; }

  std::pair<bool, float> shouldStop(const rela::Env& env,
                                    const Rollout& r) const override {
    if (r.depth() > 0 && env.subgameEnd()) {
      // Note that we don't stop if the root node has subgameEnd = true,
      //      since it means the starting of the second table.
      // std::cout << "game end" << std::endl;
      float reward = env.playerRawScore(requestPlayerIdx_);
      return std::make_pair(true, reward);
    }

    return std::make_pair(false, 0.0f);
  }

  rela::TensorDictFuture preAct(const rela::Env& env,
                                const Rollout& r) override {
    rela::TensorDict obs;

    START_TIMING("feature")
    obs = (r.depth() == 0 ? rootObs_ : env.feature());
    END_TIMING

    // auto obs = env.feature();
    int playerIdx = env.playerIdx();

    // std::cout << "playerIdx: " << playerIdx << std::endl;
    rela::TensorDictFuture f;

    START_TIMING("act")
    f = actors_[playerIdx]->act(obs);
    END_TIMING

    return f;
  }

  // Return action set, and reward.
  // If the action set is empty, then the reward is the final reward.
  // If the action set is non-empty, then the reward is 0.
  std::vector<Action> getCandidateActions(const rela::Env& env,
                                          const Rollout& r,
                                          rela::TensorDictFuture f) override {
    (void)r;

    assert(f != nullptr);

    int playerIdx = env.playerIdx();

    // std::cout << "SearchCtrl: Getting reply . playerIdx: " << playerIdx << ",
    // requestPlayerIdx: " << requestPlayerIdx_ << ", requestPartnerIdx: " <<
    // requestPartnerIdx_
    //           << ", r.depth(): " << r.depth() << ", r.generation(): " <<
    //           r.generation() << std::endl;
    START_TIMING("getFuture")
    reply_ = f();
    END_TIMING

    rela::utils::assertKeyExists(reply_, {"a"});

    // Note that if the root is terminal, then we still allow
    // getCandidateActions to be called, and stop here.
    if (skipRollout_ || env.subgameEnd()) return {};

    // std::cout << "got reply, cnt: " << rolloutCnt_ << std::endl;
    // std::cout << "*****************  " << reply_["a"].item<int>() <<
    // std::endl; std::cout << "before step, cnt: " << rolloutCnt_ << std::endl;

    std::vector<Action> candidates;
    int rep = 1;

    if (playerIdx == requestPlayerIdx_ && r.generation() == 0) {
      // first choice.
      //
      START_TIMING("player_choice")
      candidates = firstChoice();
      END_TIMING

      rep = 5;

      // If there is only 0/1 choice, no need to do search and we reduce to
      // multinomial. By returning empty action set, the search will end
      // automatically.
      if (candidates.size() <= 1) return {};

    } else if (playerIdx == requestPartnerIdx_ && r.generation() == 1) {
      // partner choice
      //
      START_TIMING("partner_choice")
      candidates = partnerChoice();
      END_TIMING
    }

    // Convert candidates to res.
    START_TIMING("build_tensor_vec")
    if (!candidates.empty()) {
      for (auto& a : candidates) {
        a.repeat = rep;
      }
    } else {
      // Normal rollouts.
      Action a;
      a.action = rela::utils::getTensorDictScalar<long>(reply_, "a");
      candidates.push_back(std::move(a));
    }
    END_TIMING

    return candidates;
  }

  std::string actionDisplay(int action) const override {
    // Action string.
    return std::to_string(action);
  }

 private:
  std::vector<std::shared_ptr<rela::Actor2>> actors_;

  rela::TensorDict rootObs_;
  rela::TensorDict reply_;

  bool skipRollout_ = false;
  int requestPlayerIdx_ = -1;
  int requestPartnerIdx_ = -1;

  bool verbose_ = false;

  std::vector<Action> firstChoice() {
    // Set thres = 1e-10 to get rid of illegal actions actions (they are masked
    // to be precisely 0).
    const auto vecPi = rela::utils::getSortedProb(reply_, "pi", 1e-10);
    std::vector<Action> candidates;

    const double kThres = 0.05;
    for (const auto& v : vecPi) {
      if (v.first > kThres) {
        Action a;
        a.action = v.second;
        // a.penalty = 1 - v.first;
        candidates.push_back(std::move(a));
      } else {
        break;
      }
    }

    if (verbose_ && candidates.size() == 0) {
      std::cout << "Player Choice: #" << candidates.size()
                << " playerIdx: " << requestPlayerIdx_ << std::endl;
    }

    return candidates;
  }

  std::vector<Action> partnerChoice() {
    const auto vecPi = rela::utils::getSortedProb(reply_, "pi", 1e-10);

    std::vector<Action> candidates;

    const double kThres = 0.05;
    for (const auto& v : vecPi) {
      if (v.first > kThres) {
        Action a;
        a.action = v.second;
        // a.penalty = 1 - v.first;
        candidates.push_back(std::move(a));
      } else {
        break;
      }
    }

    if (verbose_ && candidates.size() == 0) {
      std::cout << "Partner Choice: #" << candidates.size()
                << " partnerIdx: " << requestPartnerIdx_ << std::endl;
    }
    return candidates;
  }
};

class SearchActor : public EnvActorBase {
 public:
  SearchActor(std::shared_ptr<rela::Env> env,
              std::vector<std::shared_ptr<rela::Actor2>> actors,
              const EnvActorOptions& options,
              const SearchActorOptions& searchOptions)
      : EnvActorBase(actors, options),
        env_(std::move(env)),
        searchCtrl_(actors_),
        searchOptions_(searchOptions) {
    checkValid(*env_);
    for (int i = 0; i < (int)actors_.size(); ++i) {
      replays_.emplace_back();
    }
    env_->reset();
    spec_ = env_->spec();
  }

  void preAct() override {
    int playerIdx = env_->playerIdx();

    bool skipRollout = false;
    if (spec_.players[playerIdx] == rela::PlayerGroup::GRP_NATURE ||
        options_.eval) {
      // Do not optimize nature.
      skipRollout = true;
    } else {
      if (random() % rngConst >= searchOptions_.searchRatio * rngConst) {
        skipRollout = true;
      }
    }

    rela::TensorDict obs;

    START_TIMING("feature")
    obs = env_->feature();
    END_TIMING

    replays_[playerIdx].push_back(obs);

    std::vector<int> partnerIndices = env_->partnerIndices(playerIdx);
    int partnerIdx = partnerIndices.size() == 1 ? partnerIndices[0] : -1;

    searchCtrl_.setVerbose(random() % 500000 == 0);
    searchCtrl_.initialize(skipRollout, playerIdx, partnerIdx, obs);

    // TODO one extra copy of the environment.
    root_ = std::make_unique<Rollout>(*env_, searchCtrl_, getExecutor());
    root_->run();

    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() override {
    // std::cout << "EnvActor: Before processRequest " << std::endl;
    rela::TensorDict reply;

    using Status = rela::rollout::Rollout::Status;

    int bestAction = -1;

    START_TIMING("get_result")
    reply = searchCtrl_.reply();
    auto status = root_->status();

    if (status == Status::BRANCHED) {
      rela::rollout::Result result =
          root_->getBest(searchOptions_.bestOnBest, searchCtrl_.verbose());
      bestAction = result.action;
      reply["a"][0] = result.action;
    } else {
      assert(status == Status::TERMINATED || status == Status::EMPTY_ACTION);
      bestAction = rela::utils::getTensorDictScalar<long>(reply, "a");
    }
    END_TIMING

    auto it = reply.find("a");
    if (it == reply.end()) {
      std::cout << "No key a"
                << ", status: " << root_->printStatus() << std::endl;
    }

    auto playerIdx = env_->playerIdx();

    env_->step(bestAction);
    // Some env (e.g.. Bridge) will have multiple subgames in one game.
    bool subgameEnd = env_->subgameEnd();
    bool terminated = env_->terminated();

    // std::cout << "EnvActor: After step" << std::endl;
    reply["reward"] = torch::zeros({1});
    reply["terminal"] = torch::zeros({1}, torch::kBool);

    // std::cout << "EnvActor: before append" << std::endl;
    auto& obs = replays_[playerIdx].back();
    rela::utils::appendTensorDict(obs, reply);

    // Put terminal signal for all replays of all players.
    // Note that due to incomplete information nature, some
    // players might not see it.
    if (subgameEnd || terminated) {
      // std::cout << "terminated" << std::endl;
      // Set terminal to actors and replays.
      for (int i = 0; i < (int)actors_.size(); ++i) {
        actors_[i]->setTerminal();
        auto accessor = replays_[i].back()["terminal"].accessor<bool, 1>();
        accessor[0] = true;
      }
    }

    // std::cout << "EnvActor: after append" << std::endl;
    if (!terminated) return;

    // assert(false);
    // backfill replay buffer rewards.
    std::vector<float> rewards;
    for (int i = 0; i < (int)replays_.size(); ++i) {
      auto& r = replays_[i];
      float reward = env_->playerReward(i);
      rewards.push_back(reward);

      for (auto it = r.begin(); it != r.end(); ++it) {
        // Only assign rewards for terminal nodes.
        if ((*it)["terminal"].item<bool>()) {
          auto accessor = (*it)["reward"].accessor<float, 1>();
          accessor[0] = reward;
        }
      }
    }

    envTerminate(&rewards);
  }

  void sendExperience() override {
    if (!env_->terminated()) return;

    for (int i = 0; i < (int)replays_.size(); ++i) {
      while (replays_[i].size() > 0) {
        actors_[i]->sendExperience(replays_[i].front());
        replays_[i].pop_front();
      }
    }
  }

  void postSendExperience() override {
    if (!env_->terminated()) return;

    // std::cout << "Reseting environment" << std::endl;
    if (!env_->reset()) {
      terminateEnvActor();
    }

    for (int i = 0; i < (int)actors_.size(); ++i) {
      assert(replays_[i].empty());
    }
  }

 private:
  // One environment.
  std::shared_ptr<rela::Env> env_;
  rela::EnvSpec spec_;

  // replays, one for each actor.
  std::vector<std::deque<rela::TensorDict>> replays_;

  // For rollouts.
  SearchRolloutCtrl searchCtrl_;
  const SearchActorOptions searchOptions_;
  std::unique_ptr<Rollout> root_;

  const int rngConst = 10000;
};
