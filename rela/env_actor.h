#pragma once

#include "rela/a2c_actor.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/thread_loop.h"

#include "rela/env_actor_base.h"

struct EnvActorMoreOptions {
  bool useSad = false;
  int sadMaxActionRound = 40;
};

class EnvActor : public EnvActorBase {
 public:
  EnvActor(std::shared_ptr<rela::Env> env,
           std::vector<std::shared_ptr<rela::Actor2>> actors,
           const EnvActorOptions& options,
           const EnvActorMoreOptions& moreOptions)
      : EnvActorBase(actors, options)
      , env_(std::move(env))
      , moreOptions_(moreOptions) {
    checkValid(*env_);
    env_->reset();

    for (int i = 0; i < (int)actors_.size(); ++i) {
      replays_.emplace_back();
    }

    // Two tables.
    probs_.resize(2);
    greedyActions_.resize(2);

    auto spec = env_->spec();
    if (moreOptions_.useSad) {
      // In sad case, we want to duplicate public action part.
      maxRound_ = spec.maxActionRound;
      if (maxRound_ <= 0)
        maxRound_ = moreOptions_.sadMaxActionRound;

      maxPlayerAction_ = 0;
      for (int i = 0; i < (int)spec.maxNumActions.size(); ++i) {
        if (spec.players[i] != rela::PlayerGroup::GRP_NATURE) {
          maxPlayerAction_ = std::max(maxPlayerAction_, spec.maxNumActions[i]);
        }
      }
      // One hot representation for greedy actions.
      // Note that we never use the last action in SAD (since terminal is
      // already 1)
      // so the history is at most maxRound_ - 1 long.
      sadDimension_ = maxRound_ * maxPlayerAction_;
    } else {
      sadDimension_ = 0;
    }
  }

  int featureDim(const rela::Env&) const {
    return env_->featureDim() + sadDimension_;
  }

  void preAct() override {
    auto playerIdx = env_->playerIdx();

    // Get feature.
    auto obs = env_->feature();

    if (sadDimension_ > 0) {
      const auto& acts = greedyActions_[subgameIdx_];
      // The history is at most maxRound_ - 1 long.
      assert((int)acts.size() <= maxRound_ - 1);
      // additional dimension.
      auto greedyOneHot = torch::zeros({sadDimension_}, torch::kFloat32);
      auto accessor = greedyOneHot.accessor<float, 1>();
      for (int i = 0; i < (int)acts.size(); ++i) {
        assert(acts[i] < maxPlayerAction_);
        accessor[i * maxPlayerAction_ + acts[i]] = 1.0f;
      }
      // Update the input state.
      obs["s"] = torch::cat({obs["s"], greedyOneHot});
    }

    replays_[playerIdx].push_back(obs);

    // std::cout << "EnvActor: Before preAct " << std::endl;
    actFuture_ = actors_[playerIdx]->act(obs);
    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() override {
    // std::cout << "EnvActor: Before processRequest " << std::endl;

    auto playerIdx = env_->playerIdx();
    rela::TensorDict reply = actFuture_();
    // std::cout << "EnvActor: After got reply" << std::endl;
    //
    //
    // We don't want to add extra burden during training.
    if (options_.eval) {
      env_->setReply(reply);
    }

    auto action = rela::utils::getTensorDictScalar<long>(reply, "a");
    env_->step(action);

    if (env_->spec().players[playerIdx] != rela::PlayerGroup::GRP_NATURE) {
      std::string key = "pi";
      if (!rela::utils::hasKey(reply, key)) {
        // Surrogate key in case we use value based approach
        key = "adv";
      }
      if (rela::utils::hasKey(reply, key)) {
        auto pi = rela::utils::get(reply, key);
        auto accessor = pi.accessor<float, 1>();
        probs_[subgameIdx_].push_back(accessor[action]);

        if (moreOptions_.useSad) {
          // Also find the greedy action and save it.
          auto maxProb = rela::utils::getMaxProb(reply, key);
          greedyActions_[subgameIdx_].push_back(maxProb.second);
        }
      }
    }

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
      // Set terminal to actors and replays.
      for (int i = 0; i < (int)actors_.size(); ++i) {
        actors_[i]->setTerminal();
        auto accessor = replays_[i].back()["terminal"].accessor<bool, 1>();
        accessor[0] = true;
      }
      subgameIdx_++;
    }

    // std::cout << "EnvActor: after append" << std::endl;
    if (!terminated)
      return;

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
    if (!env_->terminated())
      return;

    for (int i = 0; i < (int)replays_.size(); ++i) {
      while (replays_[i].size() > 0) {
        actors_[i]->sendExperience(replays_[i].front());
        replays_[i].pop_front();
      }
    }
  }

  void postSendExperience() override {
    if (!env_->terminated())
      return;

    if (!env_->reset())
      terminateEnvActor();
    subgameIdx_ = 0;
    for (auto& p : probs_) {
      p.clear();
    }

    for (auto& p : greedyActions_) {
      p.clear();
    }

    for (int i = 0; i < (int)actors_.size(); ++i) {
      assert(replays_[i].empty());
    }
  }

 protected:
  std::string getSaveData() const override {
    json j = env_->jsonObj();

    // Also save probs.
    for (int i = 0; i < (int)probs_.size(); ++i) {
      j["bidd"][i]["probs"] = probs_[i];
    }

    return j.dump();
    // assert(database_->saveData(offset_ + currentIdx_, jsonStr));
  }

  std::string getDisplayData() const override {
    return env_->info();
  }

 private:
  // One environment.
  std::shared_ptr<rela::Env> env_;
  int subgameIdx_ = 0;

  // replays, one for each actor.
  std::vector<std::deque<rela::TensorDict>> replays_;

  rela::TensorDictFuture actFuture_ = nullptr;

  std::vector<std::vector<float>> probs_;
  std::vector<std::vector<int>> greedyActions_;
  const EnvActorMoreOptions moreOptions_;

  int sadDimension_ = 0;
  int maxRound_ = 0;
  int maxPlayerAction_ = 0;
};
