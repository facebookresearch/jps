#pragma once

#include "rela/a2c_actor.h"
#include "rela/model.h"
#include "rela/clock.h"
#include "rela/prioritized_replay2.h"
#include "rela/thread_loop.h"
#include "rela/utils.h"

struct Rollout {
 public:
  Rollout(std::vector<std::shared_ptr<rela::Actor2>> actors, 
          std::shared_ptr<rela::FutureExecutor> executor)
     : actors_(std::move(actors))
     , executor_(std::move(executor))
     , clock_(rela::clock::gClocks.getClock()) {
  }

  void initialize(std::unique_ptr<rela::Env> env, int requestPlayerIdx, int requestPartnerIdx) {
    //env_ = std::move(env);
    envs_.push_back(std::move(env));
    rewards_.push_back(0);
    requestPlayerIdx_ = requestPlayerIdx;
    requestPartnerIdx_ = requestPartnerIdx;
    rolloutCnt_ = 0;
  }

  void rollout(unsigned int idx) {
    // std::cout << "In rollout! this: " << std::hex << this << std::dec << ", cnt: " << rolloutCnt_ << std::endl;
    assert(envs_.size() > idx);
    // std::cout << "current envs " << std::endl; 
    // for (size_t i = 0; i < envs_.size(); i ++) {
    //   std::cout << envs_[i].get() << std::endl;
    // }
    if (envs_[idx] == nullptr) std::cout << "error! env_ is not initialized! " << std::endl;
    assert(envs_[idx] != nullptr);

    const auto &env = *envs_[idx];

    if (env.subgameEnd()) {
        // std::cout << "game end" << std::endl;
        // std::cout << idx << " " << rewards_.size() << std::endl;
        assert(rewards_.size() > idx);
        rewards_[idx] = env.playerRawScore(requestPlayerIdx_);
        return;
    }

    int playerIdx = env.playerIdx();
    rela::TensorDict obs;
   
    START_TIMING2(clock_, "feature")
    obs = env.feature();
    END_TIMING

    //std::cout << "playerIdx: " << playerIdx << std::endl;
    auto actorFuture = actors_[playerIdx]->act(obs);

    executor_->addFuture([this, actorFuture, idx]() {
      // std::cout << "in future, this: " << std::hex << this << std::dec << ", cnt: " << rolloutCnt_ << std::endl;
      // 
      rela::TensorDict reply;
      START_TIMING2(clock_, "getFuture")
      reply = actorFuture(); 
      END_TIMING

      auto action = rela::utils::getTensorDictScalar<long>(reply, "a");

      // std::cout << "got reply, cnt: " << rolloutCnt_ << std::endl;
      // std::cout << "*****************  " << reply_["a"].item<int>() << std::endl;
      // std::cout << "before step, cnt: " << rolloutCnt_ << std::endl;

      // partner step
      if (requestPartnerIdx_ >= 0 && envs_[idx]->playerIdx() == requestPartnerIdx_) {

          std::vector<int> candidates;

          START_TIMING2(clock_, "partner_candidate")

          auto pi = rela::utils::get(reply, "pi");
          auto piAccessor = pi.accessor<float, 1>();
          for (int i = 0; i < pi.size(0); i++) {
            //TODO: Pass this arg in
            if (piAccessor[i] > 0.05) {
              candidates.push_back(i);
            }
          }

          requestPartnerIdx_ = -1;

          // guard, just in case
          if (candidates.size() <= 1) {
            envs_[idx]->step(action);
            this->rollout(idx);
            return;
          }

          END_TIMING

          START_TIMING2(clock_, "partner_branch")

          int candidateIdx = 0;
          auto tmpEnv = std::move(envs_[idx]);
          envs_.clear();
          rewards_.clear();

          for (int candidate : candidates) {
            //rela::TensorDict oldReply = reply;
            auto searchEnv = tmpEnv->clone();
            // std::cout << "in future, env: " << tmpEnv.get() << std::endl;
            // std::cout << "in future, searchEnv: " << searchEnv.get() << std::endl;
            assert(searchEnv != nullptr);
            //std::cout << "inside rollout step " << reply["a"].item<int>() << std::endl;

            searchEnv->step(candidate);
            envs_.push_back(std::move(searchEnv));
            // std::cout << "inside rollout current envs " << std::endl; 
            // for (size_t i = 0; i < envs_.size(); i ++) {
            //   std::cout << envs_[i].get() << std::endl;
            // }
            rewards_.push_back(0);
            this->rollout(candidateIdx);
            // std::cout << "after rollout current envs " << std::endl; 
            // for (size_t i = 0; i < envs_.size(); i ++) {
            //   std::cout << envs_[i].get() << std::endl;
            // }
            candidateIdx += 1;
          }

          END_TIMING
      } else {
         //std::cout << "step " << reply["a"].item<int>() << std::endl;
         //
         START_TIMING2(clock_, "normal_rollout")

         envs_[idx]->step(action);
         this->rollout(idx);

         END_TIMING
      } 
      rolloutCnt_ ++;

      // std::cout << "after step, cnt: " << rolloutCnt_ << std::endl;
    });
  } 

  float reward() const { 
    size_t n = rewards_.size();
    float reward = 0;
    for (size_t i = 0; i < n; i++) {
      reward += rewards_[i];
    }
    return reward / n; 
  }

 private:
  //std::unique_ptr<rela::Env> env_;
  std::vector<std::unique_ptr<rela::Env>> envs_;
  rela::TensorDict reply_;
  int requestPlayerIdx_ = -1;
  int requestPartnerIdx_ = -1;
  std::vector<std::shared_ptr<rela::Actor2>> actors_;
  std::vector<float> rewards_;
  int rolloutCnt_ = 0;

  std::shared_ptr<rela::FutureExecutor> executor_;

  rela::clock::ThreadClock &clock_;
};

class SearchActor {
 public:
  SearchActor(std::shared_ptr<rela::Env> env,
           std::vector<std::shared_ptr<rela::Actor2>> actors, 
           float searchRatio,
           int seed,
           bool keepRewardHistory = false)
      : env_(std::move(env))
      , actors_(std::move(actors)) 
      , searchRatio_(searchRatio)
      , keepRewardHistory_(keepRewardHistory)
      , clock_(rela::clock::gClocks.getClock()) {
    rng_.seed(seed);
    assert(env_->spec().players.size() == actors_.size());
    for (int i = 0; i < (int)actors_.size(); ++i) {
      replays_.emplace_back();
    }
    env_->reset();
    spec_ = env_->spec();
  }

  void setFutureExecutor(std::shared_ptr<rela::FutureExecutor> executor) {
    for (int i = 0; i < (int)actors_.size(); ++i) {
      actors_[i]->setFutureExecutor(executor);
    }
    executor_ = executor;
  }

  void preAct() {
    auto playerIdx = env_->playerIdx();

    // Get feature. 
    rela::TensorDict obs;

    START_TIMING2(clock_, "feature")
    obs = env_->feature();
    END_TIMING

    replays_[playerIdx].push_back(obs);
    auto actorFuture = actors_[playerIdx]->act(obs);

    //std::cout << "obs is " << rela::utils::printTensorDict(obs);

    //std::cout << "EnvActor: Before preAct " << std::endl;
    auto f = [this, actorFuture, playerIdx]() {
      //std::cout << "executing lamda" << std::endl;
      //std::cout << "playeridx " << playerIdx << std::endl;
      //
      START_TIMING2(clock_, "getFuture")
      reply_ = actorFuture();
      END_TIMING

      // Cannot optimize nature. 
      if (spec_.players[playerIdx] == rela::PlayerGroup::GRP_NATURE || skipRollout_) {
        return;
      }

      START_TIMING2(clock_, "player_branch")

      auto pi = rela::utils::get(reply_, "pi");
      auto piAccessor = pi.accessor<float, 1>();
      std::vector<std::pair<float, int>> vecPi(pi.size(0));

      for (int i = 0; i < pi.size(0); i++) {
        vecPi[i] = std::make_pair(piAccessor[i], i);
      }
      std::sort(vecPi.begin(), vecPi.end());

      std::vector<int> candidates;
      for (int i = pi.size(0) - 1; i >= 0; --i) {
        if (vecPi[i].first > 0.05) {
          candidates.push_back(vecPi[i].second);
        } else {
          break;
        }
      }

      if (candidates.size() <= 1) {
        // Reduce to multinomial.
        return;
      } 

      std::vector<int> partnerIndices = env_->partnerIndices(playerIdx);
      assert(partnerIndices.size() == 1);
      int partnerIdx = partnerIndices[0];

      for (int candidate : candidates) {
        //std::cout << "rollout for action " << candidate << std::endl; 
        auto &r = rollouts_[candidate];

        for (int j = 0; j < 5; j++) {
          //rela::TensorDict oldReply = reply;
          auto searchEnv = env_->clone();
          assert(searchEnv != nullptr);
          searchEnv->step(candidate);

          r.emplace_back(std::make_unique<Rollout>(actors_, executor_));
          r.back()->initialize(std::move(searchEnv), playerIdx, partnerIdx);
          r.back()->rollout(0);
        }
      }

      END_TIMING
    };
    
    executor_->addFuture(f);

    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() {
    //std::cout << "EnvActor: Before processRequest " << std::endl;
    int bestAction = -1;
    if (! rollouts_.empty()) {
      // Collect rollouts. 
      float bestReward = -std::numeric_limits<float>::max();

      for (const auto &k2v : rollouts_) {
        float totalReward = 0;
        for (const auto &r : k2v.second) {
          totalReward += r->reward();
        }
        //std::cout << k2v.first << " reward " << totalReward / 5 << std::endl;
        if (totalReward > bestReward) {
          bestReward = totalReward;
          bestAction = k2v.first;
        }
      }
      reply_["a"][0] = bestAction;
      // Clear all rollouts
      rollouts_.clear();
    } else {
      bestAction = rela::utils::getTensorDictScalar<long>(reply_, "a");
    }

    auto playerIdx = env_->playerIdx();
    //std::cout << "EnvActor: After got reply" << std::endl;
    //std::cout << "***************** postact " << reply_["a"].item<int>() << std::endl;
    //assert(false);

    env_->step(bestAction);
    // Some env (e.g.. Bridge) will have multiple subgames in one game. 
    bool subgameEnd = env_->subgameEnd();
    bool terminated = env_->terminated();

    // std::cout << "EnvActor: After step" << std::endl;
    reply_["reward"] = torch::zeros({1});
    reply_["terminal"] = torch::zeros({1}, torch::kBool);

    // std::cout << "EnvActor: before append" << std::endl;
    auto &obs = replays_[playerIdx].back();
    rela::utils::appendTensorDict(obs, reply_);

    // Put terminal signal for all replays of all players.
    // Note that due to incomplete information nature, some
    // players might not see it.
    if (subgameEnd || terminated) {
      //std::cout << "terminated" << std::endl;
      // Set terminal to actors and replays.
      for (int i = 0; i < (int)actors_.size(); ++i) {
        actors_[i]->setTerminal();
        auto accessor = replays_[i].back()["terminal"].accessor<bool, 1>();
        accessor[0] = true;
      }
    }

    // std::cout << "EnvActor: after append" << std::endl;
    if (! terminated) return;

    if (rng_() % rngConst >= searchRatio_ * rngConst) {
      skipRollout_ = true;
    } else {
      skipRollout_ = false;
    }
    //assert(false);
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

    // push rewards to the history 
    if (keepRewardHistory_) {
      historyRewards_.push_back(rewards);
    }
  }

  void sendExperience() {
    if (!env_->terminated())
      return;

    for (int i = 0; i < (int)replays_.size(); ++i) {
      while (replays_[i].size() > 0) {
        actors_[i]->sendExperience(replays_[i].front());
        replays_[i].pop_front();
      }
    }
  }

  void postSendExperience() {
    if (! env_->terminated()) 
      return;

    // std::cout << "Reseting environment" << std::endl;
    env_->reset();
    for (int i = 0; i < (int)actors_.size(); ++i) {
      assert(replays_[i].empty());
    }
    terminalCount_++;
  }

  int getTerminalCount() const {
    return terminalCount_;
  }

  const std::vector<std::vector<float>> &getHistoryRewards() const { 
    return historyRewards_; 
  }

 private:
  // One environment.
  std::shared_ptr<rela::Env> env_;
  rela::EnvSpec spec_;
  int terminalCount_ = 0;
  
  // You might have multiple actors for multi-agent situations.
  std::vector<std::shared_ptr<rela::Actor2>> actors_;

  float searchRatio_;

  // Reward of each player in each game.
  const bool keepRewardHistory_;
  std::vector<std::vector<float>> historyRewards_;

  // replays, one for each actor.
  std::vector<std::deque<rela::TensorDict>> replays_;
  std::shared_ptr<rela::FutureExecutor> executor_;
 
  // For rollouts. 
  std::unordered_map<int, std::vector<std::unique_ptr<Rollout>>> rollouts_;
  rela::TensorDict reply_;

  std::mt19937 rng_;
  const int rngConst = 10000;
  bool skipRollout_ = true;

  rela::clock::ThreadClock& clock_;
};
