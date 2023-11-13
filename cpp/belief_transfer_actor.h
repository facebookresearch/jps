#pragma once

#include "rela/a2c_actor.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/env_actor_base.h"

#include "bridge_env.h"

struct BeliefTransferOptions {
  bool openingLead = false;
  bool debug = false;
};

class BeliefTransferEnvActor : public EnvActorBase {
 public:
  BeliefTransferEnvActor(std::shared_ptr<bridge::BridgeEnv> env,
           std::vector<std::shared_ptr<rela::Actor2>> actors, 
           const EnvActorOptions &options,
           const BeliefTransferOptions &btOptions,
           std::shared_ptr<rela::PrioritizedReplay2> replayBuffer)
      : EnvActorBase(actors, options)
      , env_(std::move(env))
      , btOptions_(btOptions)
      , replayBuffer_(replayBuffer) { 
    checkValid(*env_);
    env_->reset();
  }

  void preAct() override {
    auto playerIdx = env_->playerIdx();

    // Get feature. 
    auto obs = env_->feature();

    // std::cout << "EnvActor: Before preAct " << std::endl;
    actFuture_ = actors_[playerIdx]->act(obs);
    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() override {
    // std::cout << "EnvActor: Before processRequest " << std::endl;

    // auto playerIdx = env_->playerIdx();
    rela::TensorDict reply = actFuture_();
    // std::cout << "EnvActor: After got reply" << std::endl;
    //
    auto action = rela::utils::getTensorDictScalar<long>(reply, "a");
    env_->step(action);

    // Some env (e.g.. Bridge) will have multiple subgames in one game. 
    bool terminated = env_->terminated();

    if (! terminated) return;

    envTerminate();

    // Put terminal signal for all replays of all players.
    // Note that due to incomplete information nature, some
    // players might not see it.
    //
    // Belief encoding: for each card (0-51), tell who owns it (NESW)
    // When doing training, pick a player (or pick all four players) and predict where are all the cards.  

    // Send the data directly to replay buffer for training. 
    for (int tableIdx = 0; tableIdx < 2; ++tableIdx) {
      if (btOptions_.openingLead) {
        // only send data for opening lead. 
        if (env_->hasOpeningLead(tableIdx)) {
          auto f = env_->featureOpeningLead(tableIdx, true, btOptions_.debug);
          rela::Transition transition(f);
          replayBuffer_->add(transition, 1.0f);
        }
      } else {
        for (int seat = 0; seat < 4; ++seat) {
          auto f = env_->featureWithTableSeat(tableIdx, seat, true);
          rela::Transition transition(f);
          replayBuffer_->add(transition, 1.0f);
        }
      }
    }

    // Reset the environment.
    if (! env_->reset()) {
      terminateEnvActor();
    }
  }

 protected:
  std::string getDisplayData() const override {
    return env_->info();
  }

 private:
  // One environment.
  std::shared_ptr<bridge::BridgeEnv> env_;
  const BeliefTransferOptions btOptions_;

  std::shared_ptr<rela::PrioritizedReplay2> replayBuffer_;

  rela::TensorDictFuture actFuture_ = nullptr;
};
