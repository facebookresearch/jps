#pragma once

// #include "rela/search_actor_refactored.h"
// #include "rela/search_actor.h"
#include "rela/thread_loop.h"
#include "rela/clock.h"
#include "rela/env_actor_base.h"

// thread loop for joint Q learning
// TODO: rename properly, change vector to shared_ptr
// TODO: create a common base class for EnvActor and merge ThreadLoop2/ThreadLoop3
class ThreadLoopEnvActor : public rela::ThreadLoop {
 public:
  ThreadLoopEnvActor(int threadIdx, 
              std::vector<std::shared_ptr<EnvActorBase>> envActors,
              int numGamePerEnv = -1)
      : threadIdx_(threadIdx)
      , envActors_(std::move(envActors))
      , numGamePerEnv_(numGamePerEnv) {
      executor_ = std::make_shared<rela::FutureExecutor>();
      for (auto &ea : envActors_) {
        ea->setFutureExecutor(executor_);
      }
  }

  void mainLoop() final {
    auto& clock = rela::clock::gClocks.getClock();
    clock.setName(std::to_string(threadIdx_));
    clock.reset();

    while (!terminated()) {
      if (paused()) {
        waitUntilResume();
      }

      START_TIMING2(clock, "preAct")
      // std::cout << "Before preAct" << std::endl;
      for (auto& ea : envActors_) {
        if (!isEnvFinished(*ea)) {
          ea->preAct();
        }
      }
      END_TIMING

      START_TIMING2(clock, "preActExecute")
      executor_->execute();
      END_TIMING

      START_TIMING2(clock, "postAct")
      // std::cout << "Before postAct" << std::endl;
      for (auto& ea : envActors_) {
        if (!isEnvFinished(*ea)) {
          ea->postAct();
        }
      }
      END_TIMING

      START_TIMING2(clock, "postActExecute")
      executor_->execute();
      END_TIMING

      // std::cout << "Before sendExperience" << std::endl;
      START_TIMING2(clock, "sendExperience")
      for (auto& ea : envActors_) {
        if (!isEnvFinished(*ea)) {
          ea->sendExperience();
        }
      }
      END_TIMING

      START_TIMING2(clock, "sendExperienceExecute")
      executor_->execute();
      END_TIMING

      // std::cout << "Before postSendExperience" << std::endl;
      START_TIMING2(clock, "postSendExperience")
      for (auto& ea : envActors_) {
        if (!isEnvFinished(*ea)) {
          ea->postSendExperience();
        }
      }
      END_TIMING

      START_TIMING2(clock, "postSendExperienceExecute")
      executor_->execute();
      END_TIMING

      bool allFinished = true;
      for (auto& ea : envActors_) {
        if (!isEnvFinished(*ea)) {
          allFinished = false;
          break;
        }
      }

      START_TIMING2(clock, "finalExecute")
      executor_->execute();
      END_TIMING

      if (allFinished) {
        // std::cout << "all Finished! Existing mainloop!" << std::endl;
        break;
      }

      loopCnt_ ++;
      if (loopCnt_ % 100 == 0 && threadIdx_ == 0) {
        PRINT_TIMING_SUMMARY2(clock);
      }
    }
  }

 private:
  const int threadIdx_;

  std::vector<std::shared_ptr<EnvActorBase>> envActors_;
  std::shared_ptr<rela::FutureExecutor> executor_;
  const int numGamePerEnv_;

  int loopCnt_ = 0;

  bool isEnvFinished(const EnvActorBase& ea) const {
    return (numGamePerEnv_ > 0 && ea.getTerminalCount() >= numGamePerEnv_) || ea.isTerminated();
  }
};
