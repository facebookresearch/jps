#pragma once

#include <memory>
#include <vector>
#include <string>

#include "rela/a2c_actor.h"
#include "rela/env.h"

struct EnvActorOptions {
  int threadIdx = -1;
  int seed;
  std::string savePrefix;
  int displayFreq = 0;
  bool eval = false;

  // This is only true when we want to get #action and #featureDim.
  bool emptyInit = false;
};

class EnvActorBase {
 public:
  EnvActorBase(std::vector<std::shared_ptr<rela::Actor2>> actors, 
               const EnvActorOptions &options)
      : actors_(std::move(actors)) 
      , options_(options)
      , rng_(options_.seed) {
    // std::cout << "threadIdx: " << options_.threadIdx << std::endl; 
    if (options_.savePrefix != "") {
      output_ = std::make_unique<std::ofstream>(options_.savePrefix + "-" + std::to_string(options_.threadIdx));
    }
  }

  void setFutureExecutor(std::shared_ptr<rela::FutureExecutor> executor) {
    for (int i = 0; i < (int)actors_.size(); ++i) {
      actors_[i]->setFutureExecutor(executor);
    }
    executor_ = executor;
  }

  int getTerminalCount() const {
    return terminalCount_;
  }

  std::shared_ptr<rela::FutureExecutor> getExecutor() { 
    return executor_; 
  }

  uint64_t random() { return rng_(); }

  bool isTerminated() const { return isTerminated_; }

  const std::vector<std::vector<float>> &getHistoryRewards() const { 
    return historyRewards_; 
  }

  void checkValid(const rela::Env &env) const {
    if (options_.emptyInit) {
      assert(actors_.empty());
    } else {
      assert(env.spec().players.size() == actors_.size());
    }
  }

  virtual void preAct() = 0;
  virtual void postAct() = 0;
  virtual void sendExperience() { }
  virtual void postSendExperience() { }

  // Python module get maxNumAction from here.
  virtual int maxNumAction(const rela::Env& env) const { return env.maxNumAction(); }
  virtual int featureDim(const rela::Env& env) const { return env.featureDim(); }

 protected:
  // You might have multiple actors for multi-agent situations.
  std::vector<std::shared_ptr<rela::Actor2>> actors_;

  const EnvActorOptions options_;

  virtual std::string getSaveData() const { return ""; }
  virtual std::string getDisplayData() const { return ""; }

  void envTerminate(const std::vector<float>* rewards = nullptr) { 
    terminalCount_ ++;

    if (options_.savePrefix != "" && output_ != nullptr) {
      std::string savedData = getSaveData();
      if (savedData != "") {
        *output_ << savedData << std::endl;
      }
      // assert(database_->saveData(offset_ + currentIdx_, jsonStr));
    }

    if (options_.displayFreq > 0 && (rng_() % options_.displayFreq == 0)) {
      std::string displayData = getDisplayData();
      if (displayData != "") {
        std::cout << displayData << std::endl;
      }
    }

    // push rewards to the history 
    if (options_.eval && rewards != nullptr) {
      historyRewards_.push_back(*rewards);
    }
  }

  void terminateEnvActor() {
    isTerminated_ = true; 
    if (output_ != nullptr) { 
      output_->close();
      output_.reset(nullptr);
    }
  }

 private:
  std::shared_ptr<rela::FutureExecutor> executor_;

  // Reward of each player in each game.
  std::vector<std::vector<float>> historyRewards_;
  std::mt19937 rng_;

  int terminalCount_ = 0;
  bool isTerminated_ = false;
  std::unique_ptr<std::ofstream> output_;
};


