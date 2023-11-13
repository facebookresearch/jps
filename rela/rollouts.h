#include <memory>
#include <functional>

#include "rela/utils.h"
#include "rela/env.h"
#include "rela/actor2.h"
#include "rela/clock.h"

namespace rela {

namespace rollout {

class Rollout;

struct Action {
 int action = -1;
 float penalty = 0.0f;
 int repeat = 1;
};

class Ctrl {
 public:
   // Return (true, final_reward) if we should stop the rollout now.
   // Otherwise return (false, 0.0)
   virtual std::pair<bool, float> shouldStop(const Env &, const Rollout &) const = 0;

   virtual TensorDictFuture preAct(const Env &, const Rollout &) = 0;

   // Input env and current Rollout Object
   //
   // Return each action and its number of repetitions. 
   virtual std::vector<Action> getCandidateActions(const Env &, const Rollout &, TensorDictFuture) = 0;

   virtual std::string actionDisplay(int action) const = 0;
};

struct Result {
  int action;
  float reward;
  float undiscountedReward;
  int n;

  void initAvg() {
    reward = undiscountedReward = 0.0f;
    n = 0;
  }

  void initMax() {
    reward = undiscountedReward = -std::numeric_limits<float>::max();
    n = 1;
  }

  Result &operator+=(const Result &r) {
    n += r.n;
    reward += r.reward;
    undiscountedReward += r.undiscountedReward;
    return *this;
  }

  void avg() {
    reward /= n;
    undiscountedReward /= n;
    n = 1;
  }

  Result &argMax(const Result &r) {
    assert(n == 1);

    if (reward < r.reward) {
      reward = r.reward;
      undiscountedReward = r.undiscountedReward; 
      action = r.action;
    }
    return *this;
  }

  std::string info(const Ctrl &ctrl) const {
    std::stringstream ss;
    ss << ctrl.actionDisplay(action) << ", reward: " << reward << ", undiscounted: " << undiscountedReward;
    return ss.str();
  }
};

class Rollout {
 public:
   enum Status { INVALID = 0, STARTED, TERMINATED, BRANCHED, EMPTY_ACTION };

   Rollout(const Env& s0, Ctrl &ctrl, 
           std::shared_ptr<FutureExecutor> executor) 
     : s0_(s0), ctrl_(ctrl), executor_(std::move(executor))
     , reward_(-std::numeric_limits<float>::max())
     , depth_(0), generation_(0) {
   } 

   Rollout(std::unique_ptr<Env> &&s, Ctrl &ctrl, 
           std::shared_ptr<FutureExecutor> executor, int depth, int generation) 
     : s_(std::move(s)), s0_(*s_), ctrl_(ctrl), executor_(std::move(executor)), 
       depth_(depth), generation_(generation) {
     assert(s_ != nullptr);
   } 

   std::string printStatus() const {
     switch (status_) {
       case STARTED: return "STARTED";
       case TERMINATED: return "TERMINATED";
       case BRANCHED: return "BRANCHED";
       case EMPTY_ACTION: return "EMPTY_ACTION";
       default: return "INVALID";
     }
   }

   void run() {
     status_ = STARTED;

     START_TIMING("shouldStop")
     auto res = ctrl_.shouldStop(getS(), *this);
     if (res.first) {
       // Terminal and save the reward.
       reward_ = res.second;
       status_ = TERMINATED;
       return;
     }
     END_TIMING

     TensorDictFuture f;

     START_TIMING("rollout_preAct")
     f = ctrl_.preAct(getS(), *this);
     END_TIMING

     executor_->addFuture([f, this]() {
         START_TIMING("getCandidateActions")
         actions_ = ctrl_.getCandidateActions(getS(), *this, f);
         END_TIMING

         // Terminated.
         if (actions_.empty()) {
            status_ = EMPTY_ACTION;
            return;
         }

         // If there is only one action and that action repeats once.
         if (actions_.size() == 1 && actions_[0].repeat == 1) {
           // Normal rollout, reuse the current data structure.
           // std::cout << "Normal rollout " << ", depth: " << depth_ << ", gen: " << generation_ << std::endl;
           START_TIMING("normal_rollout")

           step(actions_[0].action);
           depth_ ++;
           executor_->addFuture([this]() {
             this->run();
           });

           END_TIMING
         } else {
           // Branching out.
           //
           START_TIMING("branching")
           rollouts_.resize(actions_.size());

           for (size_t j = 0; j < actions_.size(); ++j) {
             const auto &a = actions_[j];

             // std::cout << "[" << j << "/" << actions.size() << "] Branching out with #rep: " << a.second << ", depth: " << depth_ << ", gen: " << generation_ << std::endl;
             auto s_next = getS().clone();
             s_next->step(a.action);

             for (int i = 0; i < a.repeat; ++i) {
               std::unique_ptr<Env> s2 = (i == a.repeat - 1) ? std::move(s_next) : s_next->clone();

               auto nextRollout = std::make_unique<Rollout>(std::move(s2), ctrl_, executor_, depth_ + 1, generation_ + 1);
               rollouts_[j].push_back(std::move(nextRollout));
               executor_->addFuture([r = rollouts_[j].back().get()]() mutable {
                 r->run();
               });
             }
           }
           END_TIMING

           status_ = BRANCHED;
         }
     });
   }
   
   Result getAvg() const {
     if (rollouts_.empty() && status_ == TERMINATED) {
       Result res;
       res.reward = reward_;
       res.undiscountedReward = reward_;
       res.n = 1;

       return res;
     }

     // Otherwise, get average rewards from the children.
     if (status_ == BRANCHED && ! rollouts_.empty()) {
       Result result;
       result.initAvg();

       for (const auto &rs : rollouts_) {
         for (const auto &r : rs) {
           result += r->getAvg();
         }
       }

       result.avg();
       return result;
     }

     // error! 
     std::cout << "Rollout::getAvgReward: Error: status: " << printStatus() << ", #rollout: " << rollouts_.size() << std::endl;
     std::cout << "State: " << getS().info() << std::endl;
     throw std::runtime_error("");
   }

   Result getBest(bool bestOnBest, bool verbose) const {
     if (rollouts_.empty() && status_ == TERMINATED) {
       Result res;
       res.reward = reward_;
       res.undiscountedReward = reward_;
       res.n = 1;

       return res;
     }

     if (status_ != BRANCHED || rollouts_.empty()) {
       std::cout << "Rollout::getBest: Error: status: " << printStatus() << ", #rollout: " << rollouts_.size() << std::endl;
       std::cout << "State: " << getS().info() << std::endl;
       throw std::runtime_error("");
     }

     Result result;
     result.initMax();

     if (verbose) {
       std::cout << "Rollout Print: " << std::endl;
     }

     for (size_t i = 0; i < rollouts_.size(); ++i) {
       const auto &rs = rollouts_[i];

       Result resultAction;
       resultAction.initAvg();

       for (const auto &r : rs) {
         if (bestOnBest) resultAction += r->getBest(bestOnBest, verbose);
         else resultAction += r->getAvg();
       }

       resultAction.avg();

       // Apply penalty for selecting different actions (e.g. some action has low probability). 
       resultAction.reward -= actions_[i].penalty;
       resultAction.action = actions_[i].action;

       if (verbose) {
         std::cout << "  " << resultAction.info(ctrl_) << std::endl;
       }

       result.argMax(resultAction);
     }

     if (verbose) {
       std::cout << "Rollout Best: " << result.info(ctrl_) << std::endl;
     }

     return result;
   }

   size_t getNumChildRollouts() const { return rollouts_.size(); }
   int depth() const { return depth_; }
   int generation() const { return generation_; }
   Status status() const { return status_; }

 private:
   std::unique_ptr<Env> s_;
   const Env& s0_;

   Ctrl &ctrl_;
   std::shared_ptr<FutureExecutor> executor_;

   float reward_;

   // Next generation rollouts / actions that lead to it.
   // rollouts_[action_idx][random_idx] is the next rollouts.
   std::vector<std::vector<std::unique_ptr<Rollout>>> rollouts_;
   std::vector<Action> actions_;

   // Statistics
   int depth_;
   int generation_;
   Status status_ = INVALID;

   const Env& getS() const {
     if (s_ == nullptr) return s0_;
     return *s_;
   }

   void step(int a) {
     if (s_ == nullptr) {
       s_ = s0_.clone();
     }
     s_->step(a);
   }
};

}

}  // namespace rela
