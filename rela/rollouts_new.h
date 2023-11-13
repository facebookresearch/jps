#include <memory>
#include <functional>

#include "rela/utils.h"
#include "rela/env.h"
#include "rela/actor2.h"
#include "rela/clock.h"

namespace rela {

namespace rollout_new {

class Rollout;

struct NodeStats {
  // Reachability for the current node. 
  float reach = 1.0f;
  // value (returned by neural network so it is already normalized by prob) of the curent node.
  float V = 0.0f;
};

struct Action {
 int idx = -1;
 int action = -1;
 float prob = 0.0f;
 float Q = 0.0f;
 int repeat = 1;
 float childReach = 0.0f;

 std::string info(std::function<std::string (int)> displayer = nullptr) const {
   std::stringstream ss;
   ss << "idx: " << idx << " ";
   if (displayer) {
     ss << displayer(action);
   } else {
     ss << action;
   } 
   ss << "/p:" << std::setprecision(4) << prob 
      << ",q:" << std::setprecision(4) << Q << ", rep: " << repeat;
   return ss.str();
 }
};

struct Result;

class Ctrl {
 public:
   // Before expansion, call preprocess until it return true (finished).
   virtual bool preprocess() { return true; }

   // Return true if we should stop the rollout now.
   virtual bool shouldStop(const Rollout &) const = 0;

   virtual std::tuple<rela::TensorDict, rela::TensorDictFuture> preAct(const Rollout &) = 0;

   // Input env and current Rollout Object
   // return value and vector of edge information.
   virtual std::tuple<float, std::vector<Action>> getCandidateActions(const Rollout &) const = 0;

   // Compute edge weight. This runs during getBest. 
   virtual std::tuple<std::vector<float>, float> getEdges(const Rollout &, const std::vector<Result> &subTreeResults) const = 0;

   virtual std::string actionDisplay(int action) const = 0;
};

struct Result {
  std::vector<std::vector<Action>> actions;
  std::vector<float> scores;

  // HACK.. v^\sigma in the tree.
  float u;

  float bestScore = -std::numeric_limits<float>::max();
  int bestActionIdx = -1;
  bool isLeaf;

  Result(bool isLeaf = false) : isLeaf(isLeaf) { }

  std::string info(const Ctrl &ctrl) const {
    std::stringstream ss;

    auto print = [&](const std::vector<Action>& acts) {
      for (const auto& a : acts) {
        ss << ctrl.actionDisplay(a.action) << "/" << std::setprecision(4) 
           << a.prob << " ";
      }
    };

    std::vector<std::pair<float, int>> scores2;
    for (int i = 0; i < (int)actions.size(); ++i) {
      scores2.emplace_back(-scores[i], i);
    }
    std::sort(scores2.begin(), scores2.end());

    for (int i = 0; i < (int)actions.size(); ++i) {
      int idx = scores2[i].second;
      ss << "   [";
      print(actions[idx]);
      ss << "], score: " << std::setprecision(10) << scores[idx] << std::endl;
    }

    if (bestActionIdx >= 0) {
      ss << "BestScore: " << std::setprecision(10) << bestScore << " u: " << u << " "; 
      ss << "Action Seq: ["; 
      print(actions[bestActionIdx]);
      ss << "]";
    }
    return ss.str();
  }

  const std::vector<Action> &bestActionSeq() const { 
    return actions.at(bestActionIdx);
  }

  void add(const Action &action, const Result &r, float edge) {
    std::vector<Action> actSeq;
    actSeq.push_back(action);

    if (!r.isLeaf) {
      auto bestActSeq = r.bestActionSeq();
      actSeq.insert(actSeq.end(), bestActSeq.begin(), bestActSeq.end());
      scores.push_back(r.bestScore);
    } else {
      scores.push_back(0.0f);
    }

    actions.push_back(actSeq);
    scores.back() += edge;
    bestActionIdx = -1;
  }

  void getBest() {
    if (bestActionIdx >= 0) return;

    bestScore = std::numeric_limits<float>::lowest();
    const auto it = std::max_element(scores.cbegin(), scores.cend());
    bestScore = *it;
    bestActionIdx = it - scores.cbegin();
  }
};

class Rollout {
 public:
   enum Status { INVALID = 0, PREPROCESSED, STARTED, TERMINATED, BRANCHED, EMPTY_ACTION };

   Rollout(const Env& s0, Ctrl &ctrl, 
           std::shared_ptr<FutureExecutor> executor) 
     : s0_(s0), ctrl_(ctrl), executor_(std::move(executor))
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
       case PREPROCESSED: return "PREPROCESSED";
       case TERMINATED: return "TERMINATED";
       case BRANCHED: return "BRANCHED";
       case EMPTY_ACTION: return "EMPTY_ACTION";
       default: return "INVALID";
     }
   }
   
   rela::TensorDict reply() const { return reply_; }
   rela::TensorDict obs() const { return obs_; }

   void run() {
     status_ = STARTED;

     if (!ctrl_.preprocess()) {
       executor_->addFuture([this]() {
           run();
       });
       return;
     }

     status_ = PREPROCESSED;

     START_TIMING("shouldStop")
     if (ctrl_.shouldStop(*this)) {
       // Terminal and save the reward.
       status_ = TERMINATED;
       return;
     }
     END_TIMING

     TensorDictFuture f;

     START_TIMING("rollout_preAct")
     std::tie(obs_, f) = ctrl_.preAct(*this);
     END_TIMING

     executor_->addFuture([f, this]() {
         START_TIMING("getFuture")
         reply_ = f();
         END_TIMING

         START_TIMING("getCandidateActions")
         std::tie(stats_.V, actions_) = ctrl_.getCandidateActions(*this);
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
             auto &a = actions_[j];
             a.idx = j;
             if (a.action < 0) {
               rollouts_[j].emplace_back(nullptr);
               continue;
             }

             // std::cout << "[" << j << "/" << actions.size() << "] Branching out with #rep: " << a.second << ", depth: " << depth_ << ", gen: " << generation_ << std::endl;
             auto s_next = getS().clone();
             s_next->step(a.action);

             for (int i = 0; i < a.repeat; ++i) {
               std::unique_ptr<Env> s2 = (i == a.repeat - 1) ? std::move(s_next) : s_next->clone();

               auto nextRollout = std::make_unique<Rollout>(
                   std::move(s2), ctrl_, executor_, depth_ + 1, generation_ + 1);

               nextRollout->stats_.reach = a.childReach; 

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
   
   Result getBest(bool verbose = false) const {
     if (rollouts_.empty() && status_ == TERMINATED) {
       return Result(true);
     }

     if (status_ != BRANCHED || rollouts_.empty()) {
       std::cout << "Rollout::getBest: Error: status: " << printStatus() 
                 << ", #rollout: " << rollouts_.size() << std::endl;
       std::cout << "State: " << getS().info() << std::endl;
       throw std::runtime_error("");
     }

     // Get edges. 
     std::vector<Result> subTreeResults(rollouts_.size());

     for (size_t i = 0; i < rollouts_.size(); ++i) {
       const auto &rs = rollouts_[i];

       // only use the first move.
       if (rs.size() != 1) {
         std::cout << "rs.size() == " << rs.size() << std::endl;
         std::cout << "Env: " << getS().info() << std::endl;
         auto f = [&](int action) { return ctrl_.actionDisplay(action); };
         for (const auto &a : actions_) {
           std::cout << a.info(f) << std::endl;
         }
       }
       assert(rs.size() == 1);
       subTreeResults[i] = rs[0]->getBest(verbose);
     }

     Result result;
     std::vector<float> edges;
     std::tie(edges, result.u) = ctrl_.getEdges(*this, subTreeResults);
     assert(edges.size() == rollouts_.size());

     for (size_t i = 0; i < rollouts_.size(); ++i) {
       result.add(actions_[i], subTreeResults[i], edges[i]);
     }

     result.getBest();
     if (verbose) {
       std::cout << result.info(ctrl_) << std::endl;
     }
     return result;
   }

   size_t getNumChildRollouts() const { return rollouts_.size(); }
   const Rollout *child(int action, int randomIdx) const { 
     for (int i = 0; i < (int)actions_.size(); ++i) {
       if (actions_[i].action == action) {
         return rollouts_[i][randomIdx].get(); 
       }
     }
     std::cout << "Action " + std::to_string(action) + " cannot be found!" << std::endl;
     throw std::runtime_error("Action " + std::to_string(action) + " cannot be found!");
   }

   const Action &action(int idx) const {
     assert(idx >= 0 && idx < (int)actions_.size());
     return actions_[idx]; 
   }

   int depth() const { return depth_; }
   int generation() const { return generation_; }
   Status status() const { return status_; }

   const Env& getS() const {
     if (s_ == nullptr) return s0_;
     return *s_;
   }

   const NodeStats& getNodeStats() const {
     return stats_;
   }

 private:
   std::unique_ptr<Env> s_;
   const Env& s0_;

   Ctrl &ctrl_;
   std::shared_ptr<FutureExecutor> executor_;

   rela::TensorDict obs_;
   rela::TensorDict reply_;
   NodeStats stats_;

   // Next generation rollouts / actions that lead to it.
   // rollouts_[action_idx][random_idx] is the next rollouts.
   std::vector<std::vector<std::unique_ptr<Rollout>>> rollouts_;
   std::vector<Action> actions_;

   // Statistics
   int depth_;
   int generation_;
   Status status_ = INVALID;

   void step(int a) {
     if (s_ == nullptr) {
       s_ = s0_.clone();
     }
     s_->step(a);
   }
};

}

}  // namespace rela
