#pragma once

#include "rela/a2c_actor.h"
#include "rela/model.h"
#include "rela/prioritized_replay2.h"
#include "rela/thread_loop.h"
#include "rela/utils.h"
#include "rela/clock.h"

#include "rela/rollouts_new.h"
#include "rela/env_actor_base.h"
#include "rela/freq_utils.h"

#include "simple_game/comm.h"
#include "simple_game/search.h"
#include "rela/search_actor_options.h"

using RolloutNew = rela::rollout_new::Rollout;

// TODO: Update it to most recent version (new CFR)
class TabularSolver {
 public:
  struct InfoSetData {
    rela::TensorDict obs, reply;
  };

  void setEnvInit(const rela::Env& env) {
    tabular::Options options;
    solver_ = std::make_unique<tabular::search::Solver>(options);

    bool keepEnvInNode = true;
    solver_->init(env, keepEnvInNode);
    allKeys_ = solver_->manager().allInfoSetKeys();
    // std::cout << "Keys: " << allKeys_ << std::endl;
  }

   void reset() {
     infoKeyIdx_ = -1;
     preprocessFuture_ = nullptr;
     onReply_ = nullptr;
     data_.clear();
   }

   bool preprocess(std::function<rela::TensorDictFuture (rela::TensorDict, int)> actor, std::ostream *oo) {
     if (infoKeyIdx_ >= (int)allKeys_.size()) return true;

     if (infoKeyIdx_ >= 0) {
       auto reply = preprocessFuture_();
       onReply_(reply);
     }

     const rela::Env *env = nullptr;
     std::string infoSetKey;

     while (true) {
       infoKeyIdx_ ++; 
       if (infoKeyIdx_ >= (int)allKeys_.size()) return true;

       infoSetKey = allKeys_[infoKeyIdx_];
       const auto &states = solver_->manager()[infoSetKey].states();
       if (states.empty()) {
         std::cout << infoSetKey << " has no nodes! " << std::endl;
       }
       assert(!states.empty());

       // Any complete state from infoSet is fine.
       const auto &s = states[0];

       // If it is a chance node then we skip (no policy here).
       if (!s->infoSet().isChance()) {
         env = s->env(); 
         if (env == nullptr) {
           std::cout << infoSetKey << ": env is nullptr! " << std::endl;
         }
         assert(env != nullptr);
         if (!env->subgameEnd()) break;
       }
     } 

     int playerIdx = env->playerIdx();
     auto obs = env->feature();
     data_[infoSetKey].obs = obs;

     preprocessFuture_ = actor(obs, playerIdx);

     onReply_ = [this, infoSetKey, env, oo](const rela::TensorDict &reply) {
       // Fill in table
       data_[infoSetKey].reply = reply; 
       auto policy = rela::utils::getVectorSel(reply, "pi", env->legalActions());
       if (oo != nullptr) {
         *oo << infoSetKey << " " << policy << std::endl;
       }
       solver_->manager()[infoSetKey].setStrategy(policy);
     };

     // 
     return false;
   }

   const InfoSetData *query(const rela::Env& env) const {
     auto infoSetKey = env.infoSet();
     auto it = data_.find(infoSetKey);
     if (it == data_.end()) return nullptr;
     // Return stored.
     return &it->second;
   }

   const tabular::search::Solver &solver() const { return *solver_; }

   std::string solve(const rela::Env &env, int playerIdx) {
     auto info = solver_->manager().getInfoSetSharedPtr(env.infoSet());
     tabular::search::InfoSets infoSets;
     infoSets.emplace_back(std::move(info));
     auto result = solver_->searchOneIter(infoSets, playerIdx);

     auto node = solver_->manager().getState(env);
     nodeReach_ = node->totalReach();
     return result.info(solver_->manager());
   }

   float nodeReach() const { return nodeReach_; }

 private:
   std::unique_ptr<tabular::search::Solver> solver_;
   std::vector<std::string> allKeys_;

   rela::TensorDictFuture preprocessFuture_;
   std::function<void (const rela::TensorDict &)> onReply_; 
   int infoKeyIdx_;
   std::unordered_map<std::string, InfoSetData> data_;
   
   float nodeReach_;
};

class SearchRolloutCtrlNew : public rela::rollout_new::Ctrl {
 public:
   using Action = rela::rollout_new::Action;
   
   SearchRolloutCtrlNew(std::vector<std::shared_ptr<rela::Actor2>> actors, 
       const Freqs& freqs, const SearchActorOptions &options)
     : actors_(std::move(actors))
     , freqs_(freqs)
     , options_(options) {
       if (options_.useTabularRef) { 
         tabularSolver_ = std::make_unique<TabularSolver>();
       }
   }

   void setEnvInit(const rela::Env &env) {
     if (tabularSolver_ != nullptr) {
       tabularSolver_->setEnvInit(env);
     }
   }

   void initialize(bool skipRollout, int requestPlayerIdx, int requestPartnerIdx, rela::TensorDict rootObs) {
     rootObs_ = rootObs;

     skipRollout_ = skipRollout;
     requestPlayerIdx_ = requestPlayerIdx;
     requestPartnerIdx_ = requestPartnerIdx;

     if (tabularSolver_ != nullptr) {
       tabularSolver_->reset();
     }

     if (verbose_) {
       debugStream_ = std::make_unique<std::stringstream>();
     } else {
       debugStream_ = nullptr;
     }
   }

   bool verbose() const { return verbose_; }
   void setVerbose(bool verbose) { verbose_ = verbose; }

   bool preprocess() override { 
     if (tabularSolver_ != nullptr) {
       auto actor =
           [&](rela::TensorDict obs, int playerIdx) {
             return actors_[playerIdx]->act(obs);
           };
       return tabularSolver_->preprocess(actor, debugStream_.get());
     } else {
       return true;
     }
   }

   bool shouldStop(const RolloutNew &r) const override {
     const auto& env = r.getS();

     // Note that we don't stop if the root node has subgameEnd = true, 
     //      since it means the starting of the second table. 
     return r.depth() > 0 && env.subgameEnd();
   }

   std::tuple<rela::TensorDict, rela::TensorDictFuture> preAct(const RolloutNew &r) override {
     const auto& env = r.getS();

     if (tabularSolver_ != nullptr) {
       const auto* data = tabularSolver_->query(env);
       if (data != nullptr) {
         return {data->obs, [data]() { return data->reply; }};
       }
     }

     rela::TensorDict obs;

     START_TIMING("feature")
     obs = (r.depth() == 0 ? rootObs_ : env.feature());
     END_TIMING

     // auto obs = env.feature();
     int playerIdx = env.playerIdx();

     //std::cout << "playerIdx: " << playerIdx << std::endl;
     rela::TensorDictFuture f;

     START_TIMING("act")
     f = actors_[playerIdx]->act(obs);
     END_TIMING
      
     return { obs, f };
   }

   // Return action set, and reward.
   // If the action set is empty, then the reward is the final reward. 
   // If the action set is non-empty, then the reward is 0. 
   std::tuple<float, std::vector<Action>> getCandidateActions(const RolloutNew &r) const override {
     auto reply = r.reply();

     const auto& env = r.getS();
     // int playerIdx = env.playerIdx();

     // std::cout << "SearchCtrl: Getting reply . playerIdx: " << playerIdx << ", requestPlayerIdx: " << requestPlayerIdx_ << ", requestPartnerIdx: " << requestPartnerIdx_ 
     //           << ", r.depth(): " << r.depth() << ", r.generation(): " << r.generation() << std::endl;
     rela::utils::assertKeyExists(reply, { "a" });

     // Note that if the root is terminal, then we still allow getCandidateActions to be called, and stop here.
     // TODO: This is very weird and we would like to fix this design.
     if (skipRollout_ || env.subgameEnd()) return { 0.0f, {} };

     float V = rela::utils::getTensorDictScalar<float>(reply, "v");

     // std::cout << "got reply, cnt: " << rolloutCnt_ << std::endl;
     // std::cout << "*****************  " << reply["a"].item<int>() << std::endl;
     // std::cout << "before step, cnt: " << rolloutCnt_ << std::endl;
     //
     std::vector<Action> candidates;
     int rep = 1;
     const float reach = r.getNodeStats().reach;

     START_TIMING( "player_choice")
     SVec pi, Q;
     std::tie(pi, Q) = getPiQLegal(env, V, reply);

     for (int i = 0; i < (int)pi.size(); ++i) {
       Action a;
       a.action = pi[i].second; 
       a.prob = pi[i].first;
       a.Q = Q[i].first;
       a.repeat = rep;
       a.childReach = reach * a.prob;
       candidates.push_back(std::move(a));
     }
     END_TIMING

     assert(candidates.size() >= 1);
     return { V, candidates };
   }

   std::tuple<std::vector<float>,float> 
     getEdges(const RolloutNew &r, const std::vector<rela::rollout_new::Result> &subTreeResults) const override {
     // Parent environment. 
     const auto& env = r.getS();

     float sampleReach = r.getNodeStats().reach;
     float V = r.getNodeStats().V;

     // check freqs_
     // Prob that the complete info happens in the infoset.
     // Large prob -> you have more freedom to choose your action. 
     // Freq::Item freq = freqs_.get(TrajItem(env.infoSet(), env.completeCompactDesc())); 

     int numAction = r.getNumChildRollouts();

     // use CFR to get accurate estimation of Q.
     if (tabularSolver_ != nullptr) {
       const auto& infoSet = tabularSolver_->solver().manager()[env.infoSet()];
       float diffNorm = 0.0f;
       float gtNorm = 0.0f;

       float meanEst = 0.0f;
       float meanGt = 0.0f;

       for (int i = 0; i < numAction; ++i) {
         const Action &a = r.action(i);

         // Using the right Q but estimated frequency.
         // Compare estimated Q and real Q.
         float est = a.Q * a.prob;
         float gt = infoSet.q()[i] * a.prob;
         meanEst += est;
         meanGt += gt;

         gtNorm += gt * gt;
         float d = gt - est;
         diffNorm += d * d;
       }

       diffNorm = sqrt(diffNorm);
       gtNorm = sqrt(gtNorm);
       meanEst /= numAction;
       meanGt /= numAction;

       int errPolarity = 0;
       for (int i = 0; i < numAction; ++i) {
         const Action &a = r.action(i);
         float est = a.Q * a.prob;
         float gt = infoSet.q()[i] * a.prob;
         errPolarity += ((gt - meanGt) * (est - meanEst) < 0 ? 1 : 0); 
       }

       if (verbose_) {
         /*
         *debugStream_ << "getEdges[" << env.infoSet()
                       << "] reach gt: " << infoSet->totalReach()
                       << " estimate: " << freq.freqInfoSet << std::endl;
        */
         /*
          *debugStream_ << "getEdges[" << env.infoSet() << "] est: " << estQ
                        << "[" << freq.freqInfoSet << "]"
                        << " gt: " << gtQ << "[" << infoSet->totalReach() << "]"
                        << std::endl;
         */
         *debugStream_ << "getEdges[" << env.infoSet()
                       << "] relative_err: " << diffNorm / gtNorm
                       << ", diffNorm: " << diffNorm << ", gtNorm: " << gtNorm
                       << "errPolarity: " << errPolarity << "/" << numAction
                       << std::endl;
       }
     }

     const float coeff = options_.baselineRatio * sampleReach;

     std::vector<float> edges(numAction, 0);
     float u = 0;
     float meanU = 0;
     float meanQ = 0;

     // Loop over children and compute the edge accordingly
     for (int i = 0; i < numAction; ++i) {
       const Action &a = r.action(i);
       const auto &c = *r.child(a.action, 0);

       // Approx q with value function in the next stage.
       bool childTerminal = c.getS().subgameEnd(); 
       float childV = childTerminal ? c.getS().playerRawScore(requestPlayerIdx_) : subTreeResults[i].u;

       // connect to a terminal state. 
       // [TODO] assume collaborative...
       // edges[i] = q + (1 - sampleReach) * reward;
       // edges[i] = q * a.prob + (1 - sampleReach * a.prob) * reward;
       // edges[i] = a.Q * a.prob * reach + (1 - sampleReach * a.prob) * reward;
       edges[i] = (1 - coeff) * childV + coeff * a.Q;

       u += a.prob * childV; 
       meanU += childV;
       meanQ += a.Q;
     } 

     if (options_.useGradUpdate) {
       meanU /= numAction - 1;
       meanQ /= numAction - 1;
       for (int i = 0; i < numAction; ++i) {
         edges[i] *= static_cast<float>(numAction) / (numAction - 1);
         edges[i] -= coeff * (V + meanQ) + (1 - coeff) * (u + meanU);
       }
     } else {
       for (int i = 0; i < numAction; ++i) {
         edges[i] -= coeff * V + (1 - coeff) * u; 
       }
     }

     /*
     if (verbose_) {
       // action -> edgeWeight (note that edges is idx -> edgeWeight)
       std::vector<float> edgeActions(numAction);
       for (int i = 0; i < numAction; ++i) {
         const Action &a = r.action(i);
         edgeActions[a.action] = edges[i];
       }

       *debugStream_ << env.infoSet() << ": Q: " << Q 
                     << " V: " << sumJ2 << " edges: " << edgeActions << std::endl;
     }
     */

     // Final action. 
     // edges[numAction - 1] = (1 - sampleReach) * V; 

     return {edges, u};
   }

   std::string actionDisplay(int action) const override {
     // Action string.
     return std::to_string(action);
   }

   std::string startSolve(const rela::Env& env) {
     // Run search.
     std::string result;
     if (tabularSolver_ != nullptr) {
       result = tabularSolver_->solve(env, requestPlayerIdx_);
       nodeReach_ = tabularSolver_->nodeReach();
     } 
     //else {
     Freq::Item freq = freqs_.get(TrajItem(env.infoSet(), env.completeCompactDesc()));
     if (verbose_) {
       *debugStream_ << env.completeCompactDesc()
                     << ", nodeReach: gt: " << nodeReach_
                     << ", estimate: " << freq.freq << std::endl;
     }
     nodeReach_ = freq.freq;
     //}
     return result;
   }

   std::string getDebugInfo() {
     if (verbose_) return debugStream_->str();
     else return "";
   }

 private:
   std::vector<std::shared_ptr<rela::Actor2>> actors_;

   rela::TensorDict rootObs_;

   bool skipRollout_ = false;
   int requestPlayerIdx_ = -1;
   int requestPartnerIdx_ = -1;

   const Freqs& freqs_;
   const SearchActorOptions options_;
   float nodeReach_;

   std::unique_ptr<TabularSolver> tabularSolver_;

   bool verbose_ = false;
   std::unique_ptr<std::stringstream> debugStream_;

   using SVec = std::vector<std::pair<float, int>>;

   // Return pi and q
   std::tuple<SVec, SVec> getPiQLegal(const rela::Env &env, float V, const rela::TensorDict& reply) const {
     assert(rela::utils::hasKey(reply, "behavior_pi"));
     assert(rela::utils::hasKey(reply, "adv"));

     auto legalActions = env.legalActions();
     SVec pi = rela::utils::getVectorSelPair(reply, "behavior_pi", legalActions);
     SVec q = rela::utils::getVectorSelPair(reply, "adv", legalActions);
     for (auto &qq : q) {
       qq.first += V;
     }

     return { pi, q };
   }
};

class SearchActorNew : public EnvActorBase {
 public:
  SearchActorNew(std::shared_ptr<rela::Env> env,
           std::vector<std::shared_ptr<rela::Actor2>> actors, 
           const EnvActorOptions& options,
           const SearchActorOptions &searchOptions) 
      : EnvActorBase(actors, options)
      , env_(std::move(env))
      , searchCtrl_(actors_, freqs_, searchOptions)
      , searchOptions_(searchOptions) {
    checkValid(*env_);
    for (int i = 0; i < (int)actors_.size(); ++i) {
      replays_.emplace_back();
    }
    env_->reset();
    spec_ = env_->spec();
    searchCtrl_.setEnvInit(*env_);
  }

  void preAct() override {
    int playerIdx = env_->playerIdx();

    bool skipRollout = justRolloutPolicy_;
    if (spec_.players[playerIdx] == rela::PlayerGroup::GRP_NATURE || options_.eval) {
      // Do not optimize nature.
      skipRollout = true; 
    }

    rela::TensorDict obs;

    START_TIMING("feature")
    obs = env_->feature();
    END_TIMING

    std::vector<int> partnerIndices = env_->partnerIndices(playerIdx);

    int partnerIdx = partnerIndices.size() == 1 ? partnerIndices[0] : -1;

    bool verboseFreq = searchOptions_.verboseFreq > 0 && random() % searchOptions_.verboseFreq == 0;
    searchCtrl_.setVerbose(verboseFreq);
    searchCtrl_.initialize(skipRollout, playerIdx, partnerIdx, obs);

    // TODO one extra copy of the environment.
    root_ = std::make_unique<RolloutNew>(*env_, searchCtrl_, getExecutor());
    root_->run();

    // std::cout << "EnvActor: After preAct " << std::endl;
  }

  void postAct() override {
    //std::cout << "EnvActor: Before processRequest " << std::endl;
    using Status = rela::rollout_new::Rollout::Status;

    START_TIMING("get_result")

    // Get all recurrent actions.
    std::vector<rela::rollout_new::Action> actionSeq;

    auto status = root_->status(); 
    if (status == Status::BRANCHED) {
      // call it to get CFR results.
      auto tabularSearchResult = searchCtrl_.startSolve(*env_);

      rela::rollout_new::Result result = root_->getBest();
      actionSeq = result.bestActionSeq();

      if (searchCtrl_.verbose()) {
        // Print out everthing.
        std::stringstream ss;
        ss << "=========== Debug =================" << std::endl;
        ss << "infoSet: " << env_->infoSet()
           << ", completeInfo: " << env_->completeCompactDesc() << std::endl;
        ss << result.info(searchCtrl_) << std::endl;
        ss << searchCtrl_.getDebugInfo() << std::endl;
        ss << "Tabular result candidates:" << std::endl
           << tabularSearchResult << std::endl;
        // Print out reachability
        // ss << freqs_.info() << std::endl;
        ss << "=========== Debug End ============" << std::endl << std::endl;
        std::cout << ss.str();
      }
    } else {
      // Otherwise it is just regular rollouts. 
      // In this case, reply["a"] already have content and nothing you need to do. 
      int action = rela::utils::getTensorDictScalar<long>(root_->reply(), "a"); 
      rela::rollout_new::Action a;
      a.action = action;
      actionSeq.push_back(a);
    }

    const auto *rollout = root_.get();

    for (const auto& action : actionSeq) {
      auto obs = rollout->obs();
      auto reply = rollout->reply(); 
      auto playerIdx = env_->playerIdx();

      reply["a"][0] = action.action;

      if (justRolloutPolicy_) {
        // Only add stats when justRolloutPolicy_ is true (since we want to get \pi^\sigma(I))
        traj_.emplace_back(env_->infoSet(), env_->completeCompactDesc());
      }

      env_->step(action.action);

      // std::cout << "EnvActor: After step" << std::endl;
      reply["reward"] = torch::zeros({1});
      reply["terminal"] = torch::zeros({1}, torch::kBool);

      if (justRolloutPolicy_) {
        reply["search"] = torch::zeros({1}, torch::kBool);
      } else {
        reply["search"] = torch::ones({1}, torch::kBool);
      }

      rela::utils::appendTensorDict(obs, reply);
      replays_[playerIdx].push_back(obs);

      if (status == Status::BRANCHED) {
        rollout = rollout->child(action.action, 0);
      }
    } 
    END_TIMING

    // Some env (e.g.. Bridge) will have multiple subgames in one game. 
    bool subgameEnd = env_->subgameEnd();
    bool terminated = env_->terminated();

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
    if (! env_->terminated()) 
      return;

    // std::cout << "Reseting environment" << std::endl;
    // Row a dice and decide whether we just rollout policy or use search. 
    if (justRolloutPolicy_) {
      // Fill in freqs_
      // TODO: Only works for tabular case. We want to make it a neural network
      // for general case.
      freqs_ = globalIncFreqTable(traj_, searchOptions_.updateCount);
      traj_.clear();
    }
    justRolloutPolicy_ = (random() % rngConst >= searchOptions_.searchRatio * rngConst);

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

  bool justRolloutPolicy_ = true;

  // replays, one for each actor.
  std::vector<std::deque<rela::TensorDict>> replays_;

  // Freq table.
  Freqs freqs_;
  std::vector<TrajItem> traj_;
 
  // For rollouts. 
  SearchRolloutCtrlNew searchCtrl_;
  std::unique_ptr<RolloutNew> root_;
  const SearchActorOptions searchOptions_;

  const int rngConst = 10000;
};
