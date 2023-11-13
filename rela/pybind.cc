// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rela/actor.h"
#include "rela/actor2.h"
#include "rela/random_actor.h"
#include "rela/context.h"
// #include "rela/dqn_actor.h"
#include "rela/a2c_actor.h"
#include "rela/env.h"
#include "rela/model.h"
#include "rela/prioritized_replay.h"
#include "rela/r2d2_actor.h"
#include "rela/r2d2_actor2.h"
#include "rela/thread_loop_env_actor.h"
#include "rela/env_actor.h"
#include "rela/search_actor_refactored.h"
#include "rela/search_actor_refactored_new.h"

// #include "rpc/rpc_env.h"
//
namespace py = pybind11;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<Transition, std::shared_ptr<Transition>>(m, "Transition")
      .def_readwrite("d", &Transition::d);

  py::class_<PrioritizedReplay2, std::shared_ptr<PrioritizedReplay2>>(
      m, "PrioritizedReplay2")
      .def(py::init<int,    // capacity,
                    int,    // seed,
                    float,  // alpha, priority exponent
                    float,  // beta, importance sampling exponent
                    bool,   // whther we do prefetch
                    int>())  //batchdim axis (usually it is 0, if we use LSTM then this can be 1)
      .def("size", &PrioritizedReplay2::size)
      .def("num_add", &PrioritizedReplay2::numAdd)
      .def("sample", &PrioritizedReplay2::sample)
      .def("update_priority", &PrioritizedReplay2::updatePriority)
      .def("keep_priority", &PrioritizedReplay2::keepPriority);

  py::class_<FFTransition, std::shared_ptr<FFTransition>>(m, "FFTransition")
      .def_readwrite("obs", &FFTransition::obs)
      .def_readwrite("action", &FFTransition::action)
      .def_readwrite("reward", &FFTransition::reward)
      .def_readwrite("terminal", &FFTransition::terminal)
      .def_readwrite("bootstrap", &FFTransition::bootstrap)
      .def_readwrite("next_obs", &FFTransition::nextObs);

  py::class_<RNNTransition, std::shared_ptr<RNNTransition>>(m, "RNNTransition")
      .def_readwrite("obs", &RNNTransition::obs)
      .def_readwrite("h0", &RNNTransition::h0)
      .def_readwrite("action", &RNNTransition::action)
      .def_readwrite("reward", &RNNTransition::reward)
      .def_readwrite("terminal", &RNNTransition::terminal)
      .def_readwrite("bootstrap", &RNNTransition::bootstrap)
      .def_readwrite("seq_len", &RNNTransition::seqLen);

  py::class_<FFPrioritizedReplay, std::shared_ptr<FFPrioritizedReplay>>(
      m, "FFPrioritizedReplay")
      .def(py::init<int,    // capacity,
                    int,    // seed,
                    float,  // alpha, priority exponent
                    float,
                    bool>())  // beta, importance sampling exponent
      .def("size", &FFPrioritizedReplay::size)
      .def("num_add", &FFPrioritizedReplay::numAdd)
      .def("sample", &FFPrioritizedReplay::sample)
      .def("update_priority", &FFPrioritizedReplay::updatePriority);

  py::class_<RNNPrioritizedReplay, std::shared_ptr<RNNPrioritizedReplay>>(
      m, "RNNPrioritizedReplay")
      .def(py::init<int,    // capacity,
                    int,    // seed,
                    float,  // alpha, priority exponent
                    float,
                    bool>())  // beta, importance sampling exponent
      .def("size", &RNNPrioritizedReplay::size)
      .def("num_add", &RNNPrioritizedReplay::numAdd)
      .def("sample", &RNNPrioritizedReplay::sample)
      .def("update_priority", &RNNPrioritizedReplay::updatePriority);

  py::enum_<PlayerGroup>(m, "PlayerGroup", py::arithmetic())
      .value("GRP_NATURE", PlayerGroup::GRP_NATURE)
      .value("GRP_1", PlayerGroup::GRP_1)
      .value("GRP_2", PlayerGroup::GRP_2)
      .value("GRP_3", PlayerGroup::GRP_3);

  py::class_<EnvSpec>(m, "EnvSpec")
      .def_readonly("feature_size", &EnvSpec::featureSize)
      .def_readonly("max_num_actions", &EnvSpec::maxNumActions)
      .def_readonly("players", &EnvSpec::players);

  py::class_<Env, std::shared_ptr<Env>>(m, "Env")
      .def("max_num_action", &Env::maxNumAction)
      .def("feature_dim", &Env::featureDim)
      .def("reset", &Env::reset)
      .def("step", &Env::step)
      .def("info", &Env::info)
      .def("spec", &Env::spec)
      .def("terminated", &Env::terminated);

  py::class_<BatchProcessorUnit, std::shared_ptr<BatchProcessorUnit>>(
      m, "BatchProcessor")
      .def(py::init<std::shared_ptr<ModelLocker>, const std::string&, int,
                    const std::string&>());

  py::class_<Models, std::shared_ptr<Models>>(m, "Models")
      .def(py::init<>())
      .def("add", &Models::add, py::keep_alive<1, 2>());

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  /*
  py::class_<BasicThreadLoop, ThreadLoop, std::shared_ptr<BasicThreadLoop>>(
      m, "BasicThreadLoop")
      .def(
          py::init<std::shared_ptr<Actor>, std::shared_ptr<VectorEnv>, bool>());
  */

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("push_env_thread", &Context::pushThreadLoop, py::keep_alive<1, 2>())
      .def("start", &Context::start)
      .def("pause", &Context::pause)
      .def("resume", &Context::resume)
      .def("terminate", &Context::terminate)
      .def("terminated", &Context::terminated);

  py::class_<ModelLocker, std::shared_ptr<ModelLocker>>(m, "ModelLocker")
      .def(py::init<const std::vector<py::object>&, const std::string&>())
      .def("update_model", &ModelLocker::updateModel);

  py::class_<Actor, std::shared_ptr<Actor>>(m, "Actor");
  py::class_<Actor2, std::shared_ptr<Actor2>>(m, "Actor2");

  py::class_<RandomActor, Actor2, std::shared_ptr<RandomActor>>(m, "RandomActor")
      .def(py::init<int, int>());

  py::class_<A2CActor, Actor2, std::shared_ptr<A2CActor>>(m, "A2CActor")
      .def(py::init<std::shared_ptr<Models>,                 // modelLocker
                    int,                                     // multiStep
                    float,                                   // gamma
                    std::shared_ptr<PrioritizedReplay2>>())  // replayBuffer
      .def(py::init<std::shared_ptr<Models>>())              // evaluation mode
      .def("num_act", &A2CActor::numAct);

  py::class_<R2D2Actor, Actor, std::shared_ptr<R2D2Actor>>(m, "R2D2Actor")
      .def(py::init<std::shared_ptr<ModelLocker>,              // modelLocker
                    int,                                       // multiStep
                    int,                                       // batchsize
                    float,                                     // gamma
                    int,                                       // seqLen
                    float,                                     // greedyEps
                    int,                                       // numPlayer
                    std::shared_ptr<RNNPrioritizedReplay>>())  // replayBuffer
      .def(py::init<std::shared_ptr<ModelLocker>, int, float>())  // evaluation
                                                                  // mode
      .def("num_act", &R2D2Actor::numAct);

  py::class_<R2D2Actor2, Actor2, std::shared_ptr<R2D2Actor2>>(m, "R2D2Actor2")
      .def(py::init<std::shared_ptr<Models>,                 // modelLocker
           int,                                              // multiStep
           float,                                            // gamma
           int,                                              // seqLen
           int,                                              // burnin
           float,                                            // greedyEps
           std::shared_ptr<PrioritizedReplay2>>())
      .def("num_act", &R2D2Actor2::numAct);


  py::class_<EnvActorOptions, std::shared_ptr<EnvActorOptions>>(m, "EnvActorOptions")
      .def(py::init<>())
      .def_readwrite("thread_idx", &EnvActorOptions::threadIdx)
      .def_readwrite("save_prefix", &EnvActorOptions::savePrefix)
      .def_readwrite("display_freq", &EnvActorOptions::displayFreq)
      .def_readwrite("seed", &EnvActorOptions::seed)
      .def_readwrite("empty_init", &EnvActorOptions::emptyInit)
      .def_readwrite("eval", &EnvActorOptions::eval);

  py::class_<SearchActorOptions, std::shared_ptr<SearchActorOptions>>(m, "SearchActorOptions")
      .def(py::init<>())
      .def_readwrite("search_ratio", &SearchActorOptions::searchRatio)
      .def_readwrite("verbose_freq", &SearchActorOptions::verboseFreq)
      .def_readwrite("update_count", &SearchActorOptions::updateCount)
      .def_readwrite("use_tabular_ref", &SearchActorOptions::useTabularRef)
      .def_readwrite("baseline_ratio", &SearchActorOptions::baselineRatio)
      .def_readwrite("use_hacky", &SearchActorOptions::useHacky)
      .def_readwrite("use_grad_update", &SearchActorOptions::useGradUpdate)
      .def_readwrite("best_on_best", &SearchActorOptions::bestOnBest);

  py::class_<EnvActorMoreOptions>(m, "EnvActorMoreOptions")
      .def(py::init<>())
      .def_readwrite("use_sad", &EnvActorMoreOptions::useSad);

  py::class_<EnvActorBase, std::shared_ptr<EnvActorBase>>(m, "EnvActorBase")
      .def("max_num_action", &EnvActorBase::maxNumAction)
      .def("feature_dim", &EnvActorBase::featureDim)
      .def("get_history_rewards", &EnvActorBase::getHistoryRewards); 

  // Batch free version.
  py::class_<EnvActor, EnvActorBase, std::shared_ptr<EnvActor>>(m, "EnvActor")
      .def(py::init<std::shared_ptr<rela::Env>,
                    std::vector<std::shared_ptr<rela::Actor2>>,
                    const EnvActorOptions&, 
                    const EnvActorMoreOptions&>(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());

  // Search version.
  // [TODO] Should be deprecated soon after the new search works. 
  py::class_<SearchActor, EnvActorBase, std::shared_ptr<SearchActor>>(m, "SearchActor")
      .def(py::init<std::shared_ptr<rela::Env>,
                    std::vector<std::shared_ptr<rela::Actor2>>,
                    const EnvActorOptions&, 
                    const SearchActorOptions&>(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());

  py::class_<SearchActorNew, EnvActorBase, std::shared_ptr<SearchActorNew>>(m, "SearchActorNew")
      .def(py::init<std::shared_ptr<rela::Env>,
                    std::vector<std::shared_ptr<rela::Actor2>>,
                    const EnvActorOptions&, 
                    const SearchActorOptions&>(),
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());

  py::class_<ThreadLoopEnvActor,
             rela::ThreadLoop,
             std::shared_ptr<ThreadLoopEnvActor>>(m, "ThreadLoopEnvActor")
      .def(py::init<int,                 // thread_id 
                    const std::vector<std::shared_ptr<EnvActorBase>>&,  // vector of environments/actors.
                    int>());             // #games run before ending.

}
