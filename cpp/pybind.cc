#include <memory>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cpp/allpass_actor2.h"
#include "cpp/baseline_actor2.h"
#include "cpp/belief_transfer_actor.h"
#include "cpp/bridge_env.h"
#include "cpp/console_actor.h"
#include "cpp/console_messenger.h"
#include "cpp/cross_bidding_actor.h"
#include "cpp/duplicate_bridge_env.h"
#include "cpp/greedy_play_actor.h"
#include "cpp/random_actor.h"

namespace py = pybind11;
// using namespace bridge;

PYBIND11_MODULE(bridge, m) {
  py::class_<DBInterface, std::shared_ptr<DBInterface>>(m, "DBInterface")
      .def(py::init<const std::string&, const std::string&, int>())
      .def("get_dataset_size", &DBInterface::getDatasetSize)
      .def("get_num_threads", &DBInterface::getNumThreads);

  py::class_<bridge::ConsoleMessenger,
             std::shared_ptr<bridge::ConsoleMessenger>>(m, "ConsoleMessenger")
      .def("read_env_msg", &bridge::ConsoleMessenger::read_env_msg)
      .def("send_env_msg", &bridge::ConsoleMessenger::send_env_msg)
      .def("start", &bridge::ConsoleMessenger::start)
      .def("stop", &bridge::ConsoleMessenger::stop)
      .def_static("init_messenger", &bridge::ConsoleMessenger::init_messenger)
      .def_static("get_messenger", &bridge::ConsoleMessenger::get_messenger);

  py::class_<bridge::BridgeEnv, rela::Env, std::shared_ptr<bridge::BridgeEnv>>(
      m, "BridgeEnv")
      .def(
          py::init<std::shared_ptr<DBInterface>,
                   const std::unordered_map<std::string, std::string>&, bool>(),
          py::keep_alive<1, 2>())
      .def("curr_json_str", &bridge::BridgeEnv::currJsonStr)
      .def("step", &bridge::BridgeEnv::step)
      .def("feature", &bridge::BridgeEnv::feature)
      .def("has_opening_lead", &bridge::BridgeEnv::hasOpeningLead)
      .def("feature_with_table_seat", &bridge::BridgeEnv::featureWithTableSeat)
      .def("feature_opening_lead", &bridge::BridgeEnv::featureOpeningLead)
      .def("print_hand_and_bidding", &bridge::BridgeEnv::printHandAndBidding)
      .def("info", &bridge::BridgeEnv::info)
      .def("get_idx", &bridge::BridgeEnv::getIdx)
      .def("reset_to", &bridge::BridgeEnv::resetTo)
      .def("subgame_end", &bridge::BridgeEnv::subgameEnd)
      .def("terminated", &bridge::BridgeEnv::terminated)
      .def("get_episode_reward", &bridge::BridgeEnv::getEpisodeReward)
      .def("get_rewards", &bridge::BridgeEnv::getRewards);

  py::class_<bridge::DuplicateBridgeEnv, rela::Env,
             std::shared_ptr<bridge::DuplicateBridgeEnv>>(m,
                                                          "DuplicateBridgeEnv")
      .def(py::init<const std::unordered_map<std::string, std::string>&>())
      .def(py::init<std::shared_ptr<DBInterface>,
                    const std::unordered_map<std::string, std::string>&>(),
           py::keep_alive<1, 2>())
      // .def("curr_json_str", &bridge::BridgeEnv::currJsonStr)
      .def("step", &bridge::DuplicateBridgeEnv::step)
      .def("feature", &bridge::DuplicateBridgeEnv::feature)
      // .def("has_opening_lead", &bridge::BridgeEnv::hasOpeningLead)
      // .def("feature_with_table_seat",
      // &bridge::BridgeEnv::featureWithTableSeat)
      // .def("feature_opening_lead", &bridge::BridgeEnv::featureOpeningLead)
      // .def("print_hand_and_bidding", &bridge::BridgeEnv::printHandAndBidding)
      // .def("info", &bridge::BridgeEnv::info)
      .def("get_idx", &bridge::DuplicateBridgeEnv::getIdx)
      // .def("reset_to", &bridge::BridgeEnv::resetTo)
      .def("reset", &bridge::DuplicateBridgeEnv::reset)
      .def("subgame_end", &bridge::DuplicateBridgeEnv::subgameEnd)
      .def("terminated", &bridge::DuplicateBridgeEnv::terminated);
  // .def("get_episode_reward", &bridge::BridgeEnv::getEpisodeReward)
  // .def("get_rewards", &bridge::DuplicateBridgeEnv::getRewards);

  py::class_<AllPassActor2, rela::Actor2, std::shared_ptr<AllPassActor2>>(
      m, "AllPassActor2")
      .def(py::init<>())
      .def("num_act", &AllPassActor2::numAct);

  py::class_<BaselineActor2, rela::Actor2, std::shared_ptr<BaselineActor2>>(
      m, "BaselineActor2")
      .def(py::init<std::shared_ptr<rela::Models>>(), py::keep_alive<1, 2>())
      .def("num_act", &BaselineActor2::numAct);

  py::class_<bridge::RandomActor, rela::Actor2,
             std::shared_ptr<bridge::RandomActor>>(m, "RandomActor")
      .def(py::init<>())
      .def("num_act", &bridge::RandomActor::numAct);

  py::class_<bridge::GreedyPlayActor, rela::Actor2,
             std::shared_ptr<bridge::GreedyPlayActor>>(m, "GreedyPlayActor")
      .def(py::init<>())
      .def(py::init<std::shared_ptr<rela::Models>>(), py::keep_alive<1, 2>())
      .def("num_act", &bridge::GreedyPlayActor::numAct);

  py::class_<BeliefTransferOptions, std::shared_ptr<BeliefTransferOptions>>(
      m, "BeliefTransferOptions")
      .def(py::init<>())
      .def_readwrite("debug", &BeliefTransferOptions::debug)
      .def_readwrite("opening_lead", &BeliefTransferOptions::openingLead);

  py::class_<BeliefTransferEnvActor, EnvActorBase,
             std::shared_ptr<BeliefTransferEnvActor>>(m,
                                                      "BeliefTransferEnvActor")
      .def(py::init<std::shared_ptr<bridge::BridgeEnv>,
                    std::vector<std::shared_ptr<rela::Actor2>>,
                    const EnvActorOptions&, const BeliefTransferOptions&,
                    std::shared_ptr<rela::PrioritizedReplay2>>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 6>());

  py::class_<CrossBiddingEnvActor, EnvActorBase,
             std::shared_ptr<CrossBiddingEnvActor>>(m, "CrossBiddingEnvActor")
      .def(py::init<std::shared_ptr<bridge::BridgeEnv>,
                    std::vector<std::shared_ptr<rela::Actor2>>,
                    const EnvActorOptions&,
                    std::shared_ptr<rela::PrioritizedReplay2>>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>());

  py::class_<bridge::ConsoleActor, rela::Actor2,
             std::shared_ptr<bridge::ConsoleActor>>(m, "ConsoleActor")
      .def(py::init<int>());
}
