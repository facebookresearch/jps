// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "simple_game/comm.h"
#include "simple_game/comm2.h"
#include "simple_game/simple_bidding.h"
#include "simple_game/two_suited_bridge.h"

namespace py = pybind11;
// using namespace bridge;

PYBIND11_MODULE(simple_game, m) {
  py::class_<simple::CommOptions, std::shared_ptr<simple::CommOptions>>(m, "CommOptions")
      .def(py::init<>())
      .def_readwrite("num_round", &simple::CommOptions::numRound)
      .def_readwrite("seq_enumerate", &simple::CommOptions::seqEnumerate)
      .def_readwrite("possible_cards", &simple::CommOptions::possibleCards)
      .def_readwrite("N", &simple::CommOptions::N);

  py::class_<simple::Communicate, rela::Env, std::shared_ptr<simple::Communicate>>(
      m, "Communicate")
      .def(py::init<const simple::CommOptions&>());

  py::class_<simple::Communicate2, rela::Env, std::shared_ptr<simple::Communicate2>>(
      m, "Communicate2")
      .def(py::init<const simple::CommOptions&>());

  py::class_<simple::SimpleBidding, rela::Env, std::shared_ptr<simple::SimpleBidding>>(
      m, "SimpleBidding")
      .def(py::init<const simple::CommOptions&>());

  py::class_<simple::TwoSuitedBridge, rela::Env, std::shared_ptr<simple::TwoSuitedBridge>>(
      m, "TwoSuitedBridge")
      .def(py::init<const simple::CommOptions&>());
}
