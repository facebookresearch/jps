// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// 
// 
#pragma once

namespace simple {

struct CommOptions {
  int N = 3;
  int numRound = 4;
  int possibleCards = -1;
  bool seqEnumerate = false;
};

}
