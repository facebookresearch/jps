#pragma once

struct SearchActorOptions {
  float searchRatio = 0.1;
  int verboseFreq = 500000;
  int updateCount = 50000; 
  bool useHacky = false;
  bool useTabularRef = false;
  bool useGradUpdate = true;
  float baselineRatio = 0.1;

  // Deprecated
  bool bestOnBest = false;
};

