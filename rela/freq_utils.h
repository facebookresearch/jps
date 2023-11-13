#pragma once

#include <unordered_map>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <mutex>

struct Freq {
  struct Item {
    int lastTick = 0;
    float count = 0;
    float freq = 0;
    float freqInfoSet = 0;
  };

  std::unordered_map<std::string, Item> tbl;

  float count = 0;
  float freq = 0;
  int lastTick = 0;

  void add(const std::string& completeCompactInfo, int tick) {
    auto& item = tbl[completeCompactInfo];
    item.count += 1.0;
    item.lastTick = tick;
    lastTick = tick;
  }

  void normalize(int total_count) {
    freq = count / (total_count + 1e-6);

    for (auto& v: tbl) {
      v.second.freqInfoSet = freq;
      v.second.freq = v.second.count / (total_count + 1e-6);
    }
  }

  void discount(float r) {
    count *= r;
    for (auto& v: tbl) {
      v.second.count *= r;
    }
  }

  Freq::Item get(const std::string& completeCompactInfo) const {
    Freq::Item freq;

    auto it = tbl.find(completeCompactInfo);
    if (it != tbl.end()) {
      freq = it->second;
    } 

    return freq;
  }

  std::string info() const {
    std::stringstream ss;
    ss << "[cnt: " << count << ", freq: " << freq << ", lastTick: " << lastTick << "]" << std::endl;

    using _Pair = std::pair<std::string, Item>;

    std::vector<_Pair> tmp(tbl.begin(), tbl.end());
    std::sort(tmp.begin(), tmp.end(), 
        [](const _Pair &p1, const _Pair &p2) { return p1.second.count > p2.second.count; }); 

    for (const auto &v : tmp) {
      const auto& item = v.second;
      if (item.freq < 1e-3) break;
      ss << "  " << v.first << ": " << item.freq 
         << " (" << item.count << ", lastTick: " << item.lastTick 
         << ")" << std::endl;
    }
    return ss.str();
  }
};

struct TrajItem {
  std::string infoSet;
  std::string completeCompactInfo;
  TrajItem(const std::string& infoSet, const std::string& completeCompactInfo)
      : infoSet(infoSet)
      , completeCompactInfo(completeCompactInfo) {
  }
};

// Frequency maps from infoSet to completeInfo (only working for tabular case).
struct Freqs {
  std::unordered_map<std::string, Freq> tbl;
  float count = 0;
  int tick = 0;

  void add(const std::vector<TrajItem> &traj) {
    for (const auto& item : traj) {
      tbl[item.infoSet].add(item.completeCompactInfo, tick);
    }
    count += 1.0;
    tick ++;
  }
  
  void normalize() {
    for (auto& f : tbl) {
      f.second.normalize(count);
    }
  }

  void discount(float r) {
    for (auto& f : tbl) {
      f.second.discount(r);
    }
    count *= r;
  }

  Freq::Item get(const TrajItem& item) const {
     Freq::Item freq;

     auto it = tbl.find(item.infoSet);
     if (it != tbl.end()) {
       return it->second.get(item.completeCompactInfo);
     }
     return freq;
  }

  std::string info() const {
    std::stringstream ss;

    using _Pair = std::pair<std::string, Freq>;
    std::vector<_Pair> tmp(tbl.begin(), tbl.end());
    std::sort(tmp.begin(), tmp.end(), 
        [](const _Pair &p1, const _Pair &p2) { return p1.first < p2.first; }); 

    // Print out reachability
    ss << "Tick: " << tick << ", Total cnt: " << count << std::endl;
    for (const auto &v : tmp) {
      ss << v.first << v.second.info();
    }
    return ss.str();
  }
};

inline Freqs globalIncFreqTable(const std::vector<TrajItem>& traj, int updateCount = 0) {
  static Freqs gFreqTable;
  static int saveCnt = 0;
  static std::mutex gMutex;

  std::lock_guard<std::mutex> lock(gMutex);
  gFreqTable.add(traj);
  gFreqTable.normalize();

  if (updateCount > 0 && gFreqTable.count >= updateCount) {
    // Print it out.
    std::ofstream oo("freq_table_" + std::to_string(saveCnt) + ".txt");
    oo << gFreqTable.info();
    gFreqTable.discount(0.5);
    saveCnt++;
  }

  return gFreqTable;
} 
