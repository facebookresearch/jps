#pragma once

#include <boost/chrono/include.hpp>

#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>

namespace rela {

namespace clock {

using Duration = boost::chrono::duration<double>;
using TimePoint = boost::chrono::time_point<boost::chrono::thread_clock>;

struct Key {
  std::vector<std::string> prefix;
  friend bool operator<(const Key& k1, const Key& k2) {
    const auto &p1 = k1.prefix;
    const auto &p2 = k2.prefix;

    for (size_t i = 0; i < std::min(p1.size(), p2.size()); ++i) {
      if (p1[i] < p2[i]) return true;
      if (p1[i] > p2[i]) return false;
    }
    return p1.size() < p2.size();
  }

  friend bool operator==(const Key& k1, const Key& k2) {
    const auto &p1 = k1.prefix;
    const auto &p2 = k2.prefix;
    if (p1.size() != p2.size()) return false;

    for (size_t i = 0; i < p1.size(); ++i) {
      if (p1[i] != p2[i]) return false;
    }
    return true;
  }
};

struct Record {
  Duration sumDuration;
  int n;

  Record() {
   reset();
  }

  void add(Duration d) {
    sumDuration += d;
    n ++;
  }

  void reset() {
    sumDuration = Duration(0);
    n = 0;
  }

  friend bool operator<(const Record& r1, const Record& r2) {
    return r1.n < r2.n;
  }

  Record &operator+=(const Record &r) {
    sumDuration += r.sumDuration;
    n += r.n;
    return *this;
  }

  std::string info(double total_ms, const Key &key) const {
    std::stringstream ss;

    // Indent from key
    assert(! key.prefix.empty());

    for (size_t i = 0; i < key.prefix.size() - 1; ++i) {
      ss << "  ";
    }
    ss << key.prefix.back();

    return info(total_ms, ss.str());
  }

  std::string info(double total_ms, const std::string& key) const {
    std::stringstream ss;

    double ms = sumDuration.count() * 1000;
    ss << std::left << "  " << std::setw(60) << key << ": " << std::setw(10)
       << std::setprecision(4) << ms / 1000 << " s (" << std::setw(10)
       << std::setprecision(4) << ms / total_ms * 100
       << " %), Per call: " << std::setw(10) << std::setprecision(4)
       << ms / n << " ms [" << n << "]";

    return ss.str();
  }
};

}

}

namespace std {
  template <> struct hash<rela::clock::Key> {
    std::size_t operator()(const rela::clock::Key& k) const {
      std::size_t v = 1;

      for (const auto& kk : k.prefix) {
        v <<= 1;
        v ^= std::hash<std::string>()(kk);
      }

      return v;
    }
  };
}

namespace rela {

namespace clock {


inline TimePoint now() {
  return boost::chrono::thread_clock::now();
}

class ThreadClock {
 public:
  class Instance {
   public:
    Instance(const std::string& item, ThreadClock& clock)
        : item_(item)
        , clock_(clock) {
      timeStart_ = now();
    }

    Duration duration() const {
      return now() - timeStart_;
    }
    const std::string item() const {
      return item_;
    }

    ~Instance() {
      clock_.stop(*this);
    }

   private:
    std::string item_;
    TimePoint timeStart_;

    ThreadClock& clock_;
  };

  ThreadClock() {
    reset();
  }

  void setName(const std::string& name) {
    name_ = name;
  }

  void reset() {
    for (auto& k2v : records_) {
      k2v.second.reset();
    }

    curStack_.prefix.clear();
    timeStart_ = now();
  }

  std::string summary() const {
    std::stringstream ss;

    TimePoint t = now();

    Duration total = now() - timeStart_;
    double total_ms = total.count() * 1000;

    /*
    double total_ms = 0;
    for (const auto& k2v : records_) {
      total_ms += k2v.second.first.count() * 1000;
    }
    */

    ss << std::left << "Clock [" << name_ << "][" << std::this_thread::get_id()
       << "], "
       << "Total: " << total_ms << " ms"
       << ", Now: " << t << ", start: " << timeStart_ << std::endl;
    // ss << "current: " << t << ", start: " << timeStart_ << std::endl;
    // " ms " << " (" << total_ms / total_run_ms * 100 << "% of total : " <<
    // total_run_ms << ")" << std::endl;

    std::vector<std::pair<Key, Record>> ordered;
    for (const auto& k2v : records_) {
      ordered.push_back(k2v);
    }

    std::sort(ordered.begin(), ordered.end());

    for (const auto& k2v : ordered) {
      const auto& k = k2v.first;
      const auto& r = k2v.second;

      if (r.n > 0) {
        ss << r.info(total_ms, k) << std::endl;
      }
    }

    ss << "Clock split by Name only: " << std::endl;

    // Print overall stats for each function.
    std::unordered_map<std::string, Record> groupBy;
    for (const auto& k2v : records_) {
      groupBy[k2v.first.prefix.back()] += k2v.second;
    }

    std::vector<std::pair<std::string, Record>> ordered2;
    for (const auto& k2v : groupBy) {
      ordered2.push_back(k2v);
    }

    std::sort(ordered2.begin(), ordered2.end());

    for (const auto& k2v : ordered2) {
      const auto& k = k2v.first;
      const auto& r = k2v.second;

      if (r.n > 0) {
        ss << r.info(total_ms, k) << std::endl;
      }
    }

    return ss.str();
  }

  inline Instance start(const std::string& item) {
    curStack_.prefix.push_back(item);
    return Instance(item, *this);
  }

 private:

  std::string name_;
  TimePoint timeStart_;
  Key curStack_;
  std::unordered_map<Key, Record> records_;


  inline void stop(Instance& instance) {
    // cout << "Record: " << instance.item() << endl;
    //
    assert(curStack_.prefix.size() >= 1 && curStack_.prefix.back() == instance.item());

    auto it = records_.find(curStack_);
    if (it == records_.end()) {
      it = records_
               .insert(
                   std::make_pair(curStack_, Record()))
               .first;
    }

    it->second.add(instance.duration());

    // Backtrace. 
    curStack_.prefix.pop_back();
  }
};

// Some hacky macro
/*
#define START_TIMING(clock, name)                                              \
  {                                                                            \
    auto __instance =                                                          \
        clock.start(#name "[" __FILE__ ":" + std::to_string(__LINE__) + ", " +
__func__ + "]");

        */

class ClockManager {
 public:
  inline void registerName(const std::string& name) {
    get().setName(name);
  }

  inline ThreadClock& getClock() {
    return get();
  }

  /* This is not safe
  std::string summary() const {
         std::stringstream ss;
     std::lock_guard<std::mutex> lock(m_);
     for (const auto& k2v : id2clock_) {
       ss << k2v.second->summary() << std::endl;

       k2v.second->reset();
     }

     return ss.str();
  }
  */

 private:
  std::unordered_map<std::thread::id, std::unique_ptr<ThreadClock>> id2clock_;

  // Use std::mutex instead (rather than rwlock) due to performance issue:
  // https://stackoverflow.com/questions/14306797/c11-equivalent-to-boost-shared-mutex/45580208#45580208
  mutable std::mutex m_;

  ThreadClock& get() {
    std::lock_guard<std::mutex> lock(m_);
    auto id = std::this_thread::get_id();
    auto it = id2clock_.find(id);

    if (it == id2clock_.end()) {
      it = id2clock_.insert(make_pair(id, std::make_unique<ThreadClock>()))
               .first;
    }
    return *it->second;
  }
};

// Singleton.
ClockManager gClocks;

// #define ENABLE_PROFILING

#ifdef ENABLE_PROFILING

#define START_TIMING2(clock, name)                                             \
  {                                                                            \
    auto __instance = clock.start(std::string() + #name + ":" + __func__);

#define START_TIMING(name)                                                     \
  {                                                                            \
    auto __instance = rela::clock::gClocks.getClock().start(                   \
        std::string() + #name + ":" + __func__);

#define END_TIMING }

#define PRINT_TIMING_SUMMARY                                                   \
  {                                                                            \
    auto __clock = rela::clock::gClocks.getClock();                            \
    std::cout << __clock.summary() << std::endl;                               \
    __clock.reset();                                                           \
  }

#define PRINT_TIMING_SUMMARY2(clock)                                           \
  {                                                                            \
    std::cout << clock.summary() << std::endl;                                 \
    clock.reset();                                                             \
  }

#else

#define START_TIMING2(clock, name)                                          

#define START_TIMING(name)                                                 

#define END_TIMING 

#define PRINT_TIMING_SUMMARY                                              

#define PRINT_TIMING_SUMMARY2(clock)                                     

#endif

}  // namespace clock

}  // namespace rela
