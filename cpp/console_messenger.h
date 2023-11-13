#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <list>
#include <memory>
#include <netinet/in.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace std {
class thread;
}

namespace bridge {

// Simple semaphore to communicate producer and consumer threads
class Semaphore {
 public:
  Semaphore(int count = 0)
      : count_(count) {
  }
  Semaphore(const Semaphore& other)
      : count_(other.count_) {
  }

  Semaphore& operator=(const Semaphore& rv) {
    count_ = rv.count_;
    return *this;
  }

  inline void release() {
    std::unique_lock<std::mutex> lock(mtx_);
    ++count_;
    cv_.notify_one();
  }

  inline void acquire() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (count_ == 0) {
      cv_.wait(lock);
    }
    --count_;
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int count_;
};

// Messenger to communicate between backend (envs and actors) and game console
// It uses singleton pattern to avoid passing the instance around:
//   1. Call ConsoleMessenger::init_messenger() to create the singleton;
//   2. Using ConsoleMessenger::get_messenger() to get the singleton
class ConsoleMessenger {
 protected:
  struct Message {
    int id;
    int seq_no;
    std::vector<std::string> tokens;

    Message();
    Message(const Message& rv);
    Message(Message&& rv);
    Message(const std::string& line);
    Message(int id, int seq_no, const std::string& content);
    Message& operator=(Message&& rv);
    inline bool is_empty() const {
      return tokens.size() == 0;
    }
  };

 public:
  static const std::string READY_MSG;
  static const std::string QUIT_MSG;

  static void init_messenger(
      const std::unordered_map<std::string, std::string>& params);
  static std::shared_ptr<ConsoleMessenger> get_messenger();

  virtual ~ConsoleMessenger() {
  }

  virtual void start() {
    throw std::runtime_error("EvalMessenger::start not implemented");
  }
  virtual void stop() {
    stopped_ = true;
  }
  virtual inline std::vector<std::string> read_actor_msg(int env_id,
                                                         bool blocking = true) {
    return move(read_env_actor_msg(env_id, "a", blocking));
  }
  virtual std::vector<std::string> read_env_msg(int env_id,
                                                bool blocking = true) {
    return move(read_env_actor_msg(env_id, "e", blocking));
  }
  virtual bool send_env_msg(int env_id, const std::string& msg);
  virtual bool send_env_info(int env_id, const std::string& info);
  friend std::ostream& operator<<(std::ostream& os,
                                  ConsoleMessenger::Message const& msg);

 protected:
  static std::shared_ptr<ConsoleMessenger> the_messenger_;
  static const std::string DEBUG_PREFIX;
  template <class... Args>
  void print_debug(Args... args) {
    if (verbose_) {
      (std::cout << ConsoleMessenger::DEBUG_PREFIX << " " << ... << args)
          << std::endl;
    }
  }

  std::unordered_map<int, int> env_eval_map_, eval_env_map_;
  std::unordered_map<int, Semaphore> env_conds_, env_msg_conds_;
  std::unordered_map<int, int> eval_seq_nos_, env_seq_nos_;
  std::deque<int> env_q_, eval_q_;
  std::mutex mtx_;
  std::list<Message> eval_msgs_, env_msgs_, buffered_env_msgs_;
  std::mutex eval_msg_mtx_, env_msg_mtx_, buffered_env_msg_mtx_;
  std::condition_variable eval_msg_cv_, env_msg_cv_;
  std::atomic_bool stopped_;
  bool verbose_;

  ConsoleMessenger(bool verbose = false)
      : stopped_(false)
      , verbose_(verbose) {
  }

  virtual void process_eval_msgs();
  virtual void process_env_msgs();

  virtual std::vector<Message> read_eval_msg() {
    throw std::runtime_error("EvalMessenger::read_eval_msg not implemented");
  }
  virtual bool send_eval_msg(const Message& msg) {
    throw std::runtime_error("EvalMessenger::send_eval_msg not implemented");
  }

  virtual void add_env(int id);
  virtual void delete_env(int);
  virtual void add_eval(int id, int seq_no);
  virtual void delete_eval(int id);
  virtual std::vector<std::string> read_env_actor_msg(int env_id,
                                                      const std::string& role,
                                                      bool blocking);
};

std::ostream& operator<<(std::ostream& os,
                         ConsoleMessenger::Message const& msg);

// A test messenger using standard IO, to test message format, backend
// functions
// etc
class CmdlineConsoleMessenger : public ConsoleMessenger {
 protected:
  std::thread *read_thread_, *send_thread_;

  std::vector<Message> read_eval_msg() override;
  bool send_eval_msg(const Message& msg) override;
  CmdlineConsoleMessenger(bool verbose = false);

 public:
  virtual ~CmdlineConsoleMessenger();

  void start() override;
  void stop() override;

  friend void ConsoleMessenger::init_messenger(
      const std::unordered_map<std::string, std::string>& params);
};

// A socket messenger using socket (normally set up via ssh port tunnelling)
class SocketConsoleMessenger : public ConsoleMessenger {
 protected:
  short port_;
  int server_socket_, msg_socket_;
  std::thread *read_thread_, *send_thread_;

  SocketConsoleMessenger(short port, bool verbose = false);

  std::vector<Message> read_eval_msg() override;
  bool send_eval_msg(const Message& msg) override;

  void process_eval_msgs() override;

 public:
  static const short DEFAULT_PORT;

  virtual ~SocketConsoleMessenger();

  void start() override;
  void stop() override;

  friend void ConsoleMessenger::init_messenger(
      const std::unordered_map<std::string, std::string>& params);
};
}