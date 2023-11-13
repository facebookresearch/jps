
#include "console_messenger.h"
#include <unistd.h>
#include <sstream>
#include <thread>
#include "util_func.h"

using namespace std;

namespace bridge {

ConsoleMessenger::Message::Message() : id(-1), seq_no(-1) {}

ConsoleMessenger::Message::Message(const Message& rv)
    : id(rv.id), seq_no(rv.seq_no), tokens(rv.tokens) {}

ConsoleMessenger::Message::Message(Message&& rv)
    : id(rv.id), seq_no(rv.seq_no), tokens(move(rv.tokens)) {
  rv.id = -1;
  rv.seq_no = -1;
}

ConsoleMessenger::Message::Message(const std::string& line) {
  stringstream ss(line);
  string s;
  int i = 0;
  while (getline(ss, s, ' ')) {
    if (s.empty()) {
      continue;
    }
    if (i == 0) {
      id = stoi(s);
    } else if (i == 1) {
      seq_no = stoi(s);
    } else {
      tokens.push_back(move(s));
    }
    ++i;
  }
}

ConsoleMessenger::Message::Message(int id, int seq_no,
                                   const std::string& content) {
  this->id = id;
  this->seq_no = seq_no;
  stringstream ss(content);
  string s;
  while (getline(ss, s, ' ')) {
    if (s.empty()) {
      continue;
    }
    tokens.push_back(move(s));
  }
}

ConsoleMessenger::Message& ConsoleMessenger::Message::operator=(Message&& rv) {
  id = rv.id;
  rv.id = -1;
  seq_no = rv.seq_no;
  rv.seq_no = -1;
  tokens = move(rv.tokens);
  return *this;
}

ostream& operator<<(ostream& os, ConsoleMessenger::Message const& msg) {
  os << msg.id << " " << msg.seq_no;
  for (const auto& t : msg.tokens) os << " " << t;
  return os;
}

const string ConsoleMessenger::READY_MSG = "ready";
const string ConsoleMessenger::QUIT_MSG = "quit";
const string ConsoleMessenger::DEBUG_PREFIX = "+-+-+-+-+-+-+-+-+-+";
std::shared_ptr<ConsoleMessenger> ConsoleMessenger::the_messenger_ = nullptr;

void ConsoleMessenger::init_messenger(
    const std::unordered_map<std::string, std::string>& params) {
  bool verbose = false;
  extractParams(params, "verbose", &verbose, false);
  string type;
  if (extractParams(params, "type", &type, false)) {
    int port = SocketConsoleMessenger::DEFAULT_PORT;
    extractParams(params, "port", &port, false);
    ConsoleMessenger::the_messenger_ = std::shared_ptr<SocketConsoleMessenger>(
        new SocketConsoleMessenger((short)port, verbose));
  } else {
    ConsoleMessenger::the_messenger_ = std::shared_ptr<CmdlineConsoleMessenger>(
        new CmdlineConsoleMessenger(verbose));
  }
}

std::shared_ptr<ConsoleMessenger> ConsoleMessenger::get_messenger() {
  return ConsoleMessenger::the_messenger_;
}

void ConsoleMessenger::add_env(int id) {
  print_debug("add_env(", id, ")");
  lock_guard<mutex> lck(mtx_);
  auto it = env_eval_map_.find(id);
  if (it != env_eval_map_.end()) {
    // env id already added
    return;
  } else {
    for (auto it = env_q_.begin(); it != env_q_.end(); ++it) {
      if (*it == id) {
        // env id already added
        return;
      }
    }
  }
  env_msg_conds_[id] = Semaphore();
  env_seq_nos_[id] = 0;
  if (!eval_q_.empty()) {
    int eval_id = eval_q_.front();
    env_eval_map_[id] = eval_id;
    eval_env_map_[eval_id] = id;
    eval_q_.pop_front();
    print_debug("  env(", id, ")<==>eval(", eval_id, ")");
  } else {
    env_q_.push_back(id);
    print_debug("  env(", id, ") waiting for eval");
  }
}

void ConsoleMessenger::delete_env(int id) {
  print_debug("delete_env(", id, ")");
  lock_guard<mutex> lck(mtx_);
  auto it1 = env_eval_map_.find(id);
  if (it1 == env_eval_map_.end()) {
    for (auto it2 = env_q_.begin(); it2 != env_q_.end(); ++it2) {
      if (*it2 == id) {
        env_q_.erase(it2);
        break;
      }
    }
  } else {
    auto it2 = eval_env_map_.find(it1->second);
    if (it2 != eval_env_map_.end()) {
      eval_env_map_.erase(it2);
    }
    env_eval_map_.erase(it1);
    add_eval(it1->second, -1);
  }
  auto it2 = env_seq_nos_.find(id);
  if (it2 != env_seq_nos_.end()) {
    env_seq_nos_.erase(it2);
  }
}

void ConsoleMessenger::add_eval(int id, int seq_no) {
  print_debug("add_eval(", id, ")");
  lock_guard<mutex> lck(mtx_);
  auto it = eval_env_map_.find(id);
  if (it == eval_env_map_.end()) {
    if (!env_q_.empty()) {
      int env_id = env_q_.front();
      env_eval_map_[env_id] = id;
      eval_env_map_[id] = env_id;
      env_q_.pop_front();
      print_debug("  env(", env_id, ")<==>eval(", id, ")");
      lock_guard<mutex> lck(env_msg_mtx_);
      for (auto it = buffered_env_msgs_.begin(); it != buffered_env_msgs_.end();
           ++it) {
        if (it->id == env_id) {
          it->id = id;
          env_msgs_.push_back(move(*it));
          buffered_env_msgs_.erase(it++);
        }
      }
      env_msg_cv_.notify_all();
    } else {
      bool added = false;
      for (auto it = eval_q_.begin(); !added && it != eval_q_.end(); ++it) {
        added = *it == id;
      }
      if (!added) {
        eval_q_.push_back(id);
      }
    }
  }
  if (seq_no >= 0) {
    eval_seq_nos_[id] = seq_no;
  }
}

void ConsoleMessenger::delete_eval(int id) {
  print_debug("delete_eval(", id, ")");
  lock_guard<mutex> lck(mtx_);
  auto it1 = eval_env_map_.find(id);
  if (it1 == eval_env_map_.end()) {
    for (auto it2 = eval_q_.begin(); it2 != eval_q_.end(); ++it2) {
      if (*it2 == id) {
        eval_q_.erase(it2);
        break;
      }
    }
  } else {
    auto it2 = env_eval_map_.find(it1->second);
    if (it2 != env_eval_map_.end()) {
      env_eval_map_.erase(it2);
    }
    // TODO: reset env
    env_q_.push_back(it1->second);
    eval_env_map_.erase(it1);
  }
  auto it3 = eval_seq_nos_.find(id);
  if (it3 != eval_seq_nos_.end()) {
    eval_seq_nos_.erase(it3);
  }
}

void ConsoleMessenger::process_eval_msgs() {
  print_debug("process_eval_msgs() started");
  while (!stopped_) {
    vector<Message> msgs(move(read_eval_msg()));
    if (msgs.empty()) {
      break;
    }
    for (Message& msg : msgs) {
      if (msg.tokens[0] == READY_MSG) {
        add_eval(msg.id, msg.seq_no);
      } else if (msg.tokens[0] == QUIT_MSG) {
        delete_eval(msg.id);
      } else {
        lock_guard<mutex> lck(mtx_);
        auto it = eval_seq_nos_.find(msg.id);
        if (it != eval_seq_nos_.end() && it->second < msg.seq_no) {
          eval_seq_nos_[msg.id] = msg.seq_no;
          auto it = eval_env_map_.find(msg.id);
          if (it != eval_env_map_.end()) {
            int env_id = it->second;
            {
              msg.id = env_id;
              lock_guard<mutex> lck(eval_msg_mtx_);
              eval_msgs_.push_back(move(msg));
            }
            env_msg_conds_[env_id].release();
          }
        }
      }
    }
  }
  print_debug("process_eval_msgs() ended");
}

void ConsoleMessenger::process_env_msgs() {
  print_debug("process_env_msgs() started");
  while (!stopped_) {
    unique_lock<mutex> lck(env_msg_mtx_);
    while (env_msgs_.empty()) {
      env_msg_cv_.wait(lck);
    }
    Message msg = move(env_msgs_.front());
    env_msgs_.pop_front();
    if (msg.id < 0 || msg.tokens.size() == 0) {
      continue;
    }
    send_eval_msg(msg);
  }
  print_debug("process_env_msgs() ended");
}

vector<string> ConsoleMessenger::read_env_actor_msg(int env_id,
                                                    const std::string& role,
                                                    bool blocking) {
  print_debug("read_env_actor_msg(", env_id, ", ", role, ", ", blocking, ")");
  if (blocking) {
    unique_lock<mutex> lck(mtx_);
    auto it = env_msg_conds_.find(env_id);
    if (it != env_msg_conds_.end()) {
      lck.unlock();
      it->second.acquire();
      lck.lock();
    }
  }
  lock_guard<mutex> lck(eval_msg_mtx_);
  for (auto it = eval_msgs_.begin(); it != eval_msgs_.end(); ++it) {
    if (it->id == env_id && it->tokens[0] == role) {
      vector<string> msg(move(it->tokens));
      eval_msgs_.erase(it);
      return move(msg);
    }
  }
  return vector<string>();
}

bool ConsoleMessenger::send_env_msg(int env_id, const std::string& msg) {
  print_debug("send_env_msg(", env_id, ", ", msg, ")");
  if (msg == READY_MSG) {
    add_env(env_id);
    return true;
  } else if (msg == QUIT_MSG) {
    delete_env(env_id);
    return true;
  } else {
    int eval_id = -1, seq_no = -1;
    {
      lock_guard<mutex> lck(mtx_);
      auto it1 = env_seq_nos_.find(env_id);
      if (it1 == env_seq_nos_.end()) {
        return false;
      }
      seq_no = it1->second;
      env_seq_nos_[env_id] = seq_no + 1;
      auto it = env_eval_map_.find(env_id);
      if (it != env_eval_map_.end()) {
        eval_id = it->second;
      } else {
        buffered_env_msgs_.push_back(Message(env_id, seq_no, msg));
      }
    }
    if (eval_id >= 0) {
      lock_guard<mutex> lck(env_msg_mtx_);
      env_msgs_.push_back(Message(eval_id, seq_no, msg));
      env_msg_cv_.notify_all();
      return true;
    } else {
      return false;
    }
  }
}

bool ConsoleMessenger::send_env_info(int env_id, const std::string& info) {
  print_debug("send_env_info(", env_id, ", ", info, ")");
  int eval_id = -1;
  {
    lock_guard<mutex> lck(mtx_);
    auto it = env_eval_map_.find(env_id);
    if (it != env_eval_map_.end()) {
      eval_id = it->second;
    } else {
      buffered_env_msgs_.push_back(Message(env_id, -1, info));
    }
  }
  if (eval_id >= 0) {
    lock_guard<mutex> lck(env_msg_mtx_);
    env_msgs_.push_back(Message(eval_id, -1, info));
    env_msg_cv_.notify_all();
    return true;
  } else {
    return false;
  }
}

CmdlineConsoleMessenger::CmdlineConsoleMessenger(bool verbose)
    : ConsoleMessenger(verbose), read_thread_(nullptr), send_thread_(nullptr) {
  print_debug("CmdlineConsoleMessenger() created");
}

CmdlineConsoleMessenger::~CmdlineConsoleMessenger() {
  print_debug("CmdlineConsoleMessenger() deleted");
}

vector<ConsoleMessenger::Message> CmdlineConsoleMessenger::read_eval_msg() {
  string msg;
  cout << ">> ";
  getline(cin, msg);
  vector<ConsoleMessenger::Message> msgs;
  msgs.push_back(move(Message(msg)));
  return move(msgs);
}

bool CmdlineConsoleMessenger::send_eval_msg(const Message& msg) {
  cout << msg << endl;
  return true;
}

void CmdlineConsoleMessenger::start() {
  read_thread_ = new thread(&CmdlineConsoleMessenger::process_eval_msgs, this);
  send_thread_ = new thread(&CmdlineConsoleMessenger::process_env_msgs, this);
  print_debug("CmdlineConsoleMessenger started");
}

void CmdlineConsoleMessenger::stop() {
  ConsoleMessenger::stop();
  print_debug("CmdlineConsoleMessenger stopped");
}

const short SocketConsoleMessenger::DEFAULT_PORT = 2001;

SocketConsoleMessenger::SocketConsoleMessenger(short port, bool verbose)
    : ConsoleMessenger(verbose),
      port_(port),
      server_socket_(-1),
      msg_socket_(-1),
      read_thread_(nullptr),
      send_thread_(nullptr) {
  print_debug("SocketMessenger(port=", port, ") created");
}

SocketConsoleMessenger::~SocketConsoleMessenger() {
  stop();
  if (read_thread_) {
    delete read_thread_;
  }
  if (send_thread_) {
    delete send_thread_;
  }
  print_debug("SocketMessenger() deleted");
}

void SocketConsoleMessenger::start() {
  read_thread_ = new thread(&SocketConsoleMessenger::process_eval_msgs, this);
  print_debug("SocketMessenger started");
}

void SocketConsoleMessenger::stop() {
  ConsoleMessenger::stop();
  if (server_socket_ > 0) {
    shutdown(server_socket_, SHUT_RDWR);
  }
  if (msg_socket_ >= 0) {
    shutdown(msg_socket_, SHUT_RDWR);
  }
  if (read_thread_) {
    if (read_thread_->joinable()) read_thread_->join();
    delete read_thread_;
    read_thread_ = nullptr;
  }
  if (send_thread_) {
    if (send_thread_->joinable()) {
      send_thread_->join();
    }
    delete send_thread_;
    send_thread_ = nullptr;
  }
  server_socket_ = -1;
  msg_socket_ = -1;
  print_debug("SocketMessenger stopped");
}

vector<ConsoleMessenger::Message> SocketConsoleMessenger::read_eval_msg() {
  char buffer[1024] = {0};
  ssize_t len = recv(msg_socket_, buffer, 1024, 0);
  vector<ConsoleMessenger::Message> msgs;
  if (len <= 0) {
    return msgs;
  }
  print_debug("socket recv'd \"", buffer, "\"");
  char* token = std::strtok(buffer, "\n");
  while (token != NULL) {
    msgs.push_back(move(Message(token)));
    token = std::strtok(NULL, "\n");
  }
  return move(msgs);
}

bool SocketConsoleMessenger::send_eval_msg(
    const ConsoleMessenger::Message& msg) {
  if (msg.seq_no < 0) {
    return false;
  }
  ostringstream os;
  os << msg << "\n";
  string out = os.str();
  if (send(msg_socket_, out.c_str(), out.length(), 0) < 0) {
    return false;
  }
  print_debug("socket send \"", msg, "\"");
  return true;
}

void SocketConsoleMessenger::process_eval_msgs() {
  server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket_ == 0) {
    // error handling
    print_debug("creating server_socket error");
    return;
  }
  print_debug("server_socket created");

  int opt = 1;
  if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    // error handling
    print_debug("setting server_socket opt error");
    close(server_socket_);
    return;
  }

  struct sockaddr_in address;
  int addrlen = sizeof(address);
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port_);
  if (bind(server_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
    // error handling
    print_debug("binding server_socket error");
    close(server_socket_);
    return;
  }
  print_debug("server_socket bond to localhost:", port_);

  print_debug("server_socket listening...");
  if (listen(server_socket_, 3) < 0) {
    // error handling
    print_debug("server_socket listen error");
    close(server_socket_);
    return;
  }

  msg_socket_ =
      accept(server_socket_, (struct sockaddr*)&address, (socklen_t*)&addrlen);
  if (msg_socket_ < 0) {
    // error handling
    print_debug("server_socket accepting connection error");
    close(server_socket_);
    return;
  }
  print_debug("server_socket accepted: ", msg_socket_);

  send_thread_ = new thread(&SocketConsoleMessenger::process_env_msgs, this);

  ConsoleMessenger::process_eval_msgs();

  close(msg_socket_);
  close(server_socket_);
}
}
