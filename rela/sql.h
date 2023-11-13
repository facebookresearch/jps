#pragma once

#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <sqlite3.h>

#if defined __has_include
#if __has_include(<spdlog/sinks/stdout_color_sinks.h>)
#include <spdlog/sinks/stdout_color_sinks.h>
#endif
#endif

#include "IndexedLoggerFactory.h"

namespace elf {

typedef int SqlCB(void*, int, char**, char**);

class SQL {
 public:
  SQL(const std::string& filename, std::string tableName);

  ~SQL() {
    sqlite3_close(db_);
  }

  bool insert(int idx, const std::string& content);

  bool readSection(int start, int num_record, std::vector<std::string>* data);

  bool getSize(int* sz);

  const std::string& LastError() const {
    return last_err_;
  }

 private:
  class SaveData {
   public:
    SaveData(std::vector<std::string>* data)
        : data_(data) {
    }

    void save(char* content) {
      data_->emplace_back(content);
    }

   private:
    std::vector<std::string>* data_;
  };

  static logging::IndexedLoggerFactory* getLoggerFactory();

  static int read_callback(void* handle,
                           int num_columns,
                           char** column_texts,
                           char** column_names);

  int exec(const std::string& sql,
           SqlCB callback = nullptr,
           void* callback_handle = nullptr);

  bool table_exists();

  bool table_create();

  std::string tableName_;
  std::shared_ptr<spdlog::logger> logger_;
  sqlite3* db_;
  std::string last_err_;
};

}  // namespace elf
