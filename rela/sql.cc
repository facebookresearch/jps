#include "sql.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace elf {

SQL::SQL(const std::string& filename, std::string tableName)
    : tableName_(std::move(tableName))
    , logger_(getLoggerFactory()->makeLogger(
            "elf::sql-",
            "-" + filename + "_" + tableName))
    , db_(nullptr) {
  int rc = sqlite3_open(filename.c_str(), &db_);
  if (rc) {
    logger_->error(
        "Can't open database {filename: {}, errmsg: {}}",
        filename,
        sqlite3_errmsg(db_));
    throw std::runtime_error("Cannot open database");
  }
  if (!table_exists()) {
    table_create();
  }
}

logging::IndexedLoggerFactory* SQL::getLoggerFactory() {
  static logging::IndexedLoggerFactory factory(
      [=](const std::string& name) { return spdlog::stdout_color_mt(name); });
  return &factory;
}

int SQL::read_callback(
    void* handle,
    int num_columns,
    char** column_texts,
    char** column_names) {
  (void)num_columns;
  (void)column_names;
  SaveData* h = reinterpret_cast<SaveData*>(handle);
  h->save(column_texts[1]);
  return 0;
}

int SQL::exec(const std::string& sql, SqlCB callback, void* callback_handle) {
  char* zErrMsg;
  logger_->debug("SQL: {}", sql);
  int rc = sqlite3_exec(db_, sql.c_str(), callback, callback_handle, &zErrMsg);
  if (rc != 0) {
    last_err_ = zErrMsg;
    // logger_->error(sqlite3_errmsg(db_));
    // throw std::runtime_error("Error operating db");
  } else {
    last_err_ = "";
  }
  sqlite3_free(zErrMsg);
  return rc;
}

bool SQL::getSize(int *cnt) {
  const std::string sql = "SELECT COUNT(*) from " + tableName_ + ";";

  auto f = [](void* handle, int num_columns, char** column_texts, char** column_names) {
    int *count = reinterpret_cast<int *>(handle);
    *count = atoi(column_texts[0]);
    return 0;  
  };

  int ret = exec(sql, f, cnt);
  return exec(sql) == 0;
}

bool SQL::table_create() {
  const std::string sql = "CREATE TABLE " + tableName_ +
      " ("
      "IDX            INTEGER PRIMARY KEY,"
      "CONTENT        TEXT);";

  return exec(sql) == 0;
}

bool SQL::table_exists() {
  const std::string sql = "SELECT 1 FROM " + tableName_ + " LIMIT 1;";
  return exec(sql) == 0;
}

bool SQL::insert(int idx, const std::string& content) {
  std::string sql = "INSERT INTO " + tableName_ + " VALUES (" +
      std::to_string(idx) + ", \'" + content + "\');";
  return exec(sql) == 0;
}

bool SQL::readSection(int start, int num_record, std::vector<std::string>* data) {
  // Read things into a buffer.
  std::string sql = "SELECT * FROM " + tableName_ + " WHERE IDX BETWEEN " 
    + std::to_string(start) + " AND " + std::to_string(start + num_record - 1);
  sql += ";";

  data->clear();
  SaveData sav(data);
  int ret = exec(sql, read_callback, &sav);
  return ret == 0;
}

}

