#pragma once

#include "rela/sql.h"
#include <string>

class DBInterface {
 public:
   using Data = std::vector<std::string>;
   struct Handle {
     int offset = 0;
     int size = 0;
     Data data; 
   };

   DBInterface() = default;

   DBInterface(const std::string& filenameLoad, const std::string& filenameSave, int numThreads) {
     loader_ = std::make_unique<elf::SQL>(filenameLoad, "records");
     // Get dataset size.
     assert (loader_->getSize(&loaderSize_));

     if (filenameSave != "") {
       saver_ = std::make_unique<elf::SQL>(filenameSave, "records");
     }

     assert(loaderSize_ % numThreads == 0);

     Data allData;
     // Read them all.
     assert(loader_->readSection(0, loaderSize_, &allData));

     int secLen = loaderSize_ / numThreads;
     assert(secLen > 0);

     for (int i = 0; i < numThreads; ++i) {
       auto h = std::make_shared<Handle>();

       h->offset = secLen * i;
       h->size = secLen;
       auto start = allData.begin() + h->offset;
       h->data = Data(start, start + secLen);

       bufferedData_.push_back(h); 
     }
   } 

   int getDatasetSize() const { return loaderSize_; }
   int getNumThreads() const { return (int)bufferedData_.size(); }

   virtual std::shared_ptr<Handle> getData(int threadIdx) {
     assert(threadIdx >= 0 && threadIdx < (int)bufferedData_.size());
     return bufferedData_[threadIdx]; 
   }

   bool canSave() const { return saver_ != nullptr; }

   bool saveData(int idx, const std::string& data) {
     if (saver_ == nullptr) return false;
     return saver_->insert(idx, data);
   }

 private:
   std::unique_ptr<elf::SQL> loader_;
   std::unique_ptr<elf::SQL> saver_;
   int loaderSize_;

   // Buffer reuse for the same thread.
   // of size #threads
   std::vector<std::shared_ptr<Handle>> bufferedData_;
};

template <typename T>
T _convert(const std::string &v);


template <>
inline std::string _convert<std::string>(const std::string &v) { return v; }

template <>
inline int _convert<int>(const std::string &v) { return std::stoi(v); }

template <>
inline float _convert<float>(const std::string &v) { return std::stof(v); }

template <>
inline bool _convert<bool>(const std::string &v) { return v == "1" || v == "true" || v == "True"; }

template <typename T>
inline bool extractParams(const std::unordered_map<std::string, std::string>& gameParams, const std::string& key, T *entry, bool mandatory) {
    auto it = gameParams.find(key); 
    if (it != gameParams.end()) {
      *entry = _convert<T>(it->second);
      return true;
    } else {
      if (mandatory) {
        throw std::runtime_error(key + " missing");
      } else {
        return false;
      }
    }
} 

