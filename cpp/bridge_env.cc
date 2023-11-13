#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

#include "bridge_common.h"
#include "bridge_env.h"
#include "rela/logging.h"
#include "util.h"

namespace bridge {

void BridgeEnv::init() {
  // Compute offset.
  handle_ = database_->getData(threadIdx_);
  RELA_CHECK_NOTNULL(handle_);
  currentIdx_ = -1;

  if (consoleMessenger_) {
    consoleMessenger_->send_env_msg(threadIdx_, ConsoleMessenger::READY_MSG);
  }
}

bool BridgeEnv::reset() {
  int idx;

  if (option_.sampler == "uniform") {
    std::uniform_int_distribution<int> distrib(0, handle_->size - 1);
    idx = distrib(rng_);
  } else if (option_.sampler == "seq") {
    idx = currentIdx_ + 1;
    if (idx == handle_->size) {
      return false;
    }
  } else {
    throw std::runtime_error("Unknown sampler: " + option_.sampler);
  }
  resetTo(idx, option_.fixedDealer, option_.fixedVul);

  return true;
}

void BridgeEnv::resetTo(int recordIdx, int dealer, int vul) {
  RELA_CHECK_GT(handle_->size, 0);

  currentIdx_ = recordIdx;

  const auto j = json::parse(handle_->data[currentIdx_]);
  if (j.find("dealer") != j.end()) {
    dealer = j["dealer"];
  }
  if (j.find("vul") != j.end()) {
    vul = j["vul"];
  }

  if (dealer == -1) {
    std::uniform_int_distribution<int> distrib(0, kPlayer - 1);
    dealer = distrib(rng_);
  }
  if (vul == -1) {
    std::uniform_int_distribution<int> distrib(0, NUM_VULNERABILITY - 1);
    vul = distrib(rng_);
  }

  const std::string& pbn = j["pbn"];
  state_.reset(pbn, j["ddt"], dealer, (Vulnerability)vul);

  // std::cout << deal << std::endl;

  if (consoleMessenger_) {
    std::ostringstream ss;
    ss << "new " << pbn.substr(1, pbn.length() - 2) << " d " << dealer << " v "
       << vulMap[vul] << " t " << option_.tables;
    ConsoleMessenger::get_messenger()->send_env_msg(threadIdx_, ss.str());
    const std::vector<std::string>& msg =
        ConsoleMessenger::get_messenger()->read_env_msg(threadIdx_);
    if (msg.size() < 2 || msg[1] != "started") {
      std::cout << "not received \"started\" msg from console!";
    }
  }

  if (option_.verbose) {
    std::cout << std::endl << "dealer is " << dealer << std::endl;
    std::cout << "vul is " << vulMap[vul] << std::endl;
    std::cout << "[Vulnerability " << vulMap[vul] << "]" << std::endl;
    std::cout << j["pbn"] << std::endl;
    std::cout << j["ddt"] << std::endl;
  }

  terminated_ = false;
  tableOver_ = false;
  curr_ = j;

  // Clear the feature history as well.
  featureHistory_.clear();
}

void BridgeEnv::biddingStep(int actionIdx) {
  if (consoleMessenger_) {
    Bid bid(actionIdx);
    std::ostringstream ss;
    ss << "bid " << state_.playerIdx() << " " << bid.toString();
    consoleMessenger_->send_env_msg(threadIdx_, ss.str());
  }
  state_.biddingStep(actionIdx);
}

json BridgeEnv::jsonObj() const {
  json j = state_.getJson();
  j["data_idx"] = getIdx();

  if (option_.saveFeatureHistory) {
    j["feature_history"] = json::array();
    // save feature history if there is.
    for (const auto& f : featureHistory_) {
      json entry;
      for (const auto& k2v : f) {
        float* data = k2v.second.data_ptr<float>();
        entry[k2v.first] = std::vector<float>(data, data + k2v.second.size(0));
      }
      j["feature_history"].push_back(entry);
    }
  }

  return j;
}

rela::TensorDict BridgeEnv::getPlayingFeature(int tableIdx, int seat,
                                              bool gtCards) const {
  torch::Tensor selfHand = state_.computeHandFeature(seat, tableIdx);
  torch::Tensor cardsPlayed = state_.computePlayedCardsFeature(tableIdx);
  torch::Tensor strain =
      torch::empty({}, torch::kInt64).fill_(state_.getContractStrain(tableIdx));

  rela::TensorDict dict = {
      {"s", selfHand}, {"strain", strain}, {"cards_played", cardsPlayed}};

  if (gtCards) {
    const torch::Tensor cards = state_.getCompleteInfoEncoding(seat);
    dict.emplace("cards", cards);
  }

  return dict;
}

std::string BridgeEnv::info() const { return state_.printHandAndBidding(); }

}  // namespace bridge
