#include "cpp/score.h"
#include "cxxopts/include/cxxopts.hpp"

#include <iostream>

using namespace bridge;
using namespace std;

int main(int argc, char *argv[]) {
    // Compute score given the contract and the tricks. 
    // Usage: ./score_utils contractTbl0 numTrickTbl0 doubleTbl0 vulTbl0 contractTbl1 numTrickTbl1 doubleTbl1 vulTbl1 
    // E.g.:  ./score_utils 1H 6 1 1 3N 9 0 0

    cxxopts::Options cmdOptions(
      "Score Utility", "Bridge score utility");

    cmdOptions.add_options()(
        "c0", "contract in Table 0", cxxopts::value<string>()->default_value("1C"))(
        "declarer0", "Who is the declarer in Tbl 0", cxxopts::value<string>()->default_value("ns"))(
        "t0", "number of tricks in Table 0", cxxopts::value<int>()->default_value("7"))(
        "d0", "whether the contract is not doubled (0), doubled (1) or redoubled (2)", cxxopts::value<int>()->default_value("0"))(
        "v0", "whether the declarer is under vulnerability (false = novul, true = vul)", cxxopts::value<bool>()->default_value("false"))(
        "c1", "contract in Table 1", cxxopts::value<string>()->default_value("1C"))(
        "declarer1", "Who is the declarer in Tbl 1", cxxopts::value<string>()->default_value("ns"))(
        "t1", "number of tricks in Table 1", cxxopts::value<int>()->default_value("7"))(
        "d1", "whether the contract is not doubled (0), doubled (1) or redoubled (2)", cxxopts::value<int>()->default_value("0"))(
        "v1", "whether the declarer is under vulnerability (false = novul, true = vul)", cxxopts::value<bool>()->default_value("false"));

    auto result = cmdOptions.parse(argc, argv); 
    Bid contract0 = Bid(result["c0"].as<string>());
    Bid contract1 = Bid(result["c1"].as<string>());

    std::cout << "v0: " << result["v0"].as<bool>() << std::endl;
    std::cout << "v1: " << result["v1"].as<bool>() << std::endl;

    int score0 = computeDeclarerScore(contract0, result["t0"].as<int>(), result["d0"].as<int>(), result["v0"].as<bool>());
    int score1 = computeDeclarerScore(contract1, result["t1"].as<int>(), result["d1"].as<int>(), result["v1"].as<bool>());

    // NS or EW
    score0 = result["declarer0"].as<string>() == "ns" ? score0 : -score0;
    score1 = result["declarer1"].as<string>() == "ns" ? score1 : -score1;

    float normalized_score = computeNormalizedScore(score0, score1);
    std::cout << score0 << " " << score1 << " " << normalized_score << std::endl;

    return 0;
}
