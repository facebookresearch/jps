#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

class GameInterface {
 public:
   virtual void reset() = 0;

   // Get (key, playerId, numAction)
   // Note that if the state is terminal then call .get() might give undefined behaavior.
   virtual std::tuple<std::string, int, int> get() const = 0; 

   // Include chance.
   virtual int getNumPlayer() const = 0;
   virtual std::unique_ptr<GameInterface> forward(int action) const = 0;

   virtual bool isTerminal() const = 0;
   virtual std::vector<float> getUtilities() const = 0;

   virtual std::string info() const { return ""; }

   virtual bool hasOptimalPolicy() const { return false; }
   virtual std::vector<float> getOptimalStrategy(const std::string &) const { return {}; }

   virtual ~GameInterface() = default;
};

