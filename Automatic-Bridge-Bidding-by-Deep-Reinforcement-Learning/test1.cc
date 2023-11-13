#include <stdio.h>
#include <stdlib.h>
#include <iostream>


int main (const int argc, char ** argv) {
  auto result = system("octave automatic_bridge_bidding_model.m");
  std::cout << result << std::endl;
}
