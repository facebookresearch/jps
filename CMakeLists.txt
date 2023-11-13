CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
project(bridge)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors -fvisibility=hidden"
)

# set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_package(SQLite3 REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rela)

find_package(Torch REQUIRED)

add_library(bridge_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/auction.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bid.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bridge_env.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/duplicate_bridge_env.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/game_state.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/game_state2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/greedy_play_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/hand.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pbn.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/playing_sequence.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/random_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/score.cc
)

target_include_directories(bridge_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(bridge_cpp PUBLIC _rela ${SQLITE3_LIBRARIES} sqlite3)

pybind11_add_module(bridge
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pybind.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bridge_env.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/console_messenger.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_game/search.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rela/sql.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rela/IndexedLoggerFactory.cc)

pybind11_add_module(simple_game
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_game/pybind.cc)

message(STATUS ${SQLITE3_INCLUDE_DIRS})
message(STATUS ${SQLITE3_LIBRARIES})

target_link_libraries(bridge PUBLIC bridge_cpp ${SQLITE3_LIBRARIES} sqlite3)
target_include_directories(bridge PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
  ${SQLITE3_INCLUDE_DIRS}
)

target_link_libraries(simple_game PUBLIC _rela)
target_include_directories(simple_game PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


#include(FetchContent)

#FetchContent_Declare(googletest
#  GIT_REPOSITORY https://github.com/google/googletest.git
#  GIT_TAG        master
#)

#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# FetchContent_MakeAvailable(googletest)
#FetchContent_GetProperties(googletest)
#if (NOT googletest_POPULATED)
#  FetchContent_Populate(googletest)
#  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
#endif()

#enable_testing()
#add_subdirectory(test)
