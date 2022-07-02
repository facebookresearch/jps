cmake_minimum_required(VERSION 3.0.0)
project(score_utils VERSION 0.1.0)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O4 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

include(CTest)
enable_testing()

add_executable(score_utils ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bid.cc ${CMAKE_CURRENT_SOURCE_DIR}/cpp/score.cc ${CMAKE_CURRENT_SOURCE_DIR}/cpp/score_utils.cc)
target_include_directories(score_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set_property(TARGET score_utils PROPERTY CXX_STANDARD 14)