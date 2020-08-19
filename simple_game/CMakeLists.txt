cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(jps)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O4 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

# get and append paths for finding dep
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(jps search.cc main.cpp)
target_include_directories(jps PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_SOURCE_DIR}/../third_party ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include)

target_link_libraries(jps "${TORCH_LIBRARIES}")
set_property(TARGET jps PROPERTY CXX_STANDARD 14)
