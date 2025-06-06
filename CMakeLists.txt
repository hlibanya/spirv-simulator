cmake_minimum_required(VERSION 3.9)

project(spirv_simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_FILES main.cpp spirv_simulator.cpp)
set(HEADER_FILES spirv_simulator.hpp spirv.hpp)
add_executable(spirv_simulator ${SOURCE_FILES} ${HEADER_FILES})
target_compile_options(spirv_simulator PRIVATE -Wall -Wextra -Wpedantic -std=c++20)
