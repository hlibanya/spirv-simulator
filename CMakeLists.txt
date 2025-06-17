cmake_minimum_required(VERSION 3.9)

project(spirv_simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_FILES spirv_simulator.cpp)
set(HEADER_FILES spirv_simulator.hpp spirv.hpp)

add_executable(spirv_simulator ${SOURCE_FILES} ${HEADER_FILES} main.cpp)
target_compile_options(spirv_simulator PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20)

add_executable(spirv_opcode_support_checker ${SOURCE_FILES} ${HEADER_FILES} main_opcode_checker.cpp)
target_compile_options(spirv_opcode_support_checker PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20)
