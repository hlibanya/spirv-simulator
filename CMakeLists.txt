cmake_minimum_required(VERSION 3.9)

# Note, this MUST NOT be compile with -ffinite-math or -ffast-math (or any optimization option that affects nan of inf representations), this will break nan operations that we need for correct SPIRV simulation.
# In general, leave the -fno-fast-math option below alone so that we keep full IEEE 754 compliance.

project(spirv_simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_FILES spirv_simulator.cpp)
set(HEADER_FILES spirv_simulator.hpp spirv.hpp)

add_executable(spirv_simulator ${SOURCE_FILES} ${HEADER_FILES} main.cpp)
target_compile_options(spirv_simulator PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)

add_executable(spirv_opcode_support_checker ${SOURCE_FILES} ${HEADER_FILES} main_opcode_checker.cpp)
target_compile_options(spirv_opcode_support_checker PRIVATE -Wall -Wextra -Wpedantic -Wshadow -std=c++20 -fno-fast-math)
