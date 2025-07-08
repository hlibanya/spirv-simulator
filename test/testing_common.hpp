#pragma once

#ifndef ARM_TESTING_COMMON_HPP
#define ARM_TESTING_COMMON_HPP

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <spirv_simulator.hpp>

std::ostream& operator<<(std::ostream& os, SPIRVSimulator::Value const& v);

class SPIRVSimulatorMockBase : public SPIRVSimulator::SPIRVSimulator
{
  public:
    SPIRVSimulatorMockBase() { RegisterOpcodeHandlers(); }
    ~SPIRVSimulatorMockBase() = default;

    using SPIRVSimulator::SPIRVSimulator::ExecuteInstruction;
};

#endif