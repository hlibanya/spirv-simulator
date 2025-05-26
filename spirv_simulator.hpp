#pragma once

#ifndef ARM_SPIRV_SIMULATOR_HPP
#define ARM_SPIRV_SIMULATOR_HPP


// ============================================================================
//  Limitations (for clarity)
//    * Bounds are *not* checked.
//    * No support for scalars with more than 64 bits
//    * No byte‑addressed aliasing – each element is a distinct `Value`.
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

//  Flip SPIRV_HEADERS_PRESENT to 1 to auto‑pull the SPIR‑V-Headers from the environment.
#define SPV_ENABLE_UTILITY_CODE 1

#ifndef SPIRV_HEADERS_PRESENT
#define SPIRV_HEADERS_PRESENT 0
#endif
#if SPIRV_HEADERS_PRESENT
#include <spirv/unified1/spirv.hpp>
#else
#include "spirv.hpp"
#endif

struct Instruction{
    spv::Op opcode;

    // word_count is the total number of words, including the word holding the opcode + wordcount value
    // Therefore this is a redundant value as the first uint32 in words will also hold this, but its
    // included for simplicity and clarity
    uint16_t word_count;
    std::span<const uint32_t> words;
};



struct Type{
    enum class Kind{
        Void,
        Bool,
        Int,
        Float,
        Vector,
        Matrix,
        Array,
        Struct,
        Pointer
    } kind;

    struct ScalarTypeData{uint32_t width; bool is_signed;};
    struct VectorTypeData{uint32_t elem_type_id; uint32_t elem_count;};
    struct MatrixTypeData{uint32_t col_type_id;  uint32_t col_count;};
    struct ArrayTypeData{uint32_t elem_type_id; uint32_t length_id;};
    struct PointerTypeData{uint32_t storage_class; uint32_t pointee_type_id;};

    union{
        ScalarTypeData scalar;
        VectorTypeData vector;
        MatrixTypeData matrix;
        ArrayTypeData array;
        PointerTypeData pointer;
    };
    Type(): kind(Kind::Void){scalar = {0, false};}
};

struct AggregateV;
struct PointerV;
struct VectorV;
struct MatrixV;

using Value = std::variant<
    std::monostate,
    uint64_t,
    int64_t,
    double,
    std::shared_ptr<VectorV>,
    std::shared_ptr<MatrixV>,
    std::shared_ptr<AggregateV>,
    PointerV>;

struct VectorV{
    std::vector<Value> elems;
};

struct MatrixV{
    std::vector<Value> cols;
};

struct AggregateV{
    std::vector<Value> elems;
};  // array or struct

struct PointerV {
    // Always the index of the value that this pointer points to
    uint32_t obj_id;
    uint32_t storage_class;

    // If it points to a value inside a composite, aggregate or array value. This is the indirection path within said value.
    std::vector<uint32_t> idx_path;
};

void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction);

class SPIRVSimulator{
public:
    explicit SPIRVSimulator(std::vector<uint32_t> program_words, bool verbose=false);
    void Run();
private:
    // Parsing artefacts
    bool verbose_;
    std::vector<uint32_t> program_words_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> spec_instr_words_;
    std::span<const uint32_t> stream_;
    std::vector<Instruction> instructions_;
    std::unordered_map<uint32_t, Instruction> spec_instructions_;
    std::unordered_map<uint32_t, size_t> result_id_to_inst_index_;
    std::unordered_map<uint32_t, Type> types_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members_;
    std::vector<uint32_t> entry_points_;
    std::vector<Instruction> unimplemented_instructions_;
    std::unordered_map<uint32_t, uint32_t> forward_type_declarations_;

    struct FunctionInfo{
        size_t inst_index;
        size_t first_inst_index;
        std::vector<uint32_t> parameter_ids_;
        std::vector<uint32_t> parameter_type_ids_;
    };
    uint32_t prev_defined_func_id_;
    std::unordered_map<uint32_t, FunctionInfo> funcs_;

    std::unordered_map<uint32_t, std::string> extended_imports_;

    // Control flow
    uint32_t prev_block_id_ = 0;
    uint32_t current_block_id_ = 0;

    // Heaps & frames
    struct Frame{
        size_t pc;
        uint32_t result_id;
        std::unordered_map<uint32_t, Value> locals;
        std::unordered_map<uint32_t, Value> func_heap;
    };
    std::vector<Frame> call_stack_;
    std::unordered_map<uint32_t, Value> globals_;
    // storage‑class (key) heaps
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, Value>> heaps_; 

    // Dispatcher
    using DispatcherType = std::function<void(const Instruction&)>;
    std::unordered_map<spv::Op, DispatcherType> opcode_dispatchers_;

    // Handlers used by the OpExtInst handler
    // Implementation of the opartions in the GLSL extended set
    void GLSLExtHandler(
        uint32_t type_id,
        uint32_t result_id,
        uint32_t instruction_literal,
        const std::span<const uint32_t>& operand_words);

    // Helpers
    void DecodeHeader();
    void ParseAll();
    void RegisterOpcodeHandlers();
    void Validate();
    void ExecuteInstruction(const Instruction&);
    void PrintInstruction(const Instruction&);
    void HandleUnimplementedOpcode(const Instruction&);
    Value MakeScalar(uint32_t type_id, const std::span<const uint32_t>& words);
    Value MakeDefault(uint32_t type_id);
    Value& Deref(const PointerV &ptr);
    Value& GetValue(uint32_t result_id);
    void SetValue(uint32_t result_id, const Value& value);

    std::unordered_map<uint32_t,Value>& Heap(uint32_t sc){ return heaps_[sc]; }

    // Opcode handlers
    void T_Void(const Instruction&);
    void T_Bool(const Instruction&);
    void T_Int(const Instruction&);
    void T_Float(const Instruction&);
    void T_Vector(const Instruction&);
    void T_Matrix(const Instruction&);
    void T_Array(const Instruction&); 
    void T_Struct(const Instruction&);
    void T_Pointer(const Instruction&);
    void T_ForwardPointer(const Instruction&);
    void T_RuntimeArray(const Instruction&);
    void T_Function(const Instruction&);
    void Op_ExtInstImport(const Instruction&);
    void Op_Constant(const Instruction&);
    void Op_ConstantComposite(const Instruction&);
    void Op_CompositeConstruct(const Instruction&);
    void Op_Variable(const Instruction&);
    void Op_Load(const Instruction&);
    void Op_Store(const Instruction&);
    void Op_AccessChain(const Instruction&);
    void Op_Function(const Instruction&);
    void Op_FunctionEnd(const Instruction&);
    void Op_FunctionCall(const Instruction&);
    void Op_Label(const Instruction&);
    void Op_Branch(const Instruction&);
    void Op_BranchConditional(const Instruction&);
    void Op_Return(const Instruction&);
    void Op_ReturnValue(const Instruction&);
    void Op_INotEqual(const Instruction&);
    void Op_FAdd(const Instruction&);
    void Op_ExtInst(const Instruction&);
    void Op_SelectionMerge(const Instruction&);
    void Op_FMul(const Instruction&);
    void Op_LoopMerge(const Instruction&);
    void Op_IAdd(const Instruction&);
    void Op_LogicalNot(const Instruction&);
    void Op_Capability(const Instruction&);
    void Op_Extension(const Instruction&);
    void Op_MemoryModel(const Instruction&);
    void Op_ExecutionMode(const Instruction&);
    void Op_Source(const Instruction&);
    void Op_SourceExtension(const Instruction&);
    void Op_Name(const Instruction&);
    void Op_MemberName(const Instruction&);
    void Op_Decorate(const Instruction&);
    void Op_MemberDecorate(const Instruction&);
    void Op_SpecConstant(const Instruction&);
    void Op_SpecConstantOp(const Instruction&);
    void Op_SpecConstantComposite(const Instruction&);
    void Op_ArrayLength(const Instruction&);
    void Op_UGreaterThanEqual(const Instruction&);
    void Op_Phi(const Instruction&);
    void Op_ConvertUToF(const Instruction&);
    void Op_ConvertSToF(const Instruction&);
    void Op_FDiv(const Instruction&);
    void Op_FSub(const Instruction&);
    void Op_VectorTimesScalar(const Instruction&);
    void Op_SLessThan(const Instruction&);
    void Op_Dot(const Instruction&);
    void Op_FOrdGreaterThan(const Instruction&);
    void Op_CompositeExtract(const Instruction&);
    void Op_Bitcast(const Instruction&);
    void Op_IMul(const Instruction&);
};

#endif
