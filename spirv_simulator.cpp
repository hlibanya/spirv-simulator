#include "spirv_simulator.hpp"

constexpr uint32_t kWordCountShift = 16u;
constexpr uint32_t kOpcodeMask = 0xFFFFu;


void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction){
    uint32_t first = program_words.front();
    instruction.word_count = first >> kWordCountShift;
    instruction.opcode = (spv::Op)(first & kOpcodeMask);
    if(!instruction.word_count || instruction.word_count > program_words.size()){
        throw std::runtime_error("SPIRV simulator: Bad instruction size");
    }

    instruction.words = program_words.first(instruction.word_count);
    program_words = program_words.subspan(instruction.word_count);
}

SPIRVSimulator::SPIRVSimulator(std::vector<uint32_t> program_words, bool verbose): verbose_(verbose), program_words_(std::move(program_words)){
    stream_ = program_words_;
    DecodeHeader();
    RegisterOpcodeHandlers();
    ParseAll();
    Validate();
}

void SPIRVSimulator::DecodeHeader(){
    if(program_words_.size() < 5){
        throw std::runtime_error("SPIRV simulator: Bad header");
    }

    uint32_t magic_number = program_words_[0];
    if (magic_number != 0x07230203)
    {
        std::cout << "SPIRV simulator: WARNING magic SPIRV header number wrong, should be: " << 0x07230203 << " but is " << magic_number << std::endl;
    }
    /*
    uint32_t version = program_words_[1];
    uint32_t generator = program_words_[2];
    uint32_t bound = program_words_[3];
    uint32_t schema = program_words_[4];
    */

    stream_ = std::span<const uint32_t>(program_words_).subspan(5);
}

void SPIRVSimulator::RegisterOpcodeHandlers(){
    auto R = [this](spv::Op op, DispatcherType f){
        opcode_dispatchers_[op] = std::move(f);
    };

    R(spv::Op::OpTypeVoid,               [this](const Instruction& i){T_Void(i);});
    R(spv::Op::OpTypeBool,               [this](const Instruction& i){T_Bool(i);});
    R(spv::Op::OpTypeInt,                [this](const Instruction& i){T_Int(i);});
    R(spv::Op::OpTypeFloat,              [this](const Instruction& i){T_Float(i);});
    R(spv::Op::OpTypeVector,             [this](const Instruction& i){T_Vector(i);});
    R(spv::Op::OpTypeMatrix,             [this](const Instruction& i){T_Matrix(i);});
    R(spv::Op::OpTypeArray,              [this](const Instruction& i){T_Array(i);});
    R(spv::Op::OpTypeStruct,             [this](const Instruction& i){T_Struct(i);});
    R(spv::Op::OpTypePointer,            [this](const Instruction& i){T_Pointer(i);});
    R(spv::Op::OpTypeForwardPointer,     [this](const Instruction& i){T_ForwardPointer(i);});
    R(spv::Op::OpTypeRuntimeArray,       [this](const Instruction& i){T_RuntimeArray(i);});
    R(spv::Op::OpTypeFunction,           [this](const Instruction& i){T_Function(i);});
    R(spv::Op::OpExtInstImport,          [this](const Instruction& i){Op_ExtInstImport(i);});
    R(spv::Op::OpConstant,               [this](const Instruction& i){Op_Constant(i);});
    R(spv::Op::OpConstantComposite,      [this](const Instruction& i){Op_ConstantComposite(i);});
    R(spv::Op::OpCompositeConstruct,     [this](const Instruction& i){Op_CompositeConstruct(i);});
    R(spv::Op::OpVariable,               [this](const Instruction& i){Op_Variable(i);});
    R(spv::Op::OpLoad,                   [this](const Instruction& i){Op_Load(i);});
    R(spv::Op::OpStore,                  [this](const Instruction& i){Op_Store(i);});
    R(spv::Op::OpAccessChain,            [this](const Instruction& i){Op_AccessChain(i);});
    R(spv::Op::OpInBoundsAccessChain,    [this](const Instruction& i){Op_AccessChain(i);});
    R(spv::Op::OpFunction,               [this](const Instruction& i){Op_Function(i);});
    R(spv::Op::OpFunctionEnd,            [this](const Instruction& i){Op_FunctionEnd(i);});
    R(spv::Op::OpFunctionCall,           [this](const Instruction& i){Op_FunctionCall(i);});
    R(spv::Op::OpLabel,                  [this](const Instruction& i){Op_Label(i);});
    R(spv::Op::OpBranch,                 [this](const Instruction& i){Op_Branch(i);});
    R(spv::Op::OpBranchConditional,      [this](const Instruction& i){Op_BranchConditional(i);});
    R(spv::Op::OpReturn,                 [this](const Instruction& i){Op_Return(i);});
    R(spv::Op::OpReturnValue,            [this](const Instruction& i){Op_ReturnValue(i);});
    R(spv::Op::OpINotEqual,              [this](const Instruction& i){Op_INotEqual(i);});
    R(spv::Op::OpFAdd,                   [this](const Instruction& i){Op_FAdd(i);});
    R(spv::Op::OpExtInst,                [this](const Instruction& i){Op_ExtInst(i);});
    R(spv::Op::OpSelectionMerge,         [this](const Instruction& i){Op_SelectionMerge(i);});
    R(spv::Op::OpFMul,                   [this](const Instruction& i){Op_FMul(i);});
    R(spv::Op::OpLoopMerge,              [this](const Instruction& i){Op_LoopMerge(i);});
    R(spv::Op::OpIAdd,                   [this](const Instruction& i){Op_IAdd(i);});
    R(spv::Op::OpLogicalNot,             [this](const Instruction& i){Op_LogicalNot(i);});
    R(spv::Op::OpCapability,             [this](const Instruction& i){Op_Capability(i);});
    R(spv::Op::OpExtension,              [this](const Instruction& i){Op_Extension(i);});
    R(spv::Op::OpMemoryModel,            [this](const Instruction& i){Op_MemoryModel(i);});
    R(spv::Op::OpExecutionMode,          [this](const Instruction& i){Op_ExecutionMode(i);});
    R(spv::Op::OpSource,                 [this](const Instruction& i){Op_Source(i);});
    R(spv::Op::OpSourceExtension,        [this](const Instruction& i){Op_SourceExtension(i);});
    R(spv::Op::OpName,                   [this](const Instruction& i){Op_Name(i);});
    R(spv::Op::OpMemberName,             [this](const Instruction& i){Op_MemberName(i);});
    R(spv::Op::OpDecorate,               [this](const Instruction& i){Op_Decorate(i);});
    R(spv::Op::OpMemberDecorate,         [this](const Instruction& i){Op_MemberDecorate(i);});
    R(spv::Op::OpArrayLength,            [this](const Instruction& i){Op_ArrayLength(i);});
    R(spv::Op::OpSpecConstant,           [this](const Instruction& i){Op_SpecConstant(i);});
    R(spv::Op::OpSpecConstantOp,         [this](const Instruction& i){Op_SpecConstantOp(i);});
    R(spv::Op::OpSpecConstantComposite,  [this](const Instruction& i){Op_SpecConstantComposite(i);});
    R(spv::Op::OpUGreaterThanEqual,      [this](const Instruction& i){Op_UGreaterThanEqual(i);});
    R(spv::Op::OpPhi,                    [this](const Instruction& i){Op_Phi(i);});
    R(spv::Op::OpConvertUToF,            [this](const Instruction& i){Op_ConvertUToF(i);});
    R(spv::Op::OpConvertSToF,            [this](const Instruction& i){Op_ConvertSToF(i);});
    R(spv::Op::OpFDiv,                   [this](const Instruction& i){Op_FDiv(i);});
    R(spv::Op::OpFSub,                   [this](const Instruction& i){Op_FSub(i);});
    R(spv::Op::OpVectorTimesScalar,      [this](const Instruction& i){Op_VectorTimesScalar(i);});
    R(spv::Op::OpSLessThan,              [this](const Instruction& i){Op_SLessThan(i);});
    R(spv::Op::OpDot,                    [this](const Instruction& i){Op_Dot(i);});
    R(spv::Op::OpFOrdGreaterThan,        [this](const Instruction& i){Op_FOrdGreaterThan(i);});
    R(spv::Op::OpCompositeExtract,       [this](const Instruction& i){Op_CompositeExtract(i);});
    R(spv::Op::OpBitcast,                [this](const Instruction& i){Op_Bitcast(i);});
    R(spv::Op::OpIMul,                   [this](const Instruction& i){Op_IMul(i);});
}

void SPIRVSimulator::Validate(){
    // TODO: Expand this (a lot)
    for(auto &[id, t] : types_){
        if(t.kind == Type::Kind::Array || t.kind == Type::Kind::Vector){
            if(!types_.contains(t.vector.elem_type_id)){
                throw std::runtime_error("SPIRV simulator: Missing elem type");
            }
        }
        if(t.kind == Type::Kind::Matrix && !types_.contains(t.matrix.col_type_id)){
            throw std::runtime_error("SPIRV simulator: Missing col type");
        }
        if(t.kind == Type::Kind::Pointer && !types_.contains(t.pointer.pointee_type_id)){
            throw std::runtime_error("SPIRV simulator: Missing pointee type");
        }
        if(t.kind == Type::Kind::Bool || t.kind == Type::Kind::Int || t.kind == Type::Kind::Float){
            if (t.scalar.width != 32 && t.scalar.width != 64){
                throw std::runtime_error("SPIRV simulator: We only allow 32 and 64 bit scalars at present");
            }
        }
    }

    if (sizeof(void*) != 8){
        throw std::runtime_error("SPIRV simulator: Systems with non 64 bit pointers are not supported");
    }
}

void SPIRVSimulator::ParseAll(){
    size_t instruction_index = 0;

    if (verbose_){
        std::cout << "SPIRV simulator: Parsing instructions:" << std::endl;
    }

    bool in_function = false;

    std::vector<uint32_t> unimplemented_instructions;

    while(!stream_.empty()){
        Instruction instruction;
        DecodeInstruction(stream_, instruction);
        instructions_.push_back(instruction);

        bool has_result = false;
        bool has_type = false;

        spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

        if (has_result){
            if (has_type){
                result_id_to_inst_index_[instruction.words[2]] = instruction_index;
            } else {
                result_id_to_inst_index_[instruction.words[1]] = instruction_index;
            }
        }

        if (verbose_){
            PrintInstruction(instruction);
        }

        switch (instruction.opcode){
            case spv::Op::OpFunction:{
                in_function = true;
                funcs_[instruction.words[2]] = {instruction_index, instruction_index + 1, {}, {}};
                prev_defined_func_id_ = instruction.words[2];
                break;
            }
            case spv::Op::OpFunctionEnd:
                in_function = false;
                break;
            case spv::Op::OpFunctionParameter:{
                funcs_[prev_defined_func_id_].parameter_ids_.push_back(instruction.words[2]);
                funcs_[prev_defined_func_id_].parameter_type_ids_.push_back(instruction.words[1]);
                break;
            }
            case spv::Op::OpEntryPoint:{
                entry_points_.push_back(instruction.words[2]); // word[2] is <entry‑point id>
                break;
            }
            /*case spv::Op::OpExtInstImport:{
                extended_imports_[instruction.words[1]] = 0;
                break;
            }*/
            default:{
                if (!in_function){
                    ExecuteInstruction(instruction);
                } else {
                    bool is_implemented = opcode_dispatchers_.find(instruction.opcode) != opcode_dispatchers_.end();
                    if(!is_implemented){
                        HandleUnimplementedOpcode(instruction);
                    }
                }
                break;
            }
        }

        ++instruction_index;
    }

    if (verbose_){
        std::cout << "SPIRV simulator: Parsing complete!\n" << std::endl;
    }

    if (unimplemented_instructions_.size() && verbose_){
        // TODO: Deduplicate, probably better to use a map to track the ids
        std::cout << "SPIRV simulator: The following instructions are unsupported:" << std::endl;

        for (auto instruction : unimplemented_instructions_){
            
            PrintInstruction(instruction);
        }

        std::cout << std::endl;
    }
}

void SPIRVSimulator::Run(){
    // TODO: Add input data

    if(funcs_.empty()){
        std::cerr << "SPIRV simulator: No functions defined in the shader, cannot start execution" << std::endl;
        return;
    }

    if (verbose_){
        std::cout << "SPIRV simulator: Starting execution:" << std::endl;
    }

    // TODO: Take entry points from input data
    for (uint32_t entry_point : entry_points_){
        if (funcs_.find(entry_point) == funcs_.end()){
            if (verbose_){
                std::cout << "SPIRV simulator: Warning, entry point function with index: " << entry_point << " not found!" << std::endl;
            }
            continue;
        }

        FunctionInfo& function_info = funcs_[entry_point];
        // We can set the return value to whatever, ignored if the call stack is empty on return
        call_stack_.push_back({function_info.first_inst_index, 0, {}, {}});

        while(!call_stack_.empty()){
            auto& stack_frame = call_stack_.back();
            const Instruction& instruction = instructions_[stack_frame.pc++];

            if (verbose_){
                PrintInstruction(instruction);
            }

            ExecuteInstruction(instruction);
        }
    }

    if (verbose_){
        std::cout << "SPIRV simulator: Execution complete!\n" << std::endl;
    }
}

void SPIRVSimulator::ExecuteInstruction(const Instruction& instruction){
    auto dispatcher = opcode_dispatchers_.find(instruction.opcode);
    if(dispatcher == opcode_dispatchers_.end()){
        HandleUnimplementedOpcode(instruction);
    }
    else{
        dispatcher->second(instruction);
    }
}

void SPIRVSimulator::HandleUnimplementedOpcode(const Instruction& instruction){
    unimplemented_instructions_.push_back(instruction);
}

void SPIRVSimulator::PrintInstruction(const Instruction& instruction){
    bool has_result = false;
    bool has_type = false;

    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (verbose_){
        uint32_t result_offset = 0;
        if (has_result){
            if (has_type){
                result_offset = 2;
            } else {
                result_offset = 1;
            }
        }
            
        if (result_offset){
            std::cout << std::setw(4) << instruction.words[result_offset] << " ";
        } else {
            std::cout << std::setw(5) << " ";
        }

        std::cout << spv::OpToString(instruction.opcode) << " "; 
        for (uint32_t i = 1; i < instruction.word_count; ++i){
            if (i == result_offset){
                continue;
            }

            if (instruction.opcode == spv::Op::OpDecorate){
                if (i == 2) {
                    std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                } else {
                    std::cout << instruction.words[i] << " ";
                }
            }else if(instruction.opcode == spv::Op::OpMemberDecorate){
                if (i == 3) {
                    std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                } else {
                    std::cout << instruction.words[i] << " ";
                }
            } else {
                std::cout << instruction.words[i] << " ";
            }
        }

        std::cout << std::endl;
    }
}

// ---------------------------------------------------------------------------
//  Value creation helpers
// ---------------------------------------------------------------------------

Value SPIRVSimulator::MakeScalar(uint32_t type_id, const std::span<const uint32_t>& words){
    const Type& type = types_.at(type_id);

    switch(type.kind){
        case Type::Kind::Int:
            if (type.scalar.width > 64){
                throw std::runtime_error("SPIRV simulator: We do not support types wider than 64 bits");;
            }
            else if (type.scalar.width > 32){
                if (type.scalar.is_signed){
                    int64_t tmp_value;
                    std::memcpy(&tmp_value, &words[0], 8);
                    return tmp_value;
                } else {
                    return (static_cast<uint64_t>(words[1]) << 32) | words[0];
                }
            } else {
                if (type.scalar.is_signed){
                    int32_t tmp_value;
                    std::memcpy(&tmp_value, &words[0], 4);
                    return (int64_t)tmp_value;
                } else {
                    return (uint64_t)words[0];
                }
            }
        case Type::Kind::Bool:
            // Just treat bools as uint64_t types for simplicity
            return (uint64_t)words[0];
        case Type::Kind::Float:{
            if (type.scalar.width > 64){
                throw std::runtime_error("SPIRV simulator: We do not support types wider than 64 bits");;
            }
            else if (type.scalar.width > 32){
                double tmp_value;
                std::memcpy(&tmp_value, &words[0], 8);
                return tmp_value;
            } else {
                float tmp_value;
                std::memcpy(&tmp_value, &words[0], 4);
                return (double)tmp_value;
            }
        }
        default:{
            throw std::runtime_error("SPIRV simulator: Unsupported scalar type, instructions are possibly corrupt");
        }
    }
}

Value SPIRVSimulator::MakeDefault(uint32_t type_id){
    const Type& type = types_.at(type_id);

    switch(type.kind){
        case Type::Kind::Int:
        case Type::Kind::Float:
        case Type::Kind::Bool:{
            const uint32_t empty_array[]{0,0};
            std::span<const uint32_t> empty_span{empty_array};

            return MakeScalar(type_id, empty_span);
        }
        case Type::Kind::Vector:{
            auto vec = std::make_shared<VectorV>();
            vec->elems.reserve(type.vector.elem_count);
            for (uint32_t i = 0; i < type.vector.elem_count; ++i){
                vec->elems.push_back(MakeDefault(type.vector.elem_type_id));
            }

            return vec;
        }
        case Type::Kind::Matrix:{
            auto matrix = std::make_shared<MatrixV>();
            matrix->cols.reserve(type.matrix.col_count);
            for(uint32_t i = 0; i < type.matrix.col_count; ++i){
                Value mat_val = MakeDefault(type.matrix.col_type_id);
                matrix->cols.push_back(mat_val);
            }

            return matrix;
        }
        case Type::Kind::Array:{
            uint64_t len;
            if (type.array.length_id == 0){
                // This is a OpTypeRuntimeArray
                // Length is either set by OpArrayLength or it is unknown
                std::cout << "SPIRV simulator: WARNING not handling runtime array length at present, setting to 1 which will crash with OOB errors in most cases" << std::endl;
                len = 1;
            } else {
                len = std::get<uint64_t>(GetValue(type.array.length_id));
            }

            auto aggregate = std::make_shared<AggregateV>();
            aggregate->elems.reserve(len);
            for(uint32_t i = 0; i < len; ++i){
                aggregate->elems.push_back(MakeDefault(type.array.elem_type_id));
            }

            return aggregate;
        }
        case Type::Kind::Struct:{
            auto structure = std::make_shared<AggregateV>();
            for(auto member : struct_members_.at(type_id)){
                structure->elems.push_back(MakeDefault(member));
            }

            return structure;
        }
        default:{
            return std::monostate{};
        }
    }
}

// ---------------------------------------------------------------------------
//  Dereference and access helpers
// ---------------------------------------------------------------------------
Value& SPIRVSimulator::Deref(const PointerV &ptr){
    // TODO: Double check correctness here after refactor
    auto& heap = (ptr.storage_class == (uint32_t)spv::StorageClass::StorageClassFunction) ? call_stack_.back().func_heap : Heap(ptr.storage_class);

    Value* value = &heap.at(ptr.obj_id);
    for(size_t depth = 0; depth < ptr.idx_path.size(); ++depth){
        uint32_t indirection_index = ptr.idx_path[depth];

        if(std::holds_alternative<std::shared_ptr<AggregateV>>(*value)){
            auto agg = std::get<std::shared_ptr<AggregateV>>(*value);

            if(indirection_index >= agg->elems.size()){
                throw std::runtime_error("SPIRV simulator: Aggregate index OOB");
            }

            value = &agg->elems[indirection_index];
        } else if (std::holds_alternative<std::shared_ptr<VectorV>>(*value)){
            auto vec = std::get<std::shared_ptr<VectorV>>(*value);

            if(indirection_index >= vec->elems.size()){
                throw std::runtime_error("SPIRV simulator: Vector index OOB");
            }

            value = &vec->elems[indirection_index];
        } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*value)){
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*value);

            if(indirection_index >= matrix->cols.size()){
                throw std::runtime_error("SPIRV simulator: Matrix index OOB");
            }

            value = &matrix->cols[indirection_index];
        }
        else{
            throw std::runtime_error("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    return *value;
}

Value& SPIRVSimulator::GetValue(uint32_t result_id){
    for (auto riter = call_stack_.rbegin(); riter != call_stack_.rend(); ++riter) { 
        if (riter->locals.find(result_id) != riter->locals.end()){
            return riter->locals.at(result_id);
        }
    }

    if (globals_.find(result_id) == globals_.end()){
        throw std::runtime_error("SPIRV simulator: Access to undefined variable with ID: " + std::to_string(result_id));
    }

    return globals_.at(result_id);
}

void SPIRVSimulator::SetValue(uint32_t result_id, const Value& value){
    if (call_stack_.size()){
        call_stack_.back().locals[result_id] = value;
    } else {
        globals_[result_id] = value;
    }
}

// ---------------------------------------------------------------------------
//  Ext Import implementations
// ---------------------------------------------------------------------------

void SPIRVSimulator::GLSLExtHandler(
    uint32_t type_id,
    uint32_t result_id,
    uint32_t instruction_literal,
    const std::span<const uint32_t>& operand_words
){
    const Type& type = types_.at(type_id);

    switch(instruction_literal){
        case 14:{  // Cos
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector){
                if(!(std::holds_alternative<std::shared_ptr<VectorV>>(operand))){
                    throw std::runtime_error("SPIRV simulator: Operands not of vector type in GLSLExtHandler::cos");
                }

                Value result = std::make_shared<VectorV>();
                auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i){
                    Value elem_result = (double)std::cos(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);

            } else if (type.kind == Type::Kind::Float){
                Value result = (double)std::cos(std::get<double>(operand));
                SetValue(result_id, result);
            }
            break;
        }
        default:{
            std::cout << "SPIRV simulator: Unhandled OpExtInst GLSL set operation: " << instruction_literal << std::endl;
            std::cout << "SPIRV simulator: Setting output to default value, this will likely crash" << std::endl;
            SetValue(result_id, MakeDefault(type_id));
        }
    }
}


// ---------------------------------------------------------------------------
//  Type creation handlers
// ---------------------------------------------------------------------------
void SPIRVSimulator::T_Void(const Instruction& instruction){

    Type type;
    type.kind = Type::Kind::Void;
    type.scalar = {
        0,
        false
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Bool(const Instruction& instruction){
    // We treat bools as 64 bit unsigned ints for simplicity
    Type type;
    type.kind = Type::Kind::Bool;
    type.scalar = {
        64,
        false
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Int(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Int;
    type.scalar = {
        instruction.words[2],
        (bool)instruction.words[3]
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Float(const Instruction& instruction){
    // We dont handle floats encoded in other formats than the default at present
    if (instruction.word_count > 3){
        throw std::runtime_error("SPIRV simulator: Simulator only supports IEEE 754 encoded floats at present.");
    }

    Type type;
    type.kind = Type::Kind::Float;
    type.scalar = {
        instruction.words[2],
        false
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Vector(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Vector;
    type.vector = {
        instruction.words[2],
        instruction.words[3]
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Matrix(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Matrix;
    type.matrix = {
        instruction.words[2],
        instruction.words[3]
    };
    
    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Array(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Array;
    type.array = {
        instruction.words[2],
        instruction.words[3]
    };

    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_Struct(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Struct;

    types_[instruction.words[1]] = type;

    std::vector<uint32_t> members;
    for(auto i = 2; i < instruction.word_count; ++i){
        members.push_back(instruction.words[i]);
    }

    struct_members_[instruction.words[1]] = std::move(members);
}

void SPIRVSimulator::T_Pointer(const Instruction& instruction){
    Type type;
    type.kind = Type::Kind::Pointer;
    type.pointer = {
        instruction.words[2],
        instruction.words[3]
    };
    types_[instruction.words[1]] = type;
}

void SPIRVSimulator::T_ForwardPointer(const Instruction& instruction) {
    // TODO: May not need this
    uint32_t pointer_type_id = instruction.words[1];
    uint32_t storage_class = instruction.words[2];
    forward_type_declarations_[pointer_type_id] = storage_class;
}

void SPIRVSimulator::T_RuntimeArray(const Instruction& instruction) {
    uint32_t result_id = instruction.words[1];
    uint32_t elem_type_id = instruction.words[2];

    Type type;
    type.kind = Type::Kind::Array;
    type.array = {
        elem_type_id,
        0
    };
    types_[result_id] = type;

}

void SPIRVSimulator::T_Function(const Instruction&){
    // This info is redundant for us, so treat it as a NOP
}


// ---------------------------------------------------------------------------
//  Oparation implementations
// ---------------------------------------------------------------------------
void SPIRVSimulator::Op_ExtInstImport(const Instruction& instruction){
    /*
    OpExtInstImport

    Import an extended set of instructions. It can be later referenced by the Result <id>.

    Name is the extended instruction-set’s name string. Before version 1.6, there must be an external specification defining the semantics
    for this extended instruction set. Starting with version 1.6, if Name starts with "NonSemantic.", including the period that separates
    the namespace "NonSemantic" from the rest of the name, it is encouraged for a specification to exist on the SPIR-V Registry,
    but it is not required.

    Starting with version 1.6, an extended instruction-set name which is prefixed with "NonSemantic." is guaranteed to contain only
    non-semantic instructions, and all OpExtInst instructions referencing this set can be ignored. All instructions within such a set
    must have only <id> operands; no literals. When literals are needed, then the Result <id> from an OpConstant or OpString instruction
    is referenced as appropriate. Result <id>s from these non-semantic instruction-set instructions must be used only in other non-semantic
    instructions.

    See Extended Instruction Sets for more information.
    */
    uint32_t result_id = instruction.words[1];
    // SPIRV string literals are UTF-8 encoded, so basic c++ string functionality can be used to decode them
    extended_imports_[result_id] = std::string((char*)(&instruction.words[2]), instruction.word_count - 2);
}

void SPIRVSimulator::Op_Constant(const Instruction& instruction){
    /*
    OpConstant
    Declare a new integer-type or floating-point-type scalar constant.
    Result Type must be a scalar integer type or floating-point type.
    Value is the bit pattern for the constant. Types 32 bits wide or smaller take one word.
    Larger types take multiple words, with low-order words appearing first.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    if ((type.kind != Type::Kind::Int) && (type.kind != Type::Kind::Float)) {
        throw std::runtime_error("SPIRV simulator: Constant type unsupported");
    }

    SetValue(result_id, MakeScalar(type_id, instruction.words.subspan(3)));
}

void SPIRVSimulator::Op_ConstantComposite(const Instruction& instruction){
    /*
    OpConstantComposite

    Declare a new composite constant.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same type as the
    types of the Constituents. The ordering must be the same between the top-level types in Result Type and the Constituents.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columns of a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result.
    The Constituents must appear in the order needed by the definition of the Result Type.
    The Constituents must all be <id>s of non-specialization constant-instruction declarations or an OpUndef.
    */
    Op_CompositeConstruct(instruction);
}

void SPIRVSimulator::Op_CompositeConstruct(const Instruction& instruction){
    /*
    OpCompositeConstruct

    Construct a new composite object from a set of constituent objects.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same
    type as the types of the operands, with one exception.

    The exception is that for constructing a vector, the operands may also be vectors with the same component
    type as the Result Type component type.

    If constructing a vector, the total number of components in all the operands must equal
    the number of components in Result Type.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columnsof a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result,with one exception.

    The exception is that for constructing a vector, a contiguous subset of the scalars consumed can be represented by
    a vector operand instead.

    The Constituents must appear in the order needed by the definition of the type of the result.
    If constructing a vector, there must be at least two Constituent operands.

    */
    // Composite: An aggregate (structure or an array), a matrix, or a vector.
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    if(type.kind == Type::Kind::Vector){
        auto vec = std::make_shared<VectorV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            const Value& component_value = GetValue(instruction.words[i]);

            if (std::holds_alternative<std::shared_ptr<VectorV>>(component_value)){
                std::shared_ptr<VectorV> component_vector = std::get<std::shared_ptr<VectorV>>(component_value);

                for (auto& vec_component : component_vector->elems){
                    vec->elems.push_back(vec_component);
                }

            } else {
                vec->elems.push_back(component_value);
            }
        }

        SetValue(result_id, vec);
    }
    else if (type.kind == Type::Kind::Matrix){
        auto matrix = std::make_shared<MatrixV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            matrix->cols.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, matrix);
    }
    else if(type.kind == Type::Kind::Struct || type.kind == Type::Kind::Array){
        auto aggregate = std::make_shared<AggregateV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            aggregate->elems.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, aggregate);
    }
    else{
        throw std::runtime_error("SPIRV simulator: CompositeConstruct not implemented yet for type");
    }
}

void SPIRVSimulator::Op_Variable(const Instruction& instruction){
    /*
    OpVariable

    Allocate an object in memory, resulting in a pointer to it, which can be used with OpLoad and OpStore.

    Result Type must be an OpTypePointer. Its Type operand is the type of object in memory.
    Storage Class is the Storage Class of the memory holding the object. It must not be Generic.
    It must be the same as the Storage Class operand of the Result Type.

    If Storage Class is Function, the memory is allocated on execution of the instruction for the current invocation for
    each dynamic instance of the function. The current invocation’s memory is deallocated when it executes any function
    termination instruction of the dynamic instance of the function it was allocated by.

    Initializer is optional. If Initializer is present, it will be the initial value of the variable’s memory content.
    Initializer must be an <id> from a constant instruction or a global (module scope) OpVariable instruction.
    Initializer must have the same type as the type pointed to by Result Type.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t storage_class = instruction.words[3];

    const Type& type = types_.at(type_id);

    if (type.kind != Type::Kind::Pointer){
        throw std::runtime_error("SPIRV simulator: Op_Variable must only be used to create pointer types");
    }

    Value init;
    if(instruction.word_count >= 5){
        // The instruction has initialization data
        init = GetValue(instruction.words[4]);
    }
    else{
        // No init data, set to default
        init = MakeDefault(type.pointer.pointee_type_id);
    }

    if(storage_class == (uint32_t)spv::StorageClass::StorageClassFunction){
        call_stack_.back().func_heap[result_id] = init;
    }
    else{
        Heap(storage_class)[result_id] = init;
    }

    PointerV new_pointer{result_id, storage_class, {}};
    SetValue(result_id, new_pointer);
}

void SPIRVSimulator::Op_Load(const Instruction& instruction){
    /*
    OpLoad

    Load through a pointer.

    Result Type is the type of the loaded object. It must be a type with fixed size; i.e., it must not be, nor include,
    any OpTypeRuntimeArray types.

    Pointer is the pointer to load through.
    Its type must be an OpTypePointer whose Type operand is the same as Result Type.

    If present, any Memory Operands must begin with a memory operand literal.
    If not present, it is the same as specifying the memory operand None.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];

    const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));

    SetValue(result_id, Deref(pointer));
}

void SPIRVSimulator::Op_Store(const Instruction& instruction){
    /*
    OpStore

    Store through a pointer.

    Pointer is the pointer to store through. Its type must be an OpTypePointer whose Type operand is the same as the type of Object.
    Object is the object to store.

    If present, any Memory Operands must begin with a memory operand literal.
    If not present, it is the same as specifying the memory operand None.
    */
    const PointerV& pointer = std::get<PointerV>(GetValue(instruction.words[1]));
    Deref(pointer) = GetValue(instruction.words[2]);
}

void SPIRVSimulator::Op_AccessChain(const Instruction& instruction){
    /*
    OpAccessChain

    Create a pointer into a composite object.

    Result Type must be an OpTypePointer. Its Type operand must be the type reached by walking the Base’s type
    hierarchy down to the last provided index in Indexes, and its Storage Class operand must be the same as the
    Storage Class of Base.
    If Result Type is an array-element pointer that is decorated with ArrayStride, its Array Stride must match the
    Array Stride of the array’s type. If the array’s type is not decorated with ArrayStride, Result Type also must not
    be decorated with ArrayStride.

    Base must be a pointer, pointing to the base of a composite object.

    Indexes walk the type hierarchy to the desired depth, potentially down to scalar granularity.
    The first index in Indexes selects the top-level member/element/component/column of the base composite.
    All composite constituents use zero-based numbering, as described by their OpType…​ instruction.
    The second index applies similarly to that result, and so on. Once any non-composite type is reached, there must be
    no remaining (unused) indexes.

    Each index in Indexes
    - must have a scalar integer type
    - is treated as signed
    - if indexing into a structure, must be an OpConstant whose value is in bounds for selecting a member
    - if indexing into a vector, array, or matrix, with the result type being a logical pointer type,
      causes undefined behavior if not in bounds.

    SIMULATOR SPECIFIC: We dont support types with more than 64 bits at present.
    */
    uint32_t result_id = instruction.words[2];
    uint32_t base_id = instruction.words[3];

    PointerV pointer = std::get<PointerV>(GetValue(base_id));
    for(auto i = 4; i < instruction.word_count; ++i){
        const Value& index_value = GetValue(instruction.words[i]);

        if (std::holds_alternative<uint64_t>(index_value)){
            pointer.idx_path.push_back((uint32_t)std::get<uint64_t>(index_value));
        } else if (std::holds_alternative<int64_t>(index_value)){
            pointer.idx_path.push_back((uint32_t)std::get<int64_t>(index_value));
        } else {
            throw std::runtime_error("SPIRV simulator: Index not of integer type in Op_AccessChain");
        }
    }

    SetValue(result_id, pointer);
}

void SPIRVSimulator::Op_Function(const Instruction&){
    /*
    OpFunction

    Add a function. This instruction must be immediately followed by one OpFunctionParameter instruction per each
    formal parameter of this function. This function’s body or declaration terminates with the next OpFunctionEnd instruction.

    Result Type must be the same as the Return Type declared in Function Type.

    Function Type is the result of an OpTypeFunction, which declares the types of the return value and parameters of the function.
    */
    // Nothing to do, we handle this when parsing instructions
}

void SPIRVSimulator::Op_FunctionEnd(const Instruction&){
    // This is a NOP in our design
}

void SPIRVSimulator::Op_FunctionCall(const Instruction& instruction){
    /*
    OpFunctionCall

    Call a function.

    Result Type is the type of the return value of the function.
    It must be the same as the Return Type operand of the Function Type operand of the Function operand.

    Function is an OpFunction instruction. This could be a forward reference.

    Argument N is the object to copy to parameter N of Function.

    Note: A forward call is possible because there is no missing type information: Result Type must match the Return
    Type of the function, and the calling argument types must match the formal parameter types.
    */
    uint32_t result_id = instruction.words[2];
    uint32_t function_id = instruction.words[3];

    FunctionInfo& function_info = funcs_[function_id];
    call_stack_.push_back({function_info.first_inst_index, result_id, {}, {}});

    for (auto i = 4; i < instruction.word_count; ++i){
        // Push parameters to the local scope
        call_stack_.back().locals[function_info.parameter_ids_[i]] = GetValue(instruction.words[i]);
    }
}

void SPIRVSimulator::Op_Label(const Instruction& instruction){
    /*
    OpLabel

    The label instruction of a block.

    References to a block are through the Result <id> of its label.
    */
    uint32_t result_id = instruction.words[1];
    prev_block_id_ = current_block_id_;
    current_block_id_ = result_id;
}

void SPIRVSimulator::Op_Branch(const Instruction& instruction){
    /*
    OpBranch

    Unconditional branch to Target Label.
    Target Label must be the Result <id> of an OpLabel instruction in the current function.
    This instruction must be the last instruction in a block.
    */
    // TODO: We should probably verify that the target instructions we are jumping to are labels
    call_stack_.back().pc = result_id_to_inst_index_.at(instruction.words[1]);
}

void SPIRVSimulator::Op_BranchConditional(const Instruction& instruction){
    /*
    OpBranchConditional

    If Condition is true, branch to True Label, otherwise branch to False Label.
    Condition must be a Boolean type scalar.

    True Label must be an OpLabel in the current function.
    False Label must be an OpLabel in the current function.
    Starting with version 1.6, True Label and False Label must not be the same <id>.
    Branch weights are unsigned 32-bit integer literals. There must be either no Branch Weights or exactly two branch weights. If present, the first is the weight for branching to True Label, and the second is the weight for branching to False Label. The implied probability that a branch is taken is its weight divided by the sum of the two Branch weights. At least one weight must be non-zero. A weight of zero does not imply a branch is dead or permit its removal; branch weights are only hints. The sum of the two weights must not overflow a 32-bit unsigned integer.

    This instruction must be the last instruction in a block.
    */
    uint64_t condition = std::get<uint64_t>(GetValue(instruction.words[1]));
    call_stack_.back().pc = result_id_to_inst_index_.at(condition ? instruction.words[2] : instruction.words[3]);
}

void SPIRVSimulator::Op_Return(const Instruction&){
    /*
    OpReturn

    Return with no value from a function with void return type.
    This instruction must be the last instruction in a block.
    */
    call_stack_.pop_back();
}

void SPIRVSimulator::Op_ReturnValue(const Instruction& instruction){
    /*
    OpReturnValue

    Return a value from a function.

    Value is the value returned, by copy, and must match the Return Type operand of the OpTypeFunction
    type of the OpFunction body this return instruction is in. Value must not have type OpTypeVoid.

    This instruction must be the last instruction in a block.
    */
    uint32_t value_id = instruction.words[1];
    uint32_t result_id = call_stack_.back().result_id;
    Value return_value = GetValue(value_id);

    call_stack_.pop_back();

    if (call_stack_.size()){
        SetValue(result_id, return_value);
    }
}

void SPIRVSimulator::Op_FAdd(const Instruction& instruction){
    /*
    OpFAdd

    Floating-point addition of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_FAdd");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_FAdd");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            double elem_result = std::get<double>(vec1->elems[i]) + std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of float type in Op_FAdd");
        }

        result = std::get<double>(op1) + std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_FAdd, must be vector or float");
    }
}


void SPIRVSimulator::Op_ExtInst(const Instruction& instruction){
    /*
    Execute an instruction in an imported set of extended instructions.

    Result Type is defined, per Instruction, in the external specification for Set.
    Set is the result of an OpExtInstImport instruction.
    Instruction is the enumerant of the instruction to execute within Set.
    It is an unsigned 32-bit integer. The semantics of the instruction are defined in the external specification for Set.

    Operand 1, …​ are the operands to the extended instruction.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t set_id = instruction.words[3];
    uint32_t instruction_literal = instruction.words[4];

    if (extended_imports_.find(set_id) == extended_imports_.end()){

    }

    std::string set_literal = extended_imports_[set_id];
    const std::span<const uint32_t> operand_words = std::span<const uint32_t>(instruction.words).subspan(5);
    if (set_literal == "GLSL"){
        GLSLExtHandler(type_id, result_id, instruction_literal, operand_words);
    } else {
        std::cout << std::setw(5) << "SPIRV simulator: Setting OpExtInst result value to default, this is wrong and we need to implement this" << std::endl;
        SetValue(result_id, MakeDefault(type_id));
    }
}


void SPIRVSimulator::Op_SelectionMerge(const Instruction&){
    // This is a NOP in our design
}

void SPIRVSimulator::Op_FMul(const Instruction& instruction){
    /*
    OpFMul

    Floating-point multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_FMul");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_Mul");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            double elem_result = std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        // TODO: Check that they are doubles
        result = std::get<double>(op1) * std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_Mul, must be vector or float");
    }
}

void SPIRVSimulator::Op_LoopMerge(const Instruction&){
    // This is a NOP in our design
    // TODO: Double check this
}

void SPIRVSimulator::Op_INotEqual(const Instruction& instruction){
    /*
    OpINotEqual

    Integer comparison for inequality.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.
    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_INotEqual");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_INotEqual");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t elem_result;

            // This should compare equal if different types but same number, so cant use variant operators here
            // TODO: Refactor this and the similar blocks below
            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) != std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            } else {
                throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Bool){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) != std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) != std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)(std::get<int64_t>(op1) != std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)(std::get<int64_t>(op1) != std::get<uint64_t>(op2));
        } else {
            throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual");
        }

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_IAdd, must be vector or bool");
    }
}

void SPIRVSimulator::Op_IAdd(const Instruction& instruction){
    /*
    OpIAdd

    Integer addition of Operand 1 and Operand 2.

    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same number of components
    as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component
    width and R is computed with enough precision to avoid overflow and underflow.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_IAdd");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_IAdd");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            } else {
                throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<uint64_t>(op1) + std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<uint64_t>(op1) + std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<int64_t>(op1) + std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<int64_t>(op1) + std::get<uint64_t>(op2));
        } else {
            throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd");
        }

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_IAdd, must be vector or int");
    }
}

void SPIRVSimulator::Op_LogicalNot(const Instruction& instruction){
    /*
    OpLogicalNot

    Result is true if Operand is false. Result is false if Operand is true.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand must be the same as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& operand = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        if(!std::holds_alternative<std::shared_ptr<VectorV>>(operand)){
            // TODO: Error
        }

        auto vec = std::get<std::shared_ptr<VectorV>>(operand);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if (std::holds_alternative<double>(vec->elems[i])){
                result_vec->elems.push_back((uint64_t)(!std::get<double>(vec->elems[i])));
            } else if (std::holds_alternative<uint64_t>(vec->elems[i])){
                result_vec->elems.push_back((uint64_t)!(std::get<uint64_t>(vec->elems[i])));
            } else if (std::holds_alternative<int64_t>(vec->elems[i])){
                result_vec->elems.push_back((uint64_t)!(std::get<int64_t>(vec->elems[i])));
            } else {
                throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_LogicalNot vector operand");
            }
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Bool){
        Value result;

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if (std::holds_alternative<double>(operand)){
                result = (uint64_t)(!std::get<double>(operand));
            } else if (std::holds_alternative<uint64_t>(operand)){
                result = (uint64_t)!(std::get<uint64_t>(operand));
            } else if (std::holds_alternative<int64_t>(operand)){
                result = (uint64_t)!(std::get<int64_t>(operand));
            } else {
                throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_LogicalNot");
            }
        }

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_LogicalNot, must be vector or bool");
    }
}

void SPIRVSimulator::Op_Capability(const Instruction&) {
    // This is a NOP in our design
}

void SPIRVSimulator::Op_Extension(const Instruction&) {
    // This is a NOP in our design (at least for now)
}

void SPIRVSimulator::Op_MemoryModel(const Instruction&) {
    // This is a NOP in our design
}

void SPIRVSimulator::Op_ExecutionMode(const Instruction&) {
    // We may will need this later
}

void SPIRVSimulator::Op_Source(const Instruction&) {
    // This is a NOP in our design
}

void SPIRVSimulator::Op_SourceExtension(const Instruction&) {
    // This is a NOP in our design
}

void SPIRVSimulator::Op_Name(const Instruction&) {
    // We could use this for debug info later, for now we leave it as a NOP
}

void SPIRVSimulator::Op_MemberName(const Instruction&) {
    // We could use this for debug info later, for now we leave it as a NOP
}

void SPIRVSimulator::Op_Decorate(const Instruction& instruction) {
    /*
    OpDecorate

    Add a Decoration to another <id>.

    Target is the <id> to decorate. It can potentially be any <id> that is a forward reference.
    A set of decorations can be grouped together by having multiple decoration instructions targeting the same
    OpDecorationGroup instruction.

    This instruction is only valid if the Decoration operand is a decoration that takes no Extra Operands, or takes
    Extra Operands that are not <id> operands.
    */

    uint32_t target_id = instruction.words[1];
    spv::Decoration kind = static_cast<spv::Decoration>(instruction.words[2]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 3; i < instruction.word_count; ++i){
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{kind, std::move(literals)};
    decorators_[target_id].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_MemberDecorate(const Instruction& instruction) {
    /*
    OpMemberDecorate

    Add a Decoration to a member of a structure type.
    Structure type is the <id> of a type from OpTypeStruct.
    Member is the number of the member to decorate in the type. The first member is member 0, the next is member 1, …​

    Note: See OpDecorate for creating groups of decorations for consumption by OpGroupMemberDecorate
    */
    uint32_t structure_type_id = instruction.words[1];
    uint32_t member_literal = instruction.words[2];
    spv::Decoration kind = static_cast<spv::Decoration>(instruction.words[3]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 4; i < instruction.word_count; ++i){
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{kind, std::move(literals)};
    struct_decorators_[structure_type_id][member_literal].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_ArrayLength(const Instruction& instruction){
    /*
    OpArrayLength

    Length of a run-time array.
    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Structure must be a logical pointer to an OpTypeStruct whose last member is a run-time array.
    Array member is an unsigned 32-bit integer index of the last member of the structure that Structure points to.
    That member’s type must be from OpTypeRuntimeArray.
    */
    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    //uint32_t structure_id = instruction.words[3];
    //uint32_t literal_array_member = instruction.words[4];

    // TODO: Must query metadata here to find the length
    throw std::runtime_error("SPIRV simulator: Op_ArrayLength is unimplemented! Fix this.");

    SetValue(result_id, 0);
}

void SPIRVSimulator::Op_SpecConstant(const Instruction& instruction) {
    /*
    OpSpecConstant

    Declare a new integer-type or floating-point-type scalar specialization constant.
    Result Type must be a scalar integer type or floating-point type.
    Value is the bit pattern for the default value of the constant. Types 32 bits wide or smaller take one word.
    Larger types take multiple words, with low-order words appearing first.
    This instruction can be specialized to become an OpConstant instruction.

    See Specialization.
    */
    // TODO: May have to consume API side info here
    Op_Constant(instruction);
}

void SPIRVSimulator::Op_SpecConstantOp(const Instruction& instruction) {
    /*
    OpSpecConstantOp

    Declare a new specialization constant that results from doing an operation.
    Result Type must be the type required by the Result Type of Opcode.

    Opcode is an unsigned 32-bit integer. It must equal one of the following opcodes.
    OpSConvert, OpUConvert (missing before version 1.4), OpFConvert
    OpSNegate, OpNot, OpIAdd, OpISub
    OpIMul, OpUDiv, OpSDiv, OpUMod, OpSRem, OpSMod
    OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical
    OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd
    OpVectorShuffle, OpCompositeExtract, OpCompositeInsert
    OpLogicalOr, OpLogicalAnd, OpLogicalNot,
    OpLogicalEqual, OpLogicalNotEqual
    OpSelect
    OpIEqual, OpINotEqual
    OpULessThan, OpSLessThan
    OpUGreaterThan, OpSGreaterThan
    OpULessThanEqual, OpSLessThanEqual
    OpUGreaterThanEqual, OpSGreaterThanEqual

    If the Shader capability was declared, OpQuantizeToF16 is also valid.

    If the Kernel capability was declared, the following opcodes are also valid:
    OpConvertFToS, OpConvertSToF
    OpConvertFToU, OpConvertUToF
    OpUConvert, OpConvertPtrToU, OpConvertUToPtr
    OpGenericCastToPtr, OpPtrCastToGeneric, OpBitcast
    OpFNegate, OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem, OpFMod
    OpAccessChain, OpInBoundsAccessChain
    OpPtrAccessChain, OpInBoundsPtrAccessChain

    Operands are the operands required by opcode, and satisfy the semantics of opcode.
    In addition, all Operands that are <id>s must be either:
    - the <id>s of other constant instructions, or
    - OpUndef, when allowed by opcode, or
    - for the AccessChain named opcodes, their Base is allowed to be a global (module scope) OpVariable instruction.

    See Specialization.
    */
    uint32_t result_id = instruction.words[2];

    // TODO: Double check this after thoroughly reading the spec.
    if (spec_instructions_.find(result_id) == spec_instructions_.end()){
        uint32_t type_id = instruction.words[1];
        uint32_t opcode = instruction.words[3];

        auto& spec_instr_words = spec_instr_words_[result_id];

        Instruction spec_instruction;
        spec_instruction.opcode = (spv::Op)opcode;
        spec_instruction.word_count = instruction.words.size() - 1;

        uint32_t header_word = (spec_instruction.word_count << kWordCountShift) | spec_instruction.opcode;
        spec_instr_words.push_back(header_word);
        spec_instr_words.push_back(type_id);
        spec_instr_words.push_back(result_id);

        for (uint32_t operand_index = 4; operand_index < instruction.word_count; ++operand_index){
            spec_instr_words.push_back(instruction.words[operand_index]);
        }

        spec_instruction.words = std::span<const uint32_t>{spec_instr_words.data(), spec_instr_words.size()};
        spec_instructions_[result_id] = spec_instruction;
    }

    if (verbose_){
        PrintInstruction(spec_instructions_[result_id]);
    }

    ExecuteInstruction(spec_instructions_[result_id]);
}

void SPIRVSimulator::Op_SpecConstantComposite(const Instruction& instruction) {
    /*
    OpSpecConstantComposite

    Declare a new composite specialization constant.
    Result Type must be a composite type, whose top-level members/elements/components/columns have the
    same type as the types of the Constituents. The ordering must be the same between the top-level types in Result Type and the Constituents.
    Constituents become members of a structure, or elements of an array, or components of a vector, or columns of a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result.
    The Constituents must appear in the order needed by the definition of the type of the result.
    The Constituents must be the <id> of other specialization constants, constant declarations, or an OpUndef.
    This instruction will be specialized to an OpConstantComposite instruction.

    See Specialization.
    */
    // TODO: May have to consume API side info here
    Op_ConstantComposite(instruction);
}

void SPIRVSimulator::Op_UGreaterThanEqual(const Instruction& instruction){
    /*
    OpUGreaterThanEqual

    Unsigned-integer comparison if Operand 1 is greater than or equal to Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands set to be vector type in Op_UGreaterThanEqual, but they are not, illegal input parameters");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec1->elems[i])){
                Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) >= std::get<uint64_t>(vec2->elems[i]));
                result_vec->elems.push_back(elem_result);
            } else {
                throw std::runtime_error("SPIRV simulator: Found non-unsigned integer operand in Op_UGreaterThanEqual vector operands");
            }
        }

        SetValue(result_id, result);

    } else if (type.kind == Type::Kind::Bool){
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) >= std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type in Op_UGreaterThanEqual: " + std::to_string((uint32_t)(type.kind)) + ", must be vector or bool");
    }
}

void SPIRVSimulator::Op_Phi(const Instruction& instruction){
    /*

    OpPhi

    The SSA phi function.
    The result is selected based on control flow: If control reached the current block from Parent i, Result Id gets
    the value that Variable i had at the end of Parent i.

    Result Type can be any type except OpTypeVoid.

    Operands are a sequence of pairs: (Variable 1, Parent 1 block), (Variable 2, Parent 2 block), …​
    Each Parent i block is the label of an immediate predecessor in the CFG of the current block.
    There must be exactly one Parent i for each parent block of the current block in the CFG.
    If Parent i is reachable in the CFG and Variable i is defined in a block, that defining block must dominate Parent i.
    All Variables must have a type matching Result Type.

    Within a block, this instruction must appear before all non-OpPhi instructions (except for OpLine and OpNoLine, which can
    be mixed with OpPhi).
    */

    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    for (uint32_t operand_index = 3; operand_index < instruction.word_count; operand_index += 2){
        uint32_t variable_id = instruction.words[operand_index];
        uint64_t block_id = instruction.words[operand_index + 1];

        if (block_id == prev_block_id_){
            SetValue(result_id, GetValue(variable_id));
            return;
        }
    }

    throw std::runtime_error("SPIRV simulator: Op_Phi faield to find a valid source block ID, something is broken in the control flow handling.");
}

void SPIRVSimulator::Op_ConvertUToF(const Instruction& instruction){
    /*
    OpConvertUToF

    Convert value numerically from unsigned integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Unsigned Value must be a scalar or vector of integer type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id = instruction.words[3];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);

        if(!std::holds_alternative<std::shared_ptr<VectorV>>(val_op)){
            throw std::runtime_error("SPIRV simulator: Operand set to be vector type in OpConvertUToF, but it is not, illegal input parameters");
        }

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        if (vec->elems.size() != type.vector.elem_count){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of valid length in OpConvertUToF");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if(!std::holds_alternative<uint64_t>(vec->elems[i])){
                throw std::runtime_error("SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF vector operands");
            }

            Value elem_result = (double)std::get<uint64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op = GetValue(value_id);

        if(!std::holds_alternative<uint64_t>(op)){
            throw std::runtime_error("SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF");
        }

        Value result = (double)std::get<uint64_t>(op);
        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid return type in OpConvertUToF: " + std::to_string((uint32_t)(type.kind)) + ", must be vector or float");
    }
}

void SPIRVSimulator::Op_ConvertSToF(const Instruction& instruction){
    /*
    OpConvertSToF

    Convert value numerically from signed integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Signed Value must be a scalar or vector of integer type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id = instruction.words[3];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);
        if(!std::holds_alternative<std::shared_ptr<VectorV>>(val_op)){
            throw std::runtime_error("SPIRV simulator: Operand set to be vector type in Op_ConvertSToF, but it is not, illegal input parameters");
        }

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        if (vec->elems.size() != type.vector.elem_count){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of valid length in Op_ConvertSToF");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if(!std::holds_alternative<int64_t>(vec->elems[i])){
                throw std::runtime_error("SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF vector operands");
            }

            Value elem_result = (double)std::get<int64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op = GetValue(value_id);

        if(!std::holds_alternative<int64_t>(op)){
            throw std::runtime_error("SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF");
        }

        Value result = (double)std::get<int64_t>(op);
        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type in Op_ConvertSToF: " + std::to_string((uint32_t)(type.kind)) + ", must be vector or float");
    }
}


void SPIRVSimulator::Op_FDiv(const Instruction& instruction){
    /*
    OpFDiv

    Floating-point division of Operand 1 divided by Operand 2.

    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands set to be vector type in Op_FDiv, but they are not, illegal input parameters");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of equal length in Op_FDiv");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(!(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]))){
                throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_FDiv vector operands");
            }

            elem_result = std::get<double>(vec1->elems[i]) / std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(!(std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2))){
            throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_FDiv");
        }

        result = std::get<double>(op1) + std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_FDiv, must be vector or float");
    }
}

void SPIRVSimulator::Op_FSub(const Instruction& instruction){
    /*
    OpFSub

    Floating-point subtraction of Operand 2 from Operand 1.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands set to be vector type in Op_FSub, but they are not, illegal input parameters");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of equal length in Op_FSub");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(!(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]))){
                throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_FSub vector operands");
            }

            elem_result = std::get<double>(vec1->elems[i]) - std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(!(std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2))){
            throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_FSub");
        }

        result = std::get<double>(op1) - std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_FSub, must be vector or float");
    }
}

void SPIRVSimulator::Op_VectorTimesScalar(const Instruction& instruction){
    /*
    OpVectorTimesScalar

    Scale a floating-point vector.
    Result Type must be a vector of floating-point type.
    The type of Vector must be the same as Result Type. Each component of Vector is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];
    uint32_t scalar_id = instruction.words[4];

    const Type& type = types_.at(type_id);

    Value result = std::make_shared<VectorV>();
    auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

    // TODO: Check this before calling get just in case (for clearer error messages)
    auto vec = std::get<std::shared_ptr<VectorV>>(GetValue(vector_id));
    double scalar_value = std::get<double>(GetValue(scalar_id));

    for (uint32_t i = 0; i < type.vector.elem_count; ++i){
        Value elem_result;

        if(!std::holds_alternative<double>(vec->elems[i])){
            throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_VectorTimesScalar vector operands");
        }

        elem_result = std::get<double>(vec->elems[i]) * scalar_value;

        result_vec->elems.push_back(elem_result);
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_SLessThan(const Instruction& instruction){
    /*
    OpSLessThan

    Signed-integer comparison if Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    // No explicit requirement for ints to be signed? Assume they have to be for now (but detect if they aint)
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands set to be vector type in Op_SLessThan, but they are not, illegal input parameters");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of equal length in Op_SLessThan");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(!(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))){
                throw std::runtime_error("SPIRV simulator: Found non-signed integer operand in Op_SLessThan vector operands");
            }

            elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) < std::get<int64_t>(vec2->elems[i]));

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Bool){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(!(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))){
            throw std::runtime_error("SPIRV simulator: Found non-signed integer operand in Op_SLessThan");
        }

        result = (uint64_t)(std::get<int64_t>(op1) < std::get<int64_t>(op2));

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_SLessThan, must be vector or bool");
    }
}

void SPIRVSimulator::Op_Dot(const Instruction& instruction){
    /*
    OpDot

    Dot product of Vector 1 and Vector 2.
    Result Type must be a floating-point type scalar.
    Vector 1 and Vector 2 must be vectors of the same type, and their component type must be Result Type.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Float){
        double result = 0.0;

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_Dot");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if (vec1->elems.size() != vec2->elems.size()){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_Dot");
        }

        for (uint32_t i = 0; i < vec1->elems.size(); ++i){
            if(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i])){
                result += std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
            } else {
                throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_Dot vector operands");
            }
        }

        Value val_result = result;
        SetValue(result_id, val_result);
    }  else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_Dot, must be float");
    }
}

void SPIRVSimulator::Op_FOrdGreaterThan(const Instruction& instruction){
    /*
    OpFOrdGreaterThan

    Floating-point comparison if operands are ordered and Operand 1 is greater than Operand 2.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands set to be vector type in Op_UGreaterThanEqual, but they are not, illegal input parameters");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            if(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i])){
                Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) > std::get<double>(vec2->elems[i]));
                result_vec->elems.push_back(elem_result);
            } else {
                throw std::runtime_error("SPIRV simulator: Found non-floating point operand in Op_FOrdGreaterThan vector operands");
            }
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Bool){
        Value result = (uint64_t)(std::get<double>(val_op1) > std::get<double>(val_op2));
        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_FOrdGreaterThan, must be vector or float");
    }
}

void SPIRVSimulator::Op_CompositeExtract(const Instruction& instruction){
    /*
    OpCompositeExtract

    Extract a part of a composite object.
    Result Type must be the type of object selected by the last provided index. The instruction result is the extracted object.
    Composite is the composite to extract from.

    Indexes walk the type hierarchy, potentially down to component granularity, to select the part to extract.
    All indexes must be in bounds. All composite constituents use zero-based numbering, as described by their OpType…​ instruction.
    Each index is an unsigned 32-bit integer.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t composite_id = instruction.words[3];

    Value* current_composite = &(GetValue(composite_id));
    for (uint32_t i = 4; i < instruction.word_count; ++i){
        uint32_t literal_index = instruction.words[i];

        if(std::holds_alternative<std::shared_ptr<AggregateV>>(*current_composite)){
            auto agg = std::get<std::shared_ptr<AggregateV>>(*current_composite);

            if(literal_index >= agg->elems.size()){
                throw std::runtime_error("SPIRV simulator: Aggregate index OOB");
            }

            current_composite = &agg->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<VectorV>>(*current_composite)){
            auto vec = std::get<std::shared_ptr<VectorV>>(*current_composite);

            if(literal_index >= vec->elems.size()){
                std::cout << vec->elems.size() << " : " << i << std::endl;
                throw std::runtime_error("SPIRV simulator: Vector index OOB");
            }

            current_composite = &vec->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*current_composite)){
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*current_composite);

            if(literal_index >= matrix->cols.size()){
                throw std::runtime_error("SPIRV simulator: Matrix index OOB");
            }

            current_composite = &matrix->cols[literal_index];
        }
        else{
            throw std::runtime_error("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    SetValue(result_id, *current_composite);
}

Type SPIRVSimulator::GetType(uint32_t result_id) const{
    if (result_id_to_inst_index_.find(result_id) == result_id_to_inst_index_.end()){
        throw std::runtime_error("SPIRV simulator: No instruction found for result_id");
    }

    size_t instruction_index = result_id_to_inst_index_.at(result_id);
    const Instruction& instruction = instructions_[instruction_index];

    bool has_result = false;
    bool has_type = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (has_type){
        uint32_t inst_type_id = instruction.words[1];
        if (types_.find(inst_type_id) == types_.end()){
            throw std::runtime_error("SPIRV simulator: No type found for type_id: " + std::to_string(inst_type_id));
        }
        return types_.at(inst_type_id);
    } else {
        Type void_type;
        void_type.kind = Type::Kind::Void;
        void_type.scalar = {
            0,
            false
        };
        return void_type;
    }
}

void SPIRVSimulator::Op_Bitcast(const Instruction& instruction){
    /*
    OpBitcast

    Bit pattern-preserving type conversion.

    Result Type must be an OpTypePointer, or a scalar or vector of numerical-type.

    Operand must have a type of OpTypePointer, or a scalar or vector of numerical-type. It must be a different type than Result Type.

    Before version 1.5: If either Result Type or Operand is a pointer, the other must be a pointer or an integer scalar.
    Starting with version 1.5: If either Result Type or Operand is a pointer, the other must be a pointer, an integer scalar, or an integer vector.

    If both Result Type and the type of Operand are pointers, they both must point into same storage class.

    Behavior is undefined if the storage class of Result Type does not match the one used by the operation that produced the value of Operand.

    If Result Type has the same number of components as Operand, they must also have the same component width, and results are computed per component.

    If Result Type has a different number of components than Operand, the total number of bits in Result Type must equal the total number of
    bits in Operand.

    Let L be the type, either Result Type or Operand’s type, that has the larger number of components. Let S be the other type,
    with the smaller number of components. The number of components in L must be an integer multiple of the number of components in S.
    The first component (that is, the only or lowest-numbered component) of S maps to the first components of L, and so on, up to the last
    component of S mapping to the last components of L. Within this mapping, any single component of S (mapping to multiple components of L) maps
    its lower-ordered bits to the lower-numbered components of L.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Value& operand = GetValue(operand_id);
    Type operand_type = GetType(operand_id);

    const Type& type = types_.at(type_id);

    //
    // First, we extract all the data from the operands into a vector
    //
    std::vector<std::byte> bytes;
    if (std::holds_alternative<std::shared_ptr<VectorV>>(operand)){
        const Type& elem_type = types_.at(operand_type.vector.elem_type_id);
        std::shared_ptr<VectorV> vec = std::get<std::shared_ptr<VectorV>>(operand);
        for (const Value& element : vec->elems){
            if (std::holds_alternative<double>(element)){
                double value = std::get<double>(element);
                extract_bytes<double>(bytes, value, elem_type.scalar.width);
            } else if (std::holds_alternative<uint64_t>(element)){
                uint64_t value = std::get<uint64_t>(element);
                extract_bytes<uint64_t>(bytes, value, elem_type.scalar.width);
            } else if (std::holds_alternative<int64_t>(element)){
                int64_t value = std::get<int64_t>(element);
                extract_bytes<int64_t>(bytes, value, elem_type.scalar.width);
            } else {
                throw std::runtime_error("SPIRV simulator: invalid operand element type in Op_Bitcast, must be numeric");
            }
        }
    } else if (std::holds_alternative<double>(operand)){
        double value = std::get<double>(operand);
        extract_bytes<double>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<uint64_t>(operand)){
        uint64_t value = std::get<uint64_t>(operand);
        extract_bytes<uint64_t>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<int64_t>(operand)){
        int64_t value = std::get<int64_t>(operand);
        extract_bytes<int64_t>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<PointerV>(operand)){
        // Take the easy out if its just pointer to pointer conversion
        if (type.kind == Type::Kind::Pointer){
            SetValue(result_id, operand);
            return;
        }
        // We currently dont handle this, we could do it by storing the pointer in a
        // special container and storing a index into that container in the result here
        throw std::runtime_error("SPIRV simulator: Pointer to non-pointer Op_Bitcast detected, must add support for this!");
    } else {
        throw std::runtime_error("SPIRV simulator: invalid operand type in Op_Bitcast, must be vector or numeric");
    }

    //
    // Then we map this memory to the result value
    //
    Value result;
    if (type.kind == Type::Kind::Vector){
        const Type& elem_type = types_.at(type.vector.elem_type_id);
        uint32_t elem_size_bytes = elem_type.scalar.width / 8;
        std::shared_ptr<VectorV> vec = std::get<std::shared_ptr<VectorV>>(result);
        uint32_t current_byte = 0;

        for (Value& element : vec->elems){
            if (std::holds_alternative<double>(element)){
                double value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else if (std::holds_alternative<uint64_t>(element)){
                uint64_t value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else if (std::holds_alternative<int64_t>(element)){
                int64_t value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else {
                throw std::runtime_error("SPIRV simulator: invalid result element type in Op_Bitcast, must be numeric");
            }

            current_byte += elem_size_bytes;
        }
    } else if (type.kind == Type::Kind::Float){
        double value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if ((type.kind == Type::Kind::Int) && !type.scalar.is_signed){
        uint64_t value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if ((type.kind == Type::Kind::Int) && type.scalar.is_signed){
        int64_t value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if (type.kind == Type::Kind::Pointer) {
        // This is one of the main cases we want to detect, a non-pointer type is cast to a pointer
        // Create a new pointer to the base type, and add a mapping to the real memory (referenced by the true value held by the operands)
        // If the real memory is provided, read and initialize the target value on our simulated heap

        Value pointee_init = MakeDefault(type.pointer.pointee_type_id);

        // TODO: Set value of pointee_init if the source pointer is available

        if(type.pointer.storage_class == (uint32_t)spv::StorageClass::StorageClassFunction){
            call_stack_.back().func_heap[result_id] = pointee_init;
        }
        else{
            Heap(type.pointer.storage_class)[result_id] = pointee_init;
        }


        // TODO: Deal with pointers into compound types here (set the indirection path etc.)

        PointerV new_pointer{result_id, type.pointer.storage_class, {}};
        SetValue(result_id, new_pointer);

        void* pointer_value;
        std::memcpy(&pointer_value, bytes.data(), sizeof(void*));
        result_id_to_external_pointer_[result_id] = pointer_value;

        if (verbose_){
            std::cout << "SPIRV simulator: Found pointer with address: 0x" << std::hex  << pointer_value << std::dec << std::endl;
        }

        return;
    } else {
        throw std::runtime_error("SPIRV simulator: invalid result type in Op_Bitcast, must be vector, pointer or numeric, was: " + std::to_string((uint32_t)type.kind));
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_IMul(const Instruction& instruction){
    /*
    OpIMul

    Integer multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

    Results are computed per component.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        if(!(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2))){
            throw std::runtime_error("SPIRV simulator: Operands not of vector type in Op_IMul");
        }

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        if ((vec1->elems.size() != vec2->elems.size()) || (vec1->elems.size() != type.vector.elem_count)){
            throw std::runtime_error("SPIRV simulator: Operands not of equal/correct length in Op_IMul");
        }

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) * std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) * std::get<int64_t>(vec2->elems[i]));
            } else {
                throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_IMul vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<uint64_t>(op1) * std::get<uint64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<int64_t>(op1) * std::get<int64_t>(op2));
        } else {
            throw std::runtime_error("SPIRV simulator: Could not find valid parameter type combination for Op_IMul");
        }

        SetValue(result_id, result);
    } else {
        throw std::runtime_error("SPIRV simulator: Invalid result type int Op_IMul, must be vector or integer type");
    }
}
