#include "spirv_simulator.hpp"

static std::vector<uint32_t> ReadFile(const std::string &path){
    std::ifstream infile(path, std::ios::binary | std::ios::ate);

    if(!infile){
        throw std::runtime_error("Cannot open " + path);
    }

    auto size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<uint32_t> buf(static_cast<size_t>(size) / 4);

    infile.read(reinterpret_cast<char*>(buf.data()), size);
    if (infile.fail()){
        throw std::runtime_error("Read error");
    }

    return buf;
}


int main(int argc, char** argv){
    if(argc!=2){
        std::cerr << "Usage: " << argv[0] << " <shader.spv>\n";
        return 1;
    }

    SPIRVSimulator::InputData inputs;
    SPIRVSimulator::SPIRVSimulator sim(ReadFile(argv[1]), inputs, true);
    sim.Run();

    auto physical_address_data = sim.GetPhysicalAddressData();

    std::cout << "Pointers to pbuffers:" << std::endl;
    for (const auto& pointer_t : physical_address_data){
        std::cout << "  Found pointer with address: 0x" << std::hex << pointer_t.raw_pointer_value << std::dec << " made from input bit components:" << std::endl;
        for (auto bit_component : pointer_t.bit_components){
            if (bit_component.location == SPIRVSimulator::BitLocation::Constant) {
                std::cout << "    " << "From Constant in SPIRV input words, at Byte Offset: " << bit_component.byte_offset << std::endl;
            } else {
                if (bit_component.location == SPIRVSimulator::BitLocation::SpecConstant){
                    std::cout << "    " << "From SpecId: " << bit_component.binding_id;
                } else {
                    std::cout << "    " << "From DescriptorSetID: " << bit_component.set_id << ", Binding: " << bit_component.binding_id;
                }

                if (bit_component.location == SPIRVSimulator::BitLocation::StorageClass){
                    std::cout << ", in StorageClass: " << spv::StorageClassToString(bit_component.storage_class);
                }
                std::cout << ", Byte Offset: " << bit_component.byte_offset << ", Bitsize: " << bit_component.bitcount << ", to val Bit Offset: " << bit_component.val_bit_offset << std::endl;
            }
        }
    }

    return 0;
}
