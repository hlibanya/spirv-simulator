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

    InputData inputs;
    SPIRVSimulator em(ReadFile(argv[1]), inputs, true);
    em.Run();

    const PhysicalAddressData& physical_address_data = em.GetPhysicalAddressData();

    std::cout << "Found physical addresses:" << std::endl;
    for (auto pointer : physical_address_data.physical_address_buffer_pointers){
        std::cout << std::right << std::setw(15) << "Raw address: " << pointer.raw_pointer << std::endl;
        for (auto offset : pointer.idx_path){
            std::cout << std::setw(25) << "Object offset: " << offset << std::endl;
        }
    }

    std::cout << "Found pointers to physical address pointers:" << std::endl;
    for (auto pointer : physical_address_data.pointers_to_physical_address_buffer_pointers){
        std::cout << std::right << std::setw(15) << "Raw address: " << pointer.raw_pointer << std::endl;
        for (auto offset : pointer.idx_path){
            std::cout << std::setw(25) << "Object offset: " << offset << std::endl;
        }
    }


    return 0;
}
