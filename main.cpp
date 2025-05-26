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

    SPIRVSimulator em(ReadFile(argv[1]), true);
    em.Run();

    return 0;
}
