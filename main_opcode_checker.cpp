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
        std::cerr << "Usage: " << argv[0] << " <list_of_shader_paths>.txt\n";
        return 1;
    }

    const std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    std::string line;
    std::set<std::string> unsupported_instructions;
    while (std::getline(file, line)) {
        SPIRVSimulator::InputData inputs;
        SPIRVSimulator::SPIRVSimulator sim(ReadFile(line), inputs);
        unsupported_instructions.insert(sim.unsupported_opcodes.begin(), sim.unsupported_opcodes.end());
    }

    for (const auto& opc : unsupported_instructions){
        std::cout << opc << std::endl;
    }

    file.close();

    return 0;
}
