# spirv-simulator
Requires a C++ 20 compatible compiler.
G++ 10 or newer and Clang 10 or newer should do the trick.

To build:

cmake -H. -Bbuild;
cd build;
make

To run a test:

./spirv_simulator ../test_shaders/<shader_name>.spirv