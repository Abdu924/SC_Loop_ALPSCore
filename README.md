# SC_Loop
Implementation of self-consistent DMFT condition

Using the ALPASCore framework mostly for input data format and access.

Using Eigen for linear algebra operations

Install instructions:

Define the environment variable EIGEN3_INCLUDE_DIR containing the forlder where Eigen is to be found.

cmake ../src

for salomon:

CXX=/apps/all/icc/2017.1.132-GCC-6.3.0-2.27/compilers_and_libraries_2017.1.132/linux/bin/intel64/icc cmake ../src

make
