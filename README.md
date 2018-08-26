# SC_Loop
Implementation of self-consistent DMFT condition

Using the ALPSCore framework mostly for input data format and access.

Using Eigen for linear algebra operations

Install instructions:

Define the environment variable EIGEN3_INCLUDE_DIR containing the forlder where Eigen is to be found.

mkdir build && cd build

cmake ../src

* On salomon, and other clusters, in case the MPI compiler doesn't match the C++ compiler:

CXX=/apps/all/icc/2017.1.132-GCC-6.3.0-2.27/compilers_and_libraries_2017.1.132/linux/bin/intel64/icc cmake ../src

make

* On VSC, ALPSCORE compilation:

```module load intel/17  hdf5/1.8.18-SERIAL intel-mpi/2017 python/2.7 boost/1.62.0 intel-mkl/2017  cmake/3.9.6```

```cmake -DMPI_CXX_COMPILER=/cm/shared/apps/intel/compilers_and_libraries_2017.7.259/linux/mpi/intel64/bin/mpiicpc -DEIGEN3_INCLUDE_DIR=/home/lv70946/geffroy/Code/Eigen_334 -DCMAKE_INSTALL_PREFIX=../Install ../.```
