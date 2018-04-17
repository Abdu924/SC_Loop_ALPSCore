INCLUDE_DIRECTORIES ("$ENV{EIGEN3_INCLUDE_DIR}")

project(base_libs)

include_directories(
  SYSTEM
  ${MPI_CXX_INCLUDE_PATH}
  $ENV{EIGEN3_INCLUDE_DIR})

add_library(gf_base gf_base.cpp band_structure.cpp dmft_model.cpp self_energy.cpp tail_manager.cpp
  greens_function.cpp legendre.cpp cmd_line.cpp flavor_transformer.cpp)
target_link_libraries(gf_base ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
