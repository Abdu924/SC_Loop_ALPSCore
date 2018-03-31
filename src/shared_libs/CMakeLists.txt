INCLUDE_DIRECTORIES ("$ENV{EIGEN3_INCLUDE_DIR}")

project(base_libs)

find_package(ALPSCore REQUIRED)
message(STATUS "Found ALPSCore")

include_directories(
  SYSTEM
  ${MPI_CXX_INCLUDE_PATH}
  $ENV{EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED system thread timer chrono filesystem)


add_library(gf_base gf_base.cpp band_structure.cpp dmft_model.cpp self_energy.cpp tail_manager.cpp greens_function.cpp legendre.cpp)
target_link_libraries(gf_base ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
