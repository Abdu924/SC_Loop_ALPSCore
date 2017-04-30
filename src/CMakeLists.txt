cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# enable C and C++ compilers
# enable_language(C CXX)

project(sc_loop_alpscore CXX)

INCLUDE_DIRECTORIES ("$ENV{EIGEN3_INCLUDE_DIR}")

find_package(ALPSCore REQUIRED)
message(STATUS "Found ALPSCore")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

# Manage MPI Compilation
#find_package(MPI)
find_package(Boost REQUIRED system thread timer chrono)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# Debug or Release
set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp band_structure.cpp self_energy.cpp
  dmft_model.cpp hybridization_function.cpp tail_manager.cpp greens_function.cpp)

message("Current system: " ${CMAKE_HOST_SYSTEM} )
message ("CXX Compiler Id: " ${CMAKE_CXX_COMPILER_ID})

set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS -Wno-deprecated)
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${Boost_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME od_dmft_ALPS_Core)
