cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

# enable C and C++ compilers
# enable_language(C CXX)

project(sc_loop_alpscore CXX)

INCLUDE_DIRECTORIES ("$ENV{EIGEN3_INCLUDE_DIR}")

# find ALPS Library
#find_package(ALPS REQUIRED PATHS ${ALPS_ROOT_DIR} $ENV{ALPS_HOME} NO_SYSTEM_ENVIRONMENT_PATH)
#message(STATUS "Found ALPS: ${ALPS_ROOT_DIR} (revision: ${ALPS_VERSION})")
#include(${ALPS_USE_FILE})
find_package(ALPSCore REQUIRED)
message(STATUS "Found ALPSCore")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


# Manage MPI Compilation
#find_package(MPI)
find_package(Boost REQUIRED system thread timer chrono)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# Eigen is able to use OpenMP. Not very useful for the time being though.
find_package(OpenMP)
if (OPENMP_FOUND)
  message("found OPENMP")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# rule for generating 'main' program
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug or Release
set(CMAKE_BUILD_TYPE Release)


add_executable(${PROJECT_NAME} main.cpp band_structure.cpp self_energy.cpp
  dmft_model.cpp hybridization_function.cpp tail_manager.cpp
  greens_function.cpp legendre.cpp)

message("Current system: " ${CMAKE_HOST_SYSTEM} )
message ("Compiler Id: " ${CMAKE_CXX_COMPILER_ID})

#if (CMAKE_COMPILER_IS_GNUCC)
#if(${CMAKE_HOST_SYSTEM} STREQUAL "Linux-3.2.0-0.bpo.4-amd64" )
  # This is tested
#  target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES}
    #${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})
#    ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
#else()
set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS -Wno-deprecated)
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${Boost_LIBRARIES})
#    ${MPI_CXX_LIBRARIES}  ${GSL_LIBRARIES} ${Boost_LIBRARIES})
#endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME od_dmft_ALPS_Core)
