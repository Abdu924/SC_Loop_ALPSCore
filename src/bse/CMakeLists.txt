cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    message("Using default build type: Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

project(bseq_solver C CXX)

# Disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

INCLUDE_DIRECTORIES ("$ENV{EIGEN3_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS -DNDEBUG -std=gnu++11")

message("CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message("Current system: " ${CMAKE_HOST_SYSTEM})
message ("CXX Compiler Id: " ${CMAKE_CXX_COMPILER_ID})

include_directories(
  SYSTEM
  ${MPI_CXX_INCLUDE_PATH}
  $ENV{EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp bubble.cpp)

set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS -Wno-deprecated)
target_link_libraries(${PROJECT_NAME} gf_base ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME od_bseq_solver)
