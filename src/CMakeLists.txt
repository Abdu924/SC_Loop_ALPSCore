cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(globalProject)

find_package(ALPSCore REQUIRED)
message(STATUS "Found ALPSCore")

find_package(Boost REQUIRED system thread timer chrono filesystem)

add_subdirectory(shared_libs)
add_subdirectory(sc_loop)
#add_subdirectory(projectB)
