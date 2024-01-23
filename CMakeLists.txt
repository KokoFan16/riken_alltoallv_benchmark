cmake_minimum_required (VERSION 2.8)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project (RIKENAAV)
set (CMAKE_CXX_STANDARD 11)

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


add_executable(benchmark ./benchmarks.cpp)
add_executable(sortComp ./sortComp.cpp)


target_link_libraries(benchmark ${MPI_C_LIBRARIES})
target_link_libraries(sortComp ${MPI_C_LIBRARIES})



