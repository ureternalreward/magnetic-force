cmake_minimum_required(VERSION 3.8)
project(magnetic_force LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_STANDARD 11)

find_package(OpenMP)
include_directories(fmmtl_minimum glm)
add_executable(example magnetic_solver/main.cu magnetic_solver/magnetic_force.cu)

