cmake_minimum_required(VERSION 3.8)
project(magnetic_force LANGUAGES CXX CUDA)

find_package(OpenMP)
include_directories(fmmtl_minimum glm)
add_executable(example magnetic_solver/main.cu magnetic_solver/magnetic_force.cu)

