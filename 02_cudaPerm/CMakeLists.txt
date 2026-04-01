cmake_minimum_required(VERSION 3.9)
project(cudaPerm LANGUAGES CUDA C)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(CUDA REQUIRED)

add_executable(permutationTest_cuda permutationTest_cuda.cu)
target_compile_options(permutationTest_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89 -Xcompiler=-fopenmp>)
target_link_libraries(permutationTest_cuda PRIVATE gomp)

add_executable(permutationTest_omp permutationTest_omp.c)
target_link_libraries(permutationTest_omp PRIVATE gomp)
