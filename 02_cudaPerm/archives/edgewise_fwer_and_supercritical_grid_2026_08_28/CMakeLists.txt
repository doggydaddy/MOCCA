cmake_minimum_required(VERSION 3.9)
project(archived_edgewise_fwer_cudaPerm LANGUAGES CUDA C CXX)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(CUDA REQUIRED)

# ccmat_io.* is not duplicated here -- it remains the single active copy in
# the parent (02_cudaPerm/) directory, shared with permutationTest_cuda_bundle.
add_executable(permutationTest_cuda_fwer
    permutationTest_cuda_fwer.cu
    perm_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ccmat_io.c
    results_io.c)
target_include_directories(permutationTest_cuda_fwer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_options(permutationTest_cuda_fwer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89 -Xcompiler=-fopenmp>)
target_link_libraries(permutationTest_cuda_fwer PRIVATE gomp)
