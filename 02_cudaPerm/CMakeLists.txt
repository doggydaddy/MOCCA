cmake_minimum_required(VERSION 3.9)
project(cudaPerm LANGUAGES CUDA C CXX)
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

add_executable(permutationTest_cuda_fwer
    permutationTest_cuda_fwer.cu
    perm_kernels.cu
    ccmat_io.c
    results_io.c)
target_include_directories(permutationTest_cuda_fwer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(permutationTest_cuda_fwer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89 -Xcompiler=-fopenmp>)
target_link_libraries(permutationTest_cuda_fwer PRIVATE gomp)

# Separate sparse backend for bundle-level permutation FWER.  This does not
# alter either established permutation executable.
add_executable(permutationTest_cuda_bundle
    permutationTest_cuda_bundle.cu
    ccmat_io.c)
target_include_directories(permutationTest_cuda_bundle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(permutationTest_cuda_bundle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89 -Xcompiler=-fopenmp>)
target_link_libraries(permutationTest_cuda_bundle PRIVATE gomp)

# Optimized deterministic bundle formation for sparse permutation outputs.
# Python remains the reference implementation and is not modified/replaced.
add_executable(bundle_fwer_omp bundle_fwer_omp.cpp)
target_compile_features(bundle_fwer_omp PRIVATE cxx_std_17)
target_compile_options(bundle_fwer_omp PRIVATE -O3 -fopenmp)
target_link_libraries(bundle_fwer_omp PRIVATE gomp)
