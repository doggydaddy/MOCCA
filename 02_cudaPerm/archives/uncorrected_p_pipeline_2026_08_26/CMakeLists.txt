cmake_minimum_required(VERSION 3.18)
project(archived_uncorrected_cudaPerm LANGUAGES CUDA C CXX)

find_package(OpenMP REQUIRED)

add_executable(permutationTest_cuda permutationTest_cuda.cu)
target_compile_options(
    permutationTest_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89 -Xcompiler=-fopenmp>
)
target_link_libraries(permutationTest_cuda PRIVATE OpenMP::OpenMP_CXX)

add_executable(permutationTest_omp permutationTest_omp.c)
target_link_libraries(permutationTest_omp PRIVATE OpenMP::OpenMP_C m)
