cmake_minimum_required(VERSION 3.9)
project(cudaCC LANGUAGES CUDA C)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 75 = Turing (RTX 2000), 86 = Ampere (RTX 3000), 89 = Ada (RTX 4000)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

include_directories(nifticlib-2.0.0/include)
link_directories(nifticlib-2.0.0/lib)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(CUDA REQUIRED)

add_executable(calcCC_cuda calcCC_cuda.cu)
#set_target_properties(calcCC_cuda PROPERTIES CUDA_ARCHITECTURES "50;72")
target_compile_options(calcCC_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50 -Xcompiler=-fopenmp>)
target_link_libraries(calcCC_cuda PRIVATE gomp fslio niftiio znz m z)

add_executable(cudaCC cudaCC.cu)
target_link_libraries(cudaCC fslio niftiio znz m z)

add_executable(calcCC calcCC.c)
target_compile_options(calcCC PRIVATE --compile -fopenmp)
target_link_libraries(calcCC gomp fslio niftiio znz m z)

add_executable(calcCC_div calcCC_div.c)
target_compile_options(calcCC_div PRIVATE --compile -fopenmp)
target_link_libraries(calcCC_div gomp fslio niftiio znz m z)

# copile cudaCC_div
add_executable(cudaCC_div cudaCC_div.cu)
target_compile_options(cudaCC_div PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(cudaCC_div gomp fslio niftiio znz m z)