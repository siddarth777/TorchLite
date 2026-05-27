cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
project(TorchLite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set CUDA compiler if not found
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_C 
    src/bindings.cpp 
    src/tensor.cpp 
    src/ops.cu
)

find_package(CUDAToolkit REQUIRED)

set_target_properties(_C PROPERTIES CUDA_ARCHITECTURES "native")
target_compile_features(_C PUBLIC cxx_std_17)
target_include_directories(_C PRIVATE src)
target_link_libraries(_C PRIVATE CUDA::cudart)
