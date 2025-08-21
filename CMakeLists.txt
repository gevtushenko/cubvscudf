cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")

# Set CUDA as the language for the project  
project(cubvscudf LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CPM Configuration
include(cmake/CPM.cmake)

# cuDF repository configuration
set(CUDF_REPOSITORY "rapidsai/cudf" CACHE STRING "GitHub repository to fetch cuDF from")
set(CUDF_TAG "branch-25.08" CACHE STRING "Git tag/branch to fetch from cuDF repository")

# Try to find cuDF first, fetch if not found
CPMFindPackage(
  NAME cudf
  FIND_PACKAGE_ARGUMENTS "PATHS ${cudf_ROOT} ${cudf_ROOT}/latest"
  GIT_REPOSITORY https://github.com/${CUDF_REPOSITORY}
  GIT_TAG ${CUDF_TAG}
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR cpp
)

# Fetch rapids-cmake for build configuration
CPMAddPackage(
  NAME rapids-cmake
  GITHUB_REPOSITORY rapidsai/rapids-cmake
  GIT_TAG ${CUDF_TAG}
  DOWNLOAD_ONLY YES
)

# Add nvbench
CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG main
  GIT_SHALLOW TRUE
)

# Add CCCL (under different name to avoid conflicts with cuDF's CCCL)
CPMAddPackage(
  NAME cccl_cpp
  GITHUB_REPOSITORY gevtushenko/cccl
  GIT_TAG cccl.cpp
  GIT_SHALLOW TRUE
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/cccl_cpp-src
)

if(rapids-cmake_ADDED)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake_SOURCE_DIR}/rapids-cmake")
  include(rapids-cmake)
  rapids_cmake_build_type("Release")
endif()

# For now, disable CMake's automatic module scanning for C++ files
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Create cuDF benchmark executable
add_executable(bench_cudf bench_cudf.cu)

set_target_properties(bench_cudf PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Link with cuDF and nvbench
target_link_libraries(bench_cudf PRIVATE cudf::cudf nvbench::nvbench)

# Compiler flags for cuDF benchmark
target_compile_options(bench_cudf
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
)

# Create CCCL benchmark executable with wrapper
add_executable(bench_cccl bench_cccl.cu cccl_c_wrapper.cu)

set_target_properties(bench_cccl PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Link with our custom CCCL C API, and nvbench
target_include_directories(bench_cccl PRIVATE 
  ${cccl_cpp_SOURCE_DIR}/c/parallel/include
  ${cccl_cpp_SOURCE_DIR}/thrust 
  ${cccl_cpp_SOURCE_DIR}/cub 
  ${cccl_cpp_SOURCE_DIR}/libcudacxx/include
)

# For now, we'll skip building the CCCL C library due to conflicts
# Just use the headers and link against required CUDA libraries
target_compile_definitions(bench_cccl PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CCCL_C_EXPERIMENTAL)
target_link_libraries(bench_cccl PRIVATE nvbench::nvbench cuda cudart nvrtc nvJitLink)

# Compiler flags for CCCL benchmark
target_compile_options(bench_cccl
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
)
