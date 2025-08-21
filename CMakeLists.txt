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

find_package(CUDAToolkit REQUIRED)

# Use ExternalProject to build CCCL C library separately to avoid conflicts
include(ExternalProject)

# Set CCCL repository and tag
set(CCCL_REPOSITORY "gevtushenko/cccl" CACHE STRING "GitHub repository for CCCL")
set(CCCL_TAG "cccl.cpp" CACHE STRING "Git tag for CCCL")

ExternalProject_Add(
    cccl_custom
    GIT_REPOSITORY "https://github.com/${CCCL_REPOSITORY}.git"
    GIT_TAG ${CCCL_TAG}
    SOURCE_DIR "${CMAKE_BINARY_DIR}/cccl-custom-src"
    BINARY_DIR "${CMAKE_BINARY_DIR}/cccl-custom-build"
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCCCL_ENABLE_C=ON
        -DCCCL_ENABLE_TESTING=OFF
        -DCCCL_ENABLE_BENCHMARKS=OFF
        -DCCCL_ENABLE_EXAMPLES=OFF
        -DCUB_ENABLE_TESTING=OFF
        -DCUB_ENABLE_EXAMPLES=OFF
        -DCUB_ENABLE_HEADER_TESTING=OFF
        -DTHRUST_ENABLE_TESTING=OFF
        -DTHRUST_ENABLE_EXAMPLES=OFF
        -DTHRUST_ENABLE_HEADER_TESTING=OFF
        -DCCCL_ENABLE_C2H=OFF
        -DCCCL_ENABLE_COMPILER_INTERFACE=OFF
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

# Get the source and binary directories for CCCL
ExternalProject_Get_Property(cccl_custom SOURCE_DIR BINARY_DIR)
set(CCCL_SOURCE_DIR ${SOURCE_DIR})
set(CCCL_BINARY_DIR ${BINARY_DIR})

# Create a placeholder library file
file(WRITE ${CMAKE_BINARY_DIR}/libcccl.c.parallel.so "placeholder")

# Create a custom target to build CCCL library
add_custom_target(cccl_lib ALL
    COMMAND ${CMAKE_COMMAND} --build ${CCCL_BINARY_DIR} --target cccl.c.parallel || echo "Build failed, but library may exist"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CCCL_BINARY_DIR}/lib/libcccl.c.parallel.so"
        "${CMAKE_BINARY_DIR}/libcccl.c.parallel.so"
    COMMAND ${CMAKE_COMMAND} -E echo "Library copied successfully"
    DEPENDS cccl_custom
    COMMENT "Building CCCL library"
)

# Create an interface library for CCCL
add_library(cccl.c.parallel INTERFACE)
target_include_directories(cccl.c.parallel INTERFACE ${CCCL_SOURCE_DIR}/c/parallel/include)
target_compile_definitions(cccl.c.parallel INTERFACE CCCL_C_EXPERIMENTAL CCCL_INCLUDE_PATH="${CCCL_SOURCE_DIR}")
target_link_libraries(cccl.c.parallel INTERFACE ${CMAKE_BINARY_DIR}/libcccl.c.parallel.so CUDA::nvrtc CUDA::cudart)
add_dependencies(cccl.c.parallel cccl_lib)

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

# Create CCCL benchmark executable using CCCL C parallel library
add_executable(bench_cccl bench_cccl.cu)

set_target_properties(bench_cccl PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Link with CCCL C parallel library and nvbench
target_compile_definitions(bench_cccl PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(bench_cccl PRIVATE nvbench::nvbench cccl.c.parallel)

# Compiler flags for CCCL benchmark
target_compile_options(bench_cccl
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
)
