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

if(rapids-cmake_ADDED)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake_SOURCE_DIR}/rapids-cmake")
  include(rapids-cmake)
  rapids_cmake_build_type("Release")
endif()

# For now, disable CMake's automatic module scanning for C++ files
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Create executable
add_executable(cubvscudf main.cu)

set_target_properties(cubvscudf PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Link with cuDF
target_link_libraries(cubvscudf PRIVATE cudf::cudf)

# Compiler flags
target_compile_options(cubvscudf
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
)
