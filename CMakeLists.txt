cmake_minimum_required(VERSION 2.8.10)

project (Coffee)
enable_language(Fortran CXX)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
message ("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

option(build_tests "Build the test suite." ON)
option(use_cuda "Build the GPU part of the code" ON)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW) # CMake 3.12
endif ()
# cmake_policy(SET CMP0074 NEW)
# set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -g -DNDEBUG -xHost -no-prec-div ${CXX_EXTRA_FLAGS}")
else()
  # set(CXX_EXTRA_FLAGS "-Wno-misleading-indentation -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -g -DNDEBUG -mcpu=native ${CXX_EXTRA_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g3 -O0 ${CXX_EXTRA_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# set(FMT_INCLUDE_DIR deps/fmt)
set(CATCH_INCLUDE_DIR deps/catch/single_include/catch2)
set(CPPTOML_INCLUDE_DIR deps/cpptoml/include)

include_directories(${MPI_INCLUDE_PATH} ${CPPTOML_INCLUDE_DIR} ${CATCH_INCLUDE_DIR}
  ${HIGHFIVE_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} src)

######### Check for cuda support
if (use_cuda)
  find_package(CUDA REQUIRED)
  if (CUDA_FOUND)
    message("Enabling Cuda")
    set(CUDA_NVCC_FLAGS --expt-extended-lambda;-lcublas;-lcurand;-Xcompiler -fPIC;-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored;-Xcudafe --diag_suppress=code_is_unreachable)
    set(CUDA_SEPARABLE_COMPILATION ON)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    macro(add_cuda_target cuda_target)
      cuda_add_library(${cuda_target} ${${cuda_target}_src}
        OPTIONS --compiler-options '-fPIC'
        RELEASE -DNDEBUG -O3 -lineinfo
        DEBUG -g -G -Xcompiler -rdynamic)
      # target_compile_options(${cuda_target} PRIVATE "-arch=compute_61;-code=sm_61;-lineinfo")
    endmacro()
  endif()
endif()

# Tell the code to use double precision instead of single
if(use_double)
  add_definitions(-DUSE_DOUBLE)
endif()

# add_subdirectory(deps/fmt)
add_subdirectory(src)
# add_subdirectory(tools)

if (build_tests)
  # Enable testing
  enable_testing(true)

  add_subdirectory(tests)
endif()

# Add the original Fortran code as compile target too
add_subdirectory(original)
