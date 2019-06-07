cmake_minimum_required(VERSION 3.8)

project (Coffee LANGUAGES CXX)
message ("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

option(build_tests "Build the test suite." ON)
option(use_cuda "Build the GPU part of the code" ON)

set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -DNDEBUG -xHost -no-prec-div ${CXX_EXTRA_FLAGS}")
else()
  # set(CXX_EXTRA_FLAGS "-Wno-misleading-indentation -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -DNDEBUG -march=native ${CXX_EXTRA_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 ${CXX_EXTRA_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

set(FMT_INCLUDE_DIR deps/fmt)

include_directories(${MPI_INCLUDE_PATH} ${FMT_INCLUDE_DIR}
  ${CPPTOML_INCLUDE_DIR} ${CATCH_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} src)

######### Check for cuda support
if (use_cuda)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message("Enabling Cuda")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS "--expt-extended-lambda -lcublas -lcurand -Xcompiler -fPIC -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcudafe --diag_suppress=code_is_unreachable")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_FLAGS_RELEASE "-lineinfo -O3 -DNDEBUG")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    macro(add_cuda_target cuda_target)
      add_library(${cuda_target} ${${cuda_target}_src})
      # target_compile_options(${cuda_target} PRIVATE "-arch=compute_61;-code=sm_61;-lineinfo")
    endmacro()
    macro(add_cuda_executable cuda_target)
      add_executable(${cuda_target} ${${cuda_target}_src})
      # target_compile_options(${cuda_target} PRIVATE "-arch=compute_61;-code=sm_61;-lineinfo")
    endmacro()
  endif()
endif()

# Tell the code to use double precision instead of single
if(use_double)
  add_definitions(-DUSE_DOUBLE)
endif()

add_subdirectory(deps/fmt)
add_subdirectory(src)
# add_subdirectory(tools)

if (build_tests)
  # Enable testing
  enable_testing(true)

  add_subdirectory(tests)
endif()
