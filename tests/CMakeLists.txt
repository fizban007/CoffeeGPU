add_custom_target(check)

add_executable(tests "test.cpp"
  "test_stagger.cpp"
  "test_multi_array.cpp"
  "test_interp.cpp"
  "test_hdf5.cpp"
  # "test_algorithm.cpp"
  )
target_link_libraries(tests Coffee)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

if(use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(tests_cuda "test.cu"
    "test_array.cu"
    # "test_rho.cu"
    )
  target_link_libraries(tests_cuda Coffee)

  add_custom_target(run_cuda_tests
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests_cuda"
    DEPENDS tests_cuda
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  add_dependencies(check run_cuda_tests)
endif()

add_executable(test_mpi "test_mpi.cpp")
target_link_libraries(test_mpi Coffee)

add_executable(test_output "test_output.cpp")
target_link_libraries(test_output Coffee)
