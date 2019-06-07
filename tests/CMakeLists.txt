add_custom_target(check)

add_executable(tests "test.cpp")
target_sources(tests PRIVATE
  "test_stagger.cpp"
  )
target_link_libraries(tests
  Coffee Coffee_cuda)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)
