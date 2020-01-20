set(Coffee_src
  # "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/data_exporter.cpp"
  "data/fields.cpp"
  "data/sim_data.cpp"
  "sim_params.cpp"
  # "sim_env.cpp"
  )

if(use_cuda AND CMAKE_CUDA_COMPILER)
  list(APPEND Coffee_src
    "cuda/sim_env.cpp"
    "cuda/constant_mem.cu"
    "cuda/data/multi_array.cu"
    "algorithms/boundary.cu"
    "algorithms/field_solver.cu"
    "algorithms/field_solver_EZ.cu"
    # "algorithms/field_solver_EZ_cylindrical.cu"
    # "algorithms/field_solver_gr.cu"
    "algorithms/pulsar.cu"
   )
else()
  list(APPEND Coffee_src
    "data/multi_array.cpp"
    "sim_env.cpp"
    "algorithms/field_solver_EZ.cpp"
    "algorithms/pulsar.cpp"
   )
endif()

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})


# set(Coffee_cuda_src
#   "cuda/constant_mem.cu"
#   "data/multi_array.cu"
#   "algorithms/boundary.cu"
#   "algorithms/field_solver.cu"
#   "algorithms/field_solver_EZ.cu"
#   # "algorithms/field_solver_EZ_cylindrical.cu"
#   # "algorithms/field_solver_gr.cu"
#   "algorithms/pulsar.cu"
#   )

# add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
# target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} Coffee)

add_executable(coffee "main.cpp")
target_link_libraries(coffee Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
