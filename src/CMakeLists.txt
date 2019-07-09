set(Coffee_src
  # "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/data_exporter.cpp"
  "data/fields.cpp"
  "data/sim_data.cpp"
  "sim_params.cpp"
  "sim_env.cpp"
  )

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})


set(Coffee_cuda_src
  "cuda/constant_mem.cu"
  "data/multi_array.cu"
  "algorithms/field_solver.cu"
  "algorithms/field_solver_gr.cu")

add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} Coffee)

add_executable(coffee "main.cpp")
target_link_libraries(coffee Coffee Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
