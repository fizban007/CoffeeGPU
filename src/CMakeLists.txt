set(Coffee_src
  # "utils/logger.cpp"
  "data/fields.cpp"
  "utils/timer.cpp"
  "sim_params.cpp"
  )

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES})


set(Coffee_cuda_src
  "cuda/constant_mem.cu"
  "data/multi_array.cu")

add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} Coffee)
