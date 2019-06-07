set(Coffee_src
  "utils/logger.cpp"
  "utils/timer.cpp")

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} fmt)


set(Coffee_cuda_src
  "data/multi_array.cu")

add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} fmt Coffee)
