set(Coffee_src
  "utils/logger.cpp"
  "utils/timer.cpp")

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} fmt)
