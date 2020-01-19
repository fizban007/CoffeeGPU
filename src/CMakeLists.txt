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
  "algorithms/boundary.cu"
  "algorithms/field_solver.cu"
  "algorithms/field_solver_EZ.cu"
  "algorithms/field_solver_gr_EZ.cu"
  # "algorithms/field_solver_EZ_cylindrical.cu"
  # "algorithms/field_solver_gr.cu"
  "algorithms/pulsar.cu"
  )

add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} Coffee)

add_executable(coffee "main.cpp")
target_link_libraries(coffee Coffee Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-2d-cylindrical "main-pulsar2d-cylindrical.cpp")
target_link_libraries(coffee-2d-cylindrical Coffee Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-2d-spherical "main-pulsar2d-spherical.cpp")
target_link_libraries(coffee-2d-spherical Coffee Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-3d "main-pulsar3d.cpp")
target_link_libraries(coffee-3d Coffee Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
