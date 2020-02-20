set(Coffee_src
  # "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/data_exporter.cpp"
  "utils/hdf_wrapper.cpp"
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
    "algorithms/field_solver_gr_EZ.cu"
    "algorithms/field_solver_EZ_cylindrical.cu"
    "algorithms/field_solver_EZ_spherical.cu"
    # "algorithms/field_solver_gr.cu"
    "algorithms/pulsar.cu"
    "algorithms/metric_cks.cu"
   )
else()
  list(APPEND Coffee_src
    "data/multi_array.cpp"
    "sim_env.cpp"
    "algorithms/field_solver_EZ.cpp"
    "algorithms/pulsar.cpp"
    "algorithms/damping_boundary.cpp"
    "utils/memory.cpp"
   )
endif()

add_library(Coffee ${Coffee_src})
target_link_libraries(Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})



# add_cuda_target(Coffee_cuda ${Coffee_cuda_src})
# target_link_libraries(Coffee_cuda ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} Coffee)

add_executable(coffee "main.cpp")
target_link_libraries(coffee Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-2d-cylindrical "main-pulsar2d-cylindrical.cpp")
target_link_libraries(coffee-2d-cylindrical Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-2d-spherical "main-pulsar2d-spherical.cpp")
target_link_libraries(coffee-2d-spherical Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-3d "main-pulsar3d.cpp")
target_link_libraries(coffee-3d Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-gr "main-gr.cpp")
target_link_libraries(coffee-gr Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

add_executable(coffee-wave "main-wave.cpp")
target_link_libraries(coffee-wave Coffee ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

