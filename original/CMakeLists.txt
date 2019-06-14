 set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f90)

set(PFLAGS "-DMPI -DHDF5")
# set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${bounds}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -free -O3 ${PFLAGS} ${MPI_Fortran_COMPILE_FLAGS}")
include_directories(${MPI_Fortran_INCLUDE_PATH})

set(SRC_DIR src)
set(PREC_SRC_DIR src_)

execute_process(
  COMMAND ./precomp.sh ${SRC_DIR} ${PREC_SRC_DIR}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/original)

set(ffree_src "${PREC_SRC_DIR}/ffree3D.F90"
  "${PREC_SRC_DIR}/algorithm.f90"
  "${PREC_SRC_DIR}/auxiliary.f90"
  "${PREC_SRC_DIR}/output.f90"
  "${PREC_SRC_DIR}/boundary_conditions.f90"
  "${PREC_SRC_DIR}/user_bc_loop.f90"
  "${PREC_SRC_DIR}/user_bc_uniform.f90"
  "${PREC_SRC_DIR}/user_init_loop.f90"
  "${PREC_SRC_DIR}/user_init_uniform.f90"
)

add_executable(ffree ${ffree_src})
target_link_libraries(ffree ${MPI_Fortran_LIBRARIES} ${HDF5_LIBRARIES})
