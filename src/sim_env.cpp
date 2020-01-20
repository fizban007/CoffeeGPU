#include "sim_env.h"
#include "data/multi_array.h"
#include "data/sim_data.h"
#include "data/vec3.h"
#include "sim_env_impl.hpp"
// #include "utils/nvproftool.h"
// #include <cuda_runtime.h>

namespace Coffee {

MPI_Datatype x_type, y_type;

void
sim_environment::exchange_types(MPI_Datatype* y_type,
                                MPI_Datatype* x_type) {
  MPI_Datatype x_temp;

  // Data exchange along z direction does not need new type.

  // int MPI_Type_vector(int count, int blocklength, int stride,
  //   MPI_Datatype oldtype, MPI_Datatype *newtype)

  // int MPI_Type_create_hvector(int count, int blocklength,
  //   MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)

  // Data exchange along y direction
  MPI_Type_vector(m_grid.dims[2], m_grid.dims[0] * m_grid.guard[1],
                  m_grid.dims[0] * m_grid.dims[1], m_scalar_type,
                  y_type);
  MPI_Type_commit(y_type);

  // Data exchange along x direction
  MPI_Type_vector(m_grid.dims[1], m_grid.guard[0], m_grid.dims[0],
                  m_scalar_type, &x_temp);
  MPI_Type_commit(&x_temp);
  MPI_Type_create_hvector(
      m_grid.dims[2], 1,
      sizeof(Scalar) * m_grid.dims[0] * m_grid.dims[1], x_temp, x_type);
  MPI_Type_commit(x_type);
}

void
sim_environment::send_array_x(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;
  // MPI_Request requests[12];
  int send_offset, receive_offset;

  dest = (dir == -1 ? m_neighbor_left[0] : m_neighbor_right[0]);
  origin = (dir == -1 ? m_neighbor_right[0] : m_neighbor_left[0]);

  // if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  // if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[0]
                           : m_grid.dims[0] - 2 * m_grid.guard[0]);
  receive_offset = (dir == -1 ? m_grid.dims[0] - m_grid.guard[0] : 0);

  // MPI_Irecv(data.E.dev_ptr(0) + receive_offset, 1, x_type, origin, 0,
  //           m_cart, &requests[0]);
  // MPI_Isend(data.E.dev_ptr(0) + send_offset, 1, x_type, dest, 0,
  // m_cart,
  //           &requests[1]);
  MPI_Sendrecv(array.host_ptr() + send_offset, 1, x_type, dest, 0,
               array.host_ptr() + receive_offset, 1, x_type, origin, 0,
               m_cart, &status);
}

void
sim_environment::send_array_y(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;
  MPI_Request requests[12];
  int send_offset, receive_offset;

  dest = (dir == -1 ? m_neighbor_left[1] : m_neighbor_right[1]);
  origin = (dir == -1 ? m_neighbor_right[1] : m_neighbor_left[1]);

  // if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  // if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[1]
                           : m_grid.dims[1] - 2 * m_grid.guard[1]) *
                m_grid.dims[0];
  receive_offset = (dir == -1 ? m_grid.dims[1] - m_grid.guard[1] : 0) *
                   m_grid.dims[0];

  MPI_Sendrecv(array.host_ptr() + send_offset, 1, y_type, dest, 0,
               array.host_ptr() + receive_offset, 1, y_type, origin, 0,
               m_cart, &status);
}

void
sim_environment::send_array_z(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;
  // MPI_Request requests[12];
  int send_offset, receive_offset;
  int zsize = m_grid.dims[0] * m_grid.dims[1];

  dest = (dir == -1 ? m_neighbor_left[2] : m_neighbor_right[2]);
  origin = (dir == -1 ? m_neighbor_right[2] : m_neighbor_left[2]);

  send_offset = (dir == -1 ? m_grid.guard[2]
                           : m_grid.dims[2] - 2 * m_grid.guard[2]) *
                zsize;
  receive_offset =
      (dir == -1 ? m_grid.dims[2] - m_grid.guard[2] : 0) * zsize;

  int zsize1 = zsize * m_grid.guard[2];

  MPI_Sendrecv(array.host_ptr() + send_offset, zsize1, m_scalar_type,
               dest, 0, array.host_ptr() + receive_offset, zsize1,
               m_scalar_type, origin, 0, m_cart, &status);
}

void
sim_environment::send_guard_cell_z_old(sim_data& data, int dir) {
  int dest, origin;
  MPI_Status status;
  MPI_Request requests[12];
  int send_offset, receive_offset;
  int zsize = m_grid.dims[0] * m_grid.dims[1];
  MPI_Datatype MPI_SCALAR =
      (sizeof(Scalar) == 4 ? MPI_FLOAT : MPI_DOUBLE);

  dest = (dir == -1 ? m_neighbor_left[2] : m_neighbor_right[2]);
  origin = (dir == -1 ? m_neighbor_right[2] : m_neighbor_left[2]);

  // if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  // if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[2]
                           : m_grid.dims[2] - 2 * m_grid.guard[2]) *
                zsize;
  receive_offset =
      (dir == -1 ? m_grid.dims[2] - m_grid.guard[2] : 0) * zsize;

  int zsize1 = zsize * m_grid.guard[2];

  MPI_Sendrecv(data.E.host_ptr(0) + send_offset, zsize1, MPI_SCALAR,
               dest, 0, data.E.host_ptr(0) + receive_offset, zsize1,
               MPI_SCALAR, origin, 0, m_cart, &status);
  MPI_Sendrecv(data.E.host_ptr(1) + send_offset, zsize1, MPI_SCALAR,
               dest, 1, data.E.host_ptr(1) + receive_offset, zsize1,
               MPI_SCALAR, origin, 1, m_cart, &status);
  MPI_Sendrecv(data.E.host_ptr(2) + send_offset, zsize1, MPI_SCALAR,
               dest, 2, data.E.host_ptr(2) + receive_offset, zsize1,
               MPI_SCALAR, origin, 2, m_cart, &status);

  MPI_Sendrecv(data.B.host_ptr(0) + send_offset, zsize1, MPI_SCALAR,
               dest, 3, data.B.host_ptr(0) + receive_offset, zsize1,
               MPI_SCALAR, origin, 3, m_cart, &status);
  MPI_Sendrecv(data.B.host_ptr(1) + send_offset, zsize1, MPI_SCALAR,
               dest, 4, data.B.host_ptr(1) + receive_offset, zsize1,
               MPI_SCALAR, origin, 4, m_cart, &status);
  MPI_Sendrecv(data.B.host_ptr(2) + send_offset, zsize1, MPI_SCALAR,
               dest, 5, data.B.host_ptr(2) + receive_offset, zsize1,
               MPI_SCALAR, origin, 5, m_cart, &status);
}

void
sim_environment::setup_env_extra() {
  exchange_types(&y_type, &x_type);
}

}  // namespace Coffee
