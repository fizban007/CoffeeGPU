#include "sim_env.h"
#include "sim_env_impl.hpp"
#include "cuda/constant_mem_func.h"
#include "data/sim_data.h"
#include "data/vec3.h"
// #include "utils/nvproftool.h"
#include <cuda_runtime.h>

namespace Coffee {

void
sim_environment::exchange_types(MPI_Datatype *y_type,
                                MPI_Datatype *x_type) {}

void
sim_environment::send_array_x(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_neighbor_left[0] : m_neighbor_right[0]);
  origin = (dir == -1 ? m_neighbor_right[0] : m_neighbor_left[0]);

  cudaExtent ext = make_cudaExtent(m_grid.guard[0] * sizeof(Scalar),
                                   m_grid.dims[1], m_grid.dims[2]);

  cudaMemcpy3DParms copy_parms = {0};
  copy_parms.srcPtr = make_cudaPitchedPtr(
      array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar), m_grid.dims[0],
      m_grid.dims[1]);
  copy_parms.srcPos =
      make_cudaPos((dir == -1 ? m_grid.guard[0]
                              : m_grid.dims[0] - 2 * m_grid.guard[0]) *
                       sizeof(Scalar),
                   0, 0);
  copy_parms.dstPtr = make_cudaPitchedPtr(
      m_send_buffers[0].host_ptr(), m_grid.guard[0] * sizeof(Scalar),
      m_grid.guard[0], m_grid.dims[1]);
  copy_parms.dstPos = make_cudaPos(0, 0, 0);
  copy_parms.extent = ext;
  copy_parms.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&copy_parms);

  MPI_Sendrecv(m_send_buffers[0].host_ptr(), m_send_buffers[0].size(),
               m_scalar_type, dest, 0, m_recv_buffers[0].host_ptr(),
               m_recv_buffers[0].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    copy_parms.srcPtr = make_cudaPitchedPtr(
        m_recv_buffers[0].host_ptr(), m_grid.guard[0] * sizeof(Scalar),
        m_grid.guard[0], m_grid.dims[1]);
    copy_parms.srcPos = make_cudaPos(0, 0, 0);
    copy_parms.dstPtr = make_cudaPitchedPtr(
        array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar),
        m_grid.dims[0], m_grid.dims[1]);
    copy_parms.dstPos = make_cudaPos(
        (dir == -1 ? m_grid.dims[0] - m_grid.guard[0] : 0) *
            sizeof(Scalar),
        0, 0);
    copy_parms.extent = ext;
    copy_parms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_parms);
  }
}

void
sim_environment::send_array_y(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_neighbor_left[1] : m_neighbor_right[1]);
  origin = (dir == -1 ? m_neighbor_right[1] : m_neighbor_left[1]);

  // array.copy_to_y_buffer(m_send_buffers[1], m_grid.guard[1], dir);
  cudaExtent ext =
      make_cudaExtent(m_grid.dims[0] * sizeof(Scalar), m_grid.guard[1], m_grid.dims[2]);

  cudaMemcpy3DParms copy_parms = {0};
  copy_parms.srcPtr = make_cudaPitchedPtr(
      array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar), m_grid.dims[0],
      m_grid.dims[1]);
  copy_parms.srcPos =
      make_cudaPos(0,
                   (dir == -1 ? m_grid.guard[1]
                              : m_grid.dims[1] - 2 * m_grid.guard[1]),
                   0);
  copy_parms.dstPtr = make_cudaPitchedPtr(
      m_send_buffers[1].host_ptr(), m_grid.dims[0] * sizeof(Scalar),
      m_grid.dims[0], m_grid.guard[1]);
  copy_parms.dstPos = make_cudaPos(0, 0, 0);
  copy_parms.extent = ext;
  copy_parms.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&copy_parms);

  MPI_Sendrecv(m_send_buffers[1].host_ptr(), m_send_buffers[1].size(),
               m_scalar_type, dest, 0, m_recv_buffers[1].host_ptr(),
               m_recv_buffers[1].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    copy_parms.srcPtr = make_cudaPitchedPtr(
        m_recv_buffers[1].host_ptr(), m_grid.dims[0] * sizeof(Scalar),
        m_grid.dims[0], m_grid.guard[1]);
    copy_parms.srcPos = make_cudaPos(0, 0, 0);
    copy_parms.dstPtr = make_cudaPitchedPtr(
        array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar),
        m_grid.dims[0], m_grid.dims[1]);
    copy_parms.dstPos = make_cudaPos(
        0, (dir == -1 ? m_grid.dims[1] - m_grid.guard[1] : 0), 0);
    copy_parms.extent = ext;
    copy_parms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_parms);
  }
}

void
sim_environment::send_array_z(multi_array<Scalar>& array, int dir) {
  int dest, origin;
  MPI_Status status;

  dest = (dir == -1 ? m_neighbor_left[2] : m_neighbor_right[2]);
  origin = (dir == -1 ? m_neighbor_right[2] : m_neighbor_left[2]);

  // array.copy_to_y_buffer(m_send_buffers[1], m_grid.guard[1], dir);
  cudaExtent ext =
      make_cudaExtent(m_grid.dims[0] * sizeof(Scalar), m_grid.dims[1], m_grid.guard[2]);

  cudaMemcpy3DParms copy_parms = {0};
  copy_parms.srcPtr = make_cudaPitchedPtr(
      array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar), m_grid.dims[0],
      m_grid.dims[1]);
  copy_parms.srcPos =
      make_cudaPos(0, 0,
                   (dir == -1 ? m_grid.guard[2]
                              : m_grid.dims[2] - 2 * m_grid.guard[2]));
  copy_parms.dstPtr = make_cudaPitchedPtr(
      m_send_buffers[2].host_ptr(), m_grid.dims[0] * sizeof(Scalar),
      m_grid.dims[0], m_grid.dims[1]);
  copy_parms.dstPos = make_cudaPos(0, 0, 0);
  copy_parms.extent = ext;
  copy_parms.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&copy_parms);

  MPI_Sendrecv(m_send_buffers[2].host_ptr(), m_send_buffers[2].size(),
               m_scalar_type, dest, 0, m_recv_buffers[2].host_ptr(),
               m_recv_buffers[2].size(), m_scalar_type, origin, 0,
               m_cart, &status);

  if (status.MPI_SOURCE != MPI_PROC_NULL) {
    copy_parms.srcPtr = make_cudaPitchedPtr(
        m_recv_buffers[2].host_ptr(), m_grid.dims[0] * sizeof(Scalar),
        m_grid.dims[0], m_grid.dims[1]);
    copy_parms.srcPos = make_cudaPos(0, 0, 0);
    copy_parms.dstPtr = make_cudaPitchedPtr(
        array.dev_ptr(), m_grid.dims[0] * sizeof(Scalar),
        m_grid.dims[0], m_grid.dims[1]);
    copy_parms.dstPos = make_cudaPos(
        0, 0, (dir == -1 ? m_grid.dims[2] - m_grid.guard[2] : 0));
    copy_parms.extent = ext;
    copy_parms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_parms);
  }
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

  // MPI_Irecv(data.E.dev_ptr(0) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 0,
  //           m_cart, &requests[0]);
  // MPI_Isend(data.E.dev_ptr(0) + send_offset, zsize1, MPI_SCALAR,
  // dest, 0, m_cart,
  //           &requests[1]);
  // MPI_Irecv(data.E.dev_ptr(1) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 1,
  //           m_cart, &requests[2]);
  // MPI_Isend(data.E.dev_ptr(1) + send_offset, zsize1, MPI_SCALAR,
  // dest, 1, m_cart,
  //           &requests[3]);
  // MPI_Irecv(data.E.dev_ptr(2) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 2,
  //           m_cart, &requests[4]);
  // MPI_Isend(data.E.dev_ptr(2) + send_offset, zsize1, MPI_SCALAR,
  // dest, 2, m_cart,
  //           &requests[5]);

  // MPI_Irecv(data.B.dev_ptr(0) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 3,
  //           m_cart, &requests[6]);
  // MPI_Isend(data.B.dev_ptr(0) + send_offset, zsize1, MPI_SCALAR,
  // dest, 3, m_cart,
  //           &requests[7]);
  // MPI_Irecv(data.B.dev_ptr(1) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 4,
  //           m_cart, &requests[8]);
  // MPI_Isend(data.B.dev_ptr(1) + send_offset, zsize1, MPI_SCALAR,
  // dest, 4, m_cart,
  //           &requests[9]);
  // MPI_Irecv(data.B.dev_ptr(2) + receive_offset, zsize1, MPI_SCALAR,
  // origin, 5,
  //           m_cart, &requests[10]);
  // MPI_Isend(data.B.dev_ptr(2) + send_offset, zsize1, MPI_SCALAR,
  // dest, 5, m_cart,
  //           &requests[11]);

  // MPI_Waitall(12, requests, NULL);

  MPI_Sendrecv(data.E.dev_ptr(0) + send_offset, zsize1, MPI_SCALAR,
               dest, 0, data.E.dev_ptr(0) + receive_offset, zsize1,
               MPI_SCALAR, origin, 0, m_cart, &status);
  MPI_Sendrecv(data.E.dev_ptr(1) + send_offset, zsize1, MPI_SCALAR,
               dest, 1, data.E.dev_ptr(1) + receive_offset, zsize1,
               MPI_SCALAR, origin, 1, m_cart, &status);
  MPI_Sendrecv(data.E.dev_ptr(2) + send_offset, zsize1, MPI_SCALAR,
               dest, 2, data.E.dev_ptr(2) + receive_offset, zsize1,
               MPI_SCALAR, origin, 2, m_cart, &status);

  MPI_Sendrecv(data.B.dev_ptr(0) + send_offset, zsize1, MPI_SCALAR,
               dest, 3, data.B.dev_ptr(0) + receive_offset, zsize1,
               MPI_SCALAR, origin, 3, m_cart, &status);
  MPI_Sendrecv(data.B.dev_ptr(1) + send_offset, zsize1, MPI_SCALAR,
               dest, 4, data.B.dev_ptr(1) + receive_offset, zsize1,
               MPI_SCALAR, origin, 4, m_cart, &status);
  MPI_Sendrecv(data.B.dev_ptr(2) + send_offset, zsize1, MPI_SCALAR,
               dest, 5, data.B.dev_ptr(2) + receive_offset, zsize1,
               MPI_SCALAR, origin, 5, m_cart, &status);
}

void
sim_environment::setup_env_extra() {
  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    std::cerr << "No usable Cuda device found!!" << std::endl;
    exit(1);
  }
  // TODO: This way of finding device id may not be reliable
  m_dev_id = m_rank % n_devices;
  cudaSetDevice(m_dev_id);

  // Copy the simulations params to the device constant memory
  init_dev_params(m_params);

  // Send the grid info to the device
  init_dev_grid(m_grid);
}

}  // namespace Coffee
