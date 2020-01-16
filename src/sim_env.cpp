#include "sim_env.h"
#include "cuda/constant_mem_func.h"
#include "data/sim_data.h"
#include "data/vec3.h"
#include "utils/nvproftool.h"
#include <cuda_runtime.h>
#include <mpi.h>

namespace Coffee {

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
sim_environment::send_guard_cell_x(sim_data& data, int dir) {
  send_array_x(data.E.data(0), dir);
  send_array_x(data.E.data(1), dir);
  send_array_x(data.E.data(2), dir);
  send_array_x(data.B.data(0), dir);
  send_array_x(data.B.data(1), dir);
  send_array_x(data.B.data(2), dir);
  send_array_x(data.P.data(), dir);
}

void
sim_environment::send_guard_cell_y(sim_data& data, int dir) {
  send_array_y(data.E.data(0), dir);
  send_array_y(data.E.data(1), dir);
  send_array_y(data.E.data(2), dir);
  send_array_y(data.B.data(0), dir);
  send_array_y(data.B.data(1), dir);
  send_array_y(data.B.data(2), dir);
  send_array_y(data.P.data(), dir);
}

void
sim_environment::send_guard_cell_z(sim_data& data, int dir) {
  send_array_z(data.E.data(0), dir);
  send_array_z(data.E.data(1), dir);
  send_array_z(data.E.data(2), dir);
  send_array_z(data.B.data(0), dir);
  send_array_z(data.B.data(1), dir);
  send_array_z(data.B.data(2), dir);
  send_array_z(data.P.data(), dir);
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
sim_environment::send_guard_cells(sim_data& data) {
  RANGE_PUSH("communication", CLR_CYAN);
  send_guard_cell_x(data, -1);
  send_guard_cell_x(data, 1);
  send_guard_cell_y(data, -1);
  send_guard_cell_y(data, 1);
  send_guard_cell_z(data, -1);
  send_guard_cell_z(data, 1);
  RANGE_POP;
}

void
sim_environment::send_guard_cell_array(multi_array<Scalar>& array) {
  send_array_x(array, -1);
  send_array_x(array, 1);
  send_array_y(array, -1);
  send_array_y(array, 1);
  send_array_z(array, -1);
  send_array_z(array, 1);
}

sim_environment::sim_environment(int* argc, char*** argv) {
  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  RANGE_PUSH("Initialization", CLR_BLUE);
  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }

  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

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

  // Hard coded to read the file config.toml in the current directory.
  // May want to make this more flexible
  m_params = parse_config("config.toml");

  // Copy the simulations params to the device constant memory
  init_dev_params(m_params);

  // Initialize the complete simulation grid
  for (int i = 0; i < 3; i++) {
    m_grid.guard[i] = m_params.guard[i];
    m_grid.sizes[i] = m_params.size[i];
    m_grid.lower[i] = m_params.lower[i];
    m_grid.dims[i] = m_params.N[i] + 2 * m_params.guard[i];
    m_grid.delta[i] = m_params.size[i] / m_params.N[i];
    m_grid.inv_delta[i] = 1.0 / m_grid.delta[i];
    m_is_periodic[i] = m_params.periodic_boundary[i];
  }

  // Setup domain decomposition, and update the grid according to the
  // local domain
  setup_domain();

  // Send the grid info to the device
  init_dev_grid(m_grid);
  RANGE_POP;

  m_send_buffers.resize(3);
  m_send_buffers[0] = multi_array<Scalar>(
      m_grid.guard[0], m_grid.dims[1], m_grid.dims[2]);
  m_send_buffers[1] = multi_array<Scalar>(
      m_grid.dims[0], m_grid.guard[1], m_grid.dims[2]);
  m_send_buffers[2] = multi_array<Scalar>(
      m_grid.dims[0], m_grid.dims[1], m_grid.guard[2]);
  m_recv_buffers.resize(3);
  m_recv_buffers[0] = multi_array<Scalar>(
      m_grid.guard[0], m_grid.dims[1], m_grid.dims[2]);
  m_recv_buffers[1] = multi_array<Scalar>(
      m_grid.dims[0], m_grid.guard[1], m_grid.dims[2]);
  m_recv_buffers[2] = multi_array<Scalar>(
      m_grid.dims[0], m_grid.dims[1], m_grid.guard[2]);

  m_scalar_type = (sizeof(Scalar) == 4 ? MPI_FLOAT : MPI_DOUBLE);
}

sim_environment::~sim_environment() {
  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

void
sim_environment::setup_domain() {
  // Split the whole world into number of cartesian dimensions
  int dims[3] = {1, 1, 1};
  int total_dim = 1;
  for (int i = 0; i < 3; i++) {
    dims[i] = m_params.nodes[i];
    total_dim *= dims[i];
  }

  if (total_dim != m_size) {
    // Given node configuration is not correct, create one on our own
    std::cerr << "Domain decomp in config file does not make sense!"
              << std::endl;
    for (int i = 0; i < 3; i++) dims[i] = 0;
    MPI_Dims_create(m_size, m_grid.dim(), dims);
  }

  for (int i = 0; i < 3; i++) m_mpi_dims[i] = dims[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, 3, dims, m_is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, 3, m_mpi_coord);

  // Figure out if the current rank is at any boundary
  int xleft, xright, yleft, yright, zleft, zright;
  int rank;
  MPI_Cart_shift(m_cart, 0, -1, &rank, &xleft);
  MPI_Cart_shift(m_cart, 0, 1, &rank, &xright);
  m_neighbor_left[0] = xleft;
  m_neighbor_right[0] = xright;
  if (xleft < 0) m_is_boundary[0] = true;
  if (xright < 0) m_is_boundary[1] = true;

  MPI_Cart_shift(m_cart, 1, -1, &rank, &yleft);
  MPI_Cart_shift(m_cart, 1, 1, &rank, &yright);
  m_neighbor_left[1] = yleft;
  m_neighbor_right[1] = yright;
  if (yleft < 0) m_is_boundary[2] = true;
  if (yright < 0) m_is_boundary[3] = true;

  MPI_Cart_shift(m_cart, 2, -1, &rank, &zleft);
  MPI_Cart_shift(m_cart, 2, 1, &rank, &zright);
  m_neighbor_left[2] = zleft;
  m_neighbor_right[2] = zright;
  if (zleft < 0) m_is_boundary[4] = true;
  if (zright < 0) m_is_boundary[5] = true;

  // Adjust the grid so that it matches the local domain
  for (int i = 0; i < 3; i++) {
    m_grid.dims[i] =
        2 * m_grid.guard[i] + m_grid.reduced_dim(i) / m_mpi_dims[i];
    m_grid.sizes[i] /= m_mpi_dims[i];
    m_grid.lower[i] =
        m_grid.lower[i] + m_mpi_coord[i] * m_grid.sizes[i];
    m_grid.offset[i] = m_grid.reduced_dim(i) * m_mpi_coord[i];
  }
}

void
sim_environment::initialize() {}

}  // namespace Coffee
