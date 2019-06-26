#include "sim_env.h"
#include "cuda/constant_mem_func.h"
#include "data/sim_data.h"
#include "data/vec3.h"
#include <cuda_runtime.h>
#include <mpi.h>

namespace Coffee {

// TODO: Finish this class
MPI_Datatype x_type, y_type;

void
sim_environment::exchange_type(MPI_Datatype* y_type,
                               MPI_Datatype* x_type) {
  MPI_Datatype x_temp;
  MPI_Datatype MPI_SCALAR =
      (sizeof(Scalar) == 4 ? MPI_FLOAT : MPI_DOUBLE);

  // Data exchange along z direction does not need new type.

  // int MPI_Type_vector(int count, int blocklength, int stride,
  //   MPI_Datatype oldtype, MPI_Datatype *newtype)

  // int MPI_Type_create_hvector(int count, int blocklength,
  //   MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype)

  // Data exchange along y direction
  MPI_Type_vector(m_grid.dims[2], m_grid.dims[0] * m_grid.guard[1],
                  m_grid.dims[0] * m_grid.dims[1], MPI_SCALAR, y_type);
  MPI_Type_commit(y_type);

  // Data exchange along x direction
  MPI_Type_vector(m_grid.dims[1], m_grid.guard[0], m_grid.dims[0],
                  MPI_SCALAR, &x_temp);
  MPI_Type_commit(&x_temp);
  MPI_Type_create_hvector(
      m_grid.dims[2], 1,
      sizeof(Scalar) * m_grid.dims[0] * m_grid.dims[1], x_temp, x_type);
  MPI_Type_commit(x_type);
}

void
sim_environment::send_guard_cell_x(sim_data& data, int dir) {
  int dest, origin;
  MPI_Status status;
  int send_offset, receive_offset;

  dest = (dir == -1 ? m_neighbor_left[0] : m_neighbor_right[0]);
  origin = (dir == -1 ? m_neighbor_right[0] : m_neighbor_left[0]);

  if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[0]
                           : m_grid.dims[0] - 2 * m_grid.guard[0]);
  receive_offset = (dir == -1 ? m_grid.dims[0] - m_grid.guard[0] : 0);

  // int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int
  // dest,
  //    int tag, MPI_Comm comm)
  // int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
  //    int source, int tag, MPI_Comm comm, MPI_Status *status)

  MPI_Send(data.E.dev_ptr(0) + send_offset, 1, x_type, dest, 0, m_cart);
  MPI_Recv(data.E.dev_ptr(0) + receive_offset, 1, x_type, origin, 0,
           m_cart, &status);
  MPI_Send(data.E.dev_ptr(1) + send_offset, 1, x_type, dest, 1, m_cart);
  MPI_Recv(data.E.dev_ptr(1) + receive_offset, 1, x_type, origin, 1,
           m_cart, &status);
  MPI_Send(data.E.dev_ptr(2) + send_offset, 1, x_type, dest, 2, m_cart);
  MPI_Recv(data.E.dev_ptr(2) + receive_offset, 1, x_type, origin, 2,
           m_cart, &status);

  MPI_Send(data.B.dev_ptr(0) + send_offset, 1, x_type, dest, 3, m_cart);
  MPI_Recv(data.B.dev_ptr(0) + receive_offset, 1, x_type, origin, 3,
           m_cart, &status);
  MPI_Send(data.B.dev_ptr(1) + send_offset, 1, x_type, dest, 4, m_cart);
  MPI_Recv(data.B.dev_ptr(1) + receive_offset, 1, x_type, origin, 4,
           m_cart, &status);
  MPI_Send(data.B.dev_ptr(2) + send_offset, 1, x_type, dest, 5, m_cart);
  MPI_Recv(data.B.dev_ptr(2) + receive_offset, 1, x_type, origin, 5,
           m_cart, &status);
}

void
sim_environment::send_guard_cell_y(sim_data& data, int dir) {
  int dest, origin;
  MPI_Status status;
  int send_offset, receive_offset;

  dest = (dir == -1 ? m_neighbor_left[1] : m_neighbor_right[1]);
  origin = (dir == -1 ? m_neighbor_right[1] : m_neighbor_left[1]);

  if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[1]
                           : m_grid.dims[1] - 2 * m_grid.guard[1]) *
                m_grid.dims[0];
  receive_offset = (dir == -1 ? m_grid.dims[1] - m_grid.guard[1] : 0) *
                   m_grid.dims[0];

  MPI_Sendrecv(data.E.dev_ptr(0) + send_offset, 1, y_type, dest, 0,
               data.E.dev_ptr(0) + receive_offset, 1, y_type, origin, 0, m_cart, &status);
  // MPI_Recv(data.E.dev_ptr(0) + receive_offset, 1, y_type, origin, 0,
  //          m_cart, &status);
  // MPI_Send(data.E.dev_ptr(1) + send_offset, 1, y_type, dest, 1, m_cart);
  // MPI_Recv(data.E.dev_ptr(1) + receive_offset, 1, y_type, origin, 1,
  //          m_cart, &status);
  // MPI_Send(data.E.dev_ptr(2) + send_offset, 1, y_type, dest, 2, m_cart);
  // MPI_Recv(data.E.dev_ptr(2) + receive_offset, 1, y_type, origin, 2,
  //          m_cart, &status);

  // MPI_Send(data.B.dev_ptr(0) + send_offset, 1, y_type, dest, 3, m_cart);
  // MPI_Recv(data.B.dev_ptr(0) + receive_offset, 1, y_type, origin, 3,
  //          m_cart, &status);
  // MPI_Send(data.B.dev_ptr(1) + send_offset, 1, y_type, dest, 4, m_cart);
  // MPI_Recv(data.B.dev_ptr(1) + receive_offset, 1, y_type, origin, 4,
  //          m_cart, &status);
  // MPI_Send(data.B.dev_ptr(2) + send_offset, 1, y_type, dest, 5, m_cart);
  // MPI_Recv(data.B.dev_ptr(2) + receive_offset, 1, y_type, origin, 5,
  //          m_cart, &status);
}

void
sim_environment::send_guard_cell_z(sim_data& data, int dir) {
  int dest, origin;
  MPI_Status status;
  int send_offset, receive_offset;
  int zsize = m_grid.dims[0] * m_grid.dims[1];
  MPI_Datatype MPI_SCALAR =
      (sizeof(Scalar) == 4 ? MPI_FLOAT : MPI_DOUBLE);

  dest = (dir == -1 ? m_neighbor_left[2] : m_neighbor_right[2]);
  origin = (dir == -1 ? m_neighbor_right[2] : m_neighbor_left[2]);

  if (dest == NEIGHBOR_NULL) dest = MPI_PROC_NULL;
  if (origin == NEIGHBOR_NULL) origin = MPI_PROC_NULL;

  send_offset = (dir == -1 ? m_grid.guard[2]
                           : m_grid.dims[2] - 2 * m_grid.guard[2]) *
                zsize;
  receive_offset =
      (dir == -1 ? m_grid.dims[2] - m_grid.guard[2] : 0) * zsize;

  MPI_Send(data.E.dev_ptr(0) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 0, m_cart);
  MPI_Recv(data.E.dev_ptr(0) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 0, m_cart, &status);
  MPI_Send(data.E.dev_ptr(1) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 1, m_cart);
  MPI_Recv(data.E.dev_ptr(1) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 1, m_cart, &status);
  MPI_Send(data.E.dev_ptr(2) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 2, m_cart);
  MPI_Recv(data.E.dev_ptr(2) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 2, m_cart, &status);

  MPI_Send(data.B.dev_ptr(0) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 3, m_cart);
  MPI_Recv(data.B.dev_ptr(0) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 3, m_cart, &status);
  MPI_Send(data.B.dev_ptr(1) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 4, m_cart);
  MPI_Recv(data.B.dev_ptr(1) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 4, m_cart, &status);
  MPI_Send(data.B.dev_ptr(2) + send_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, dest, 5, m_cart);
  MPI_Recv(data.B.dev_ptr(2) + receive_offset, m_grid.guard[2] * zsize,
           MPI_SCALAR, origin, 5, m_cart, &status);
}

void
sim_environment::send_guard_cells(sim_data& data) {
  send_guard_cell_x(data, -1);
  send_guard_cell_x(data, 1);
  send_guard_cell_y(data, -1);
  send_guard_cell_y(data, 1);
  send_guard_cell_z(data, -1);
  send_guard_cell_z(data, 1);
}

sim_environment::sim_environment(int* argc, char*** argv) {
  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

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
  for (int i = 0; i < m_grid.dim(); i++) dims[i] = 0;

  MPI_Dims_create(m_size, m_grid.dim(), dims);
  for (int i = 0; i < 3; i++) m_mpi_dims[i] = dims[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, m_grid.dim(), dims, m_is_periodic, true,
                  &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, 3, m_mpi_coord);

  // Figure out if the current rank is at any boundary
  int xleft, xright, yleft, yright, zleft, zright;
  int rank;
  MPI_Cart_shift(m_cart, 0, -1, &rank, &xleft);
  MPI_Cart_shift(m_cart, 0, 1, &rank, &xright);
  if (xleft < 0)
    m_is_boundary[0] = true;
  else
    m_neighbor_left[0] = xleft;
  if (xright < 0)
    m_is_boundary[1] = true;
  else
    m_neighbor_right[0] = xright;

  MPI_Cart_shift(m_cart, 1, -1, &rank, &yleft);
  MPI_Cart_shift(m_cart, 1, 1, &rank, &yright);
  if (yleft < 0)
    m_is_boundary[2] = true;
  else
    m_neighbor_left[1] = yleft;
  if (yright < 0)
    m_is_boundary[3] = true;
  else
    m_neighbor_right[1] = yright;

  MPI_Cart_shift(m_cart, 2, -1, &rank, &zleft);
  MPI_Cart_shift(m_cart, 2, 1, &rank, &zright);
  if (zleft < 0)
    m_is_boundary[4] = true;
  else
    m_neighbor_left[2] = zleft;
  if (zright < 0)
    m_is_boundary[5] = true;
  else
    m_neighbor_right[2] = zright;

  // Adjust the grid so that it matches the local domain
  for (int i = 0; i < m_grid.dim(); i++) {
    m_grid.dims[i] =
        2 * m_grid.guard[i] + m_grid.reduced_dim(i) / m_mpi_dims[i];
    m_grid.sizes[i] /= m_mpi_dims[i];
    m_grid.lower[i] =
        m_grid.lower[i] + m_mpi_coord[i] * m_grid.sizes[i];
    m_grid.offset[i] = m_grid.reduced_dim(i) * m_mpi_coord[i];
  }

  exchange_type(&y_type, &x_type);
}

void
sim_environment::initialize() {}

}  // namespace Coffee
