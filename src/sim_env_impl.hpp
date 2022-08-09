#ifndef _SIM_ENV_IMPL_H_
#define _SIM_ENV_IMPL_H_

#include "cxxopts.hpp"
#include "data/sim_data.h"
#include "data/vec3.h"
#include "sim_env.h"
#include <mpi.h>

namespace Coffee {

sim_environment::sim_environment(int* argc, char*** argv) {
  // Parse options
  cxxopts::Options options("Coffee",
                           "Computational Force-free Electrodynamics");
  options.add_options()("r,restart-file", "Path of the restart file",
                        cxxopts::value<std::string>()->implicit_value(
                            "snapshot.h5"))("h,help", "Print usage");

  auto result = options.parse(*argc, *argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  if (result.count("restart-file")) {
    m_is_restart = true;
    // m_restart_file = result["restart-file"].as<std::string>();
  }

  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  // RANGE_PUSH("Initialization", CLR_BLUE);
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

  // Hard coded to read the file config.toml in the current directory.
  // May want to make this more flexible
  m_params = parse_config("config.toml");

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

  // RANGE_POP;

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

  setup_env_extra();
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
    std::cerr << "Created domain decomp as ";
    for (int i = 0; i < m_grid.dim(); i++) {
      std::cerr << dims[i];
      if (i != m_grid.dim() - 1) std::cerr << " x ";
    }
    std::cerr << "\n";
  }

  for (int i = 0; i < 3; i++) m_mpi_dims[i] = dims[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, m_grid.dim(), dims, m_is_periodic, true,
                  &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, m_grid.dim(), m_mpi_coord);

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
  for (int i = 0; i < m_grid.dim(); i++) {
    m_grid.dims[i] =
        2 * m_grid.guard[i] + m_grid.reduced_dim(i) / m_mpi_dims[i];
    m_grid.sizes[i] /= m_mpi_dims[i];
    m_grid.lower[i] =
        m_grid.lower[i] + m_mpi_coord[i] * m_grid.sizes[i];
    m_grid.offset[i] = m_grid.reduced_dim(i) * m_mpi_coord[i];
  }
}

void
sim_environment::send_guard_cell_array(multi_array<Scalar>& array) {
  if (m_mpi_dims[0] > 1 || m_is_periodic[0]) {
    send_array_x(array, -1);
    send_array_x(array, 1);
  }
  if (m_mpi_dims[1] > 1 || m_is_periodic[1]) {
    send_array_y(array, -1);
    send_array_y(array, 1);
  }
  if (m_mpi_dims[2] > 1 || m_is_periodic[2]) {
    send_array_z(array, -1);
    send_array_z(array, 1);
  }
}

void
sim_environment::send_guard_cells(sim_data& data) {
  // RANGE_PUSH("communication", CLR_CYAN);
  if (m_mpi_dims[0] > 1 || m_is_periodic[0]) {
    send_guard_cell_x(data, -1);
    send_guard_cell_x(data, 1);
  }
  if (m_mpi_dims[1] > 1 || m_is_periodic[1]) {
    send_guard_cell_y(data, -1);
    send_guard_cell_y(data, 1);
  }
  if (m_mpi_dims[2] > 1 || m_is_periodic[2]) {
    send_guard_cell_z(data, -1);
    send_guard_cell_z(data, 1);
  }
  // RANGE_POP;
}

void
sim_environment::send_guard_cell_x(sim_data& data, int dir) {
  send_array_x(data.E.data(0), dir);
  send_array_x(data.E.data(1), dir);
  send_array_x(data.E.data(2), dir);
  send_array_x(data.B.data(0), dir);
  send_array_x(data.B.data(1), dir);
  send_array_x(data.B.data(2), dir);
  send_array_x(data.P, dir);
}

void
sim_environment::send_guard_cell_y(sim_data& data, int dir) {
  send_array_y(data.E.data(0), dir);
  send_array_y(data.E.data(1), dir);
  send_array_y(data.E.data(2), dir);
  send_array_y(data.B.data(0), dir);
  send_array_y(data.B.data(1), dir);
  send_array_y(data.B.data(2), dir);
  send_array_y(data.P, dir);
}

void
sim_environment::send_guard_cell_z(sim_data& data, int dir) {
  send_array_z(data.E.data(0), dir);
  send_array_z(data.E.data(1), dir);
  send_array_z(data.E.data(2), dir);
  send_array_z(data.B.data(0), dir);
  send_array_z(data.B.data(1), dir);
  send_array_z(data.B.data(2), dir);
  send_array_z(data.P, dir);
}

}  // namespace Coffee

#endif  // _SIM_ENV_IMPL_H_
