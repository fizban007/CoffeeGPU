#include "sim_env.h"
#include "cuda/constant_mem_func.h"
#include <mpi.h>
#include "data/vec3.h"

namespace Coffee {

// TODO: Finish this class

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
  for (int i = 0; i < m_grid.dim(); i++)
    dims[i] = 0;

  MPI_Dims_create(m_size, m_grid.dim(), dims);
  for (int i = 0; i < 3; i++)
    m_mpi_dims[i] = dims[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, m_grid.dim(), dims, m_is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_rank, 3, m_mpi_coord);

  // Figure out if the current rank is at any boundary
  int xleft, xright, yleft, yright, zleft, zright;
  int rank;
  MPI_Cart_shift(m_cart, 0, -1, &rank, &xleft);
  // std::cout << "xleft of " << m_rank << " is " << xleft << std::endl;
  MPI_Cart_shift(m_cart, 0, 1, &rank, &xright);
  // std::cout << "xright of " << m_rank << " is " << xright << std::endl;
  if (xleft < 0) m_is_boundary[0] = true;
  if (xright < 0) m_is_boundary[1] = true;
  MPI_Cart_shift(m_cart, 1, -1, &rank, &yleft);
  // std::cout << "yleft of " << m_rank << " is " << yleft << std::endl;
  MPI_Cart_shift(m_cart, 1, 1, &rank, &yright);
  // std::cout << "yright of " << m_rank << " is " << yright << std::endl;
  if (yleft < 0) m_is_boundary[2] = true;
  if (yright < 0) m_is_boundary[3] = true;
  MPI_Cart_shift(m_cart, 2, -1, &rank, &zleft);
  // std::cout << "zleft of " << m_rank << " is " << zleft << std::endl;
  MPI_Cart_shift(m_cart, 2, 1, &rank, &zright);
  // std::cout << "zright of " << m_rank << " is " << zright << std::endl;
  if (zleft < 0) m_is_boundary[4] = true;
  if (zright < 0) m_is_boundary[5] = true;
}

void
sim_environment::initialize() {}

}  // namespace Coffee
