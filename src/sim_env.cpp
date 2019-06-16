#include "sim_env.h"
#include "cuda/constant_mem_func.h"

namespace Coffee {

// TODO: Finish this class

sim_environment::sim_environment(int* argc, char*** argv) {
  // TODO: initialize mpi here using argc and argv

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
  }

  // Setup domain decomposition, and update the grid according to the
  // local domain
  setup_domain();
}

sim_environment::~sim_environment() {}

void
sim_environment::setup_domain() {}

void
sim_environment::initialize() {}

}  // namespace Coffee
