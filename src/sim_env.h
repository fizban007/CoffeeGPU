#ifndef _SIM_ENV_H_
#define _SIM_ENV_H_

#include "data/grid.h"
#include "data/multi_array.h"
#include "sim_params.h"
#include <vector>

namespace Coffee {

class sim_environment {
 public:
  sim_environment(int* argc, char*** argv);
  ~sim_environment();

  // Disable copy and assignment operators
  sim_environment(sim_environment const&) = delete;
  sim_environment& operator=(sim_environment const&) = delete;

  const Grid& grid() const { return m_grid; }
  const sim_params& params() const { return m_params; }

 private:
  void initialize();
  void setup_domain();

  sim_params m_params;
  Grid m_grid;

  int m_dim = 1;
  int m_rank = 0;
  int m_mpi_dims[3] = {1};
  int m_mpi_coord[3] = {0};
  bool m_is_boundary = {false};
  bool m_is_periodic = {false};
  multi_array<int> m_domain_map;
};  // ----- end of class sim_environment

}  // namespace Coffee

#endif  // _SIM_ENV_H_
