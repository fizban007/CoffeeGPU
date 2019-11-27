#ifndef _SIM_ENV_H_
#define _SIM_ENV_H_

#include "data/grid.h"
#include "data/multi_array.h"
#include "sim_params.h"
#include <vector>
#include <array>
#include <mpi.h>

#define NEIGHBOR_NULL -2

namespace Coffee {

struct sim_data;

class sim_environment {
 public:
  sim_environment(int* argc, char*** argv);
  ~sim_environment();

  // Disable copy and assignment operators
  sim_environment(sim_environment const&) = delete;
  sim_environment& operator=(sim_environment const&) = delete;

  void send_guard_cells(sim_data& data);

  void send_guard_cell_x(sim_data& data, int dir);
  void send_guard_cell_y(sim_data& data, int dir);
  void send_guard_cell_z_old(sim_data& data, int dir);
  void send_guard_cell_z(sim_data& data, int dir);

  void send_guard_cell_array(multi_array<Scalar>& array);

  const Grid& grid() const { return m_grid; }
  const sim_params& params() const { return m_params; }

  int size() const { return m_size; }
  int rank() const { return m_rank; }
  MPI_Comm world() const { return m_world; }

  bool is_boundary(int n) const { return m_is_boundary[n]; }
  bool is_periodic(int n) const { return m_is_periodic[n]; }

 private:
  void initialize();
  void setup_domain();

  void send_array_x(multi_array<Scalar>& array, int dir);
  void send_array_y(multi_array<Scalar>& array, int dir);
  void send_array_z(multi_array<Scalar>& array, int dir);

  sim_params m_params;
  Grid m_grid;
  int m_dev_id;

  int m_size = 1; ///< Size of MPI_COMM_WORLD
  int m_rank = 0; ///< Rank of current process
  int m_mpi_dims[3] = {1}; ///< Size of the domain decomposition in 3 directions
  int m_mpi_coord[3] = {0}; ///< The 3D MPI coordinate of this rank
  bool m_is_boundary[6] = {false}; ///< Is this rank at boundary in each direction
  int m_is_periodic[3] = {0}; ///< Whether to use periodic boundary conditions in each direction
  int m_neighbor_left[3] = {NEIGHBOR_NULL};
  int m_neighbor_right[3] = {NEIGHBOR_NULL};
  multi_array<int> m_domain_map;

  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;

  std::vector<multi_array<Scalar>> m_send_buffers;
  std::vector<multi_array<Scalar>> m_recv_buffers;
};  // ----- end of class sim_environment

}  // namespace Coffee

#endif  // _SIM_ENV_H_
