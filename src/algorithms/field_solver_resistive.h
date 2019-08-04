#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver_resistive {
 public:

  vector_field<Scalar> En, Bn, dE, dB;
  multi_array<Scalar> rho;

  field_solver_resistive(sim_data& mydata, sim_environment& env);
  ~field_solver_resistive();

  void evolve_fields();
  void light_curve(uint32_t step);

 private:
  sim_data& m_data;
  sim_environment& m_env;
  std::vector<Scalar> lc, lc0;
  void copy_fields();

  void rk_push_noj();
  void rk_push_ffjperp();
  void rk_push_jvacuum();
  void rk_push_rjperp();
  void rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3);
  void rk_update_rjparsub(Scalar rk_c3);

  void clean_epar();
  void check_eGTb();

  void absorbing_boundary();
  void disk_boundary();
};

}  // namespace Coffee
