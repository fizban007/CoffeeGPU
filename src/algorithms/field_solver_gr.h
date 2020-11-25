#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver_gr {
 public:

  vector_field<Scalar> Dn, Bn, dD, dB, Ed, Hd;

  field_solver_gr(sim_data& mydata, sim_environment& env);
  ~field_solver_gr();

  void evolve_fields_gr();

 private:
  sim_data& m_data;
  sim_environment& m_env;
  void copy_fields_gr();

  void rk_push_gr();
  void rk_update_gr(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3);

  void check_eGTb_gr();
  void clean_epar_gr();
  void compute_E_gr();
  void compute_H_gr();

  void absorbing_boundary();
};

}  // namespace Coffee
