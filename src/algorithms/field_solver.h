#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver {
 public:

  vector_field<Scalar> En, Bn, dE, dB;
  multi_array<Scalar> rho;

  field_solver(sim_data& mydata, sim_environment& env);
  ~field_solver();

  void evolve_fields(Scalar t);
  Scalar total_energy(vector_field<Scalar> &f);

 private:
  sim_data& m_data;
  sim_environment& m_env;

  void copy_fields();

  void rk_push();
  void rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3);

  void clean_epar();
  void check_eGTb();

  void boundary_pulsar(Scalar t);
  void boundary_absorbing();
};

}  // namespace Coffee
