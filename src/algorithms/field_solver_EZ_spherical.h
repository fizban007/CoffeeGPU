#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver_EZ_spherical {
 public:
  vector_field<Scalar> Etmp, Btmp, dE, dB;
  multi_array<Scalar> dP, Ptmp;

  field_solver_EZ_spherical(sim_data& mydata, sim_environment& env);
  ~field_solver_EZ_spherical();

  void evolve_fields(Scalar time);

  Scalar total_energy(vector_field<Scalar>& f);

 private:
  sim_data& m_data;
  sim_environment& m_env;

  void check_eGTb();
  void clean_epar();

  void rk_step(Scalar As, Scalar Bs);
  void Kreiss_Oliger();

  void boundary_axis();
  void boundary_pulsar(Scalar t);
  void boundary_absorbing();
};

}  // namespace Coffee