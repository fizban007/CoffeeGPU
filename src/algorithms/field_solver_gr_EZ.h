#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver_gr_EZ {
 public:
  vector_field<Scalar> Dtmp, Btmp, dD, dB, Ed, Hd, Bbg;
  multi_array<Scalar> dP, Ptmp;

  field_solver_gr_EZ(sim_data& mydata, sim_environment& env);
  ~field_solver_gr_EZ();

  void evolve_fields(Scalar time);

  Scalar total_energy(vector_field<Scalar>& f);

 private:
  sim_data& m_data;
  sim_environment& m_env;

  void check_eGTb();
  void clean_epar();

  void rk_step(Scalar As, Scalar Bs);
  void Kreiss_Oliger();

  void boundary_absorbing();

  void get_Ed();
  void get_Hd();
};

}  // namespace Coffee
