#include "data/sim_data.h"
#include "sim_env.h"

namespace Coffee {

class field_solver {
 public:
  sim_data& m_data;

  vector_field<Scalar> En, Bn, dE, dB;
  multi_array<Scalar> rho;

  field_solver(sim_data& mydata);
  ~field_solver();

  void evolve_fields();

 private:
  void copy_fields();

  void rk_push();
  void rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3);

  void clean_epar();
};

}  // namespace Coffee
