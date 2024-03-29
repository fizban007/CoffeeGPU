#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "fields.h"
#include "grid.h"

namespace Coffee {

class sim_environment;

class sim_data {
 public:
  vector_field<Scalar> E, B, B0;
  multi_array<Scalar> P, divB, divE;
  multi_array<Scalar> dU_EgtB, dU_Epar, dU_KO;
  multi_array<Scalar> dU_EgtB_cum, dU_Epar_cum, dU_KO_cum;
  const sim_environment& env;

  sim_data(const sim_environment& env);
  ~sim_data();

  void sync_to_host();
  void sync_to_device();
};

}  // namespace Coffee

#endif  // _SIM_DATA_H_
