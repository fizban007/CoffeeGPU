#ifndef _SIM_DATA_H_
#define _SIM_DATA_H_

#include "fields.h"
#include "grid.h"

namespace Coffee {

class sim_environment;

class sim_data {
 public:
  vector_field<Scalar> E, B, B0;
  multi_array<Scalar> P;
  const sim_environment& env;

  sim_data(const sim_environment& env);
  ~sim_data();

  void sync_to_host();
};

}  // namespace Coffee

#endif  // _SIM_DATA_H_
