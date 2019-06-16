#include "sim_data.h"
#include "sim_env.h"

namespace Coffee {

sim_data::sim_data(const sim_environment& e) : env(e) {
  E = vector_field<Scalar>(e.grid());
  B = vector_field<Scalar>(e.grid());
}

sim_data::~sim_data() {}

}  // namespace Coffee
