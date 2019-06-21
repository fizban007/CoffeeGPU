#include "sim_data.h"
#include "sim_env.h"

namespace Coffee {

sim_data::sim_data(const sim_environment& e) : env(e) {
  E = vector_field<Scalar>(env.grid());
  B = vector_field<Scalar>(env.grid());
  B0 = vector_field<Scalar>(env.grid());
  Stagger st_e[3] = {Stagger(0b110), Stagger(0b101), Stagger(0b011)};
  Stagger st_b[3] = {Stagger(0b001), Stagger(0b010), Stagger(0b100)};
  B.set_stagger(st_b);
  B0.set_stagger(st_b);
  // Although this is the default, we explicitly set it here, so that it sticks
  // even if we change the default.
  E.set_stagger(st_e);
}

sim_data::~sim_data() {}

}  // namespace Coffee
