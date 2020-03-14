#include "sim_data.h"
#include "sim_env.h"

namespace Coffee {

sim_data::sim_data(const sim_environment& e) : env(e) {
  E = vector_field<Scalar>(env.grid());
  B = vector_field<Scalar>(env.grid());
  B0 = vector_field<Scalar>(env.grid());
  P = multi_array<Scalar>(env.grid().extent());
  divB = multi_array<Scalar>(env.grid().extent());
  divE = multi_array<Scalar>(env.grid().extent());
  Stagger st_e[3] = {Stagger(0b110), Stagger(0b101), Stagger(0b011)};
  Stagger st_b[3] = {Stagger(0b001), Stagger(0b010), Stagger(0b100)};
  B.set_stagger(st_b);
  B0.set_stagger(st_b);
  // Although this is the default, we explicitly set it here, so that it sticks
  // even if we change the default.
  E.set_stagger(st_e);
}

sim_data::~sim_data() {}

void
sim_data::sync_to_host() {
  E.sync_to_host();
  B.sync_to_host();
  B0.sync_to_host();
  P.sync_to_host();
  divB.sync_to_host();
  divE.sync_to_host();
}

void
sim_data::sync_to_device() {
  E.sync_to_device();
  B.sync_to_device();
  B0.sync_to_device();
  P.sync_to_device();
  divB.sync_to_device();
  divE.sync_to_device();
}

}  // namespace Coffee
