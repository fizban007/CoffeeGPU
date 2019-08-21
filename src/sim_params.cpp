#include "sim_params.h"
#include "cpptoml.h"

namespace Coffee {

sim_params
parse_config(const std::string& filename) {
  auto config = cpptoml::parse_file(filename);
  sim_params defaults, result;

  result.dt = config->get_as<double>("dt").value_or(defaults.dt);
  result.max_steps = config->get_as<uint64_t>("max_steps")
                         .value_or(defaults.max_steps);
  result.data_interval = config->get_as<int>("data_interval")
      .value_or(defaults.data_interval);
  result.downsample = config->get_as<int>("downsample")
      .value_or(defaults.downsample);

  auto periodic_boundary =
      config->get_array_of<bool>("periodic_boundary");
  if (periodic_boundary) {
    result.periodic_boundary[0] = (*periodic_boundary)[0];
    result.periodic_boundary[1] = (*periodic_boundary)[1];
    result.periodic_boundary[2] = (*periodic_boundary)[2];
  }

  auto guard = config->get_array_of<int64_t>("guard");
  if (guard)
    for (int i = 0; i < 3; i++) result.guard[i] = (*guard)[i];

  auto N = config->get_array_of<int64_t>("N");
  if (N)
    for (int i = 0; i < 3; i++) result.N[i] = (*N)[i];

  auto lower = config->get_array_of<double>("lower");
  if (lower)
    for (int i = 0; i < 3; i++) result.lower[i] = (*lower)[i];

  auto size = config->get_array_of<double>("size");
  if (size)
    for (int i = 0; i < 3; i++) result.size[i] = (*size)[i];

  auto nodes = config->get_array_of<double>("nodes");
  if (nodes)
    for (int i = 0; i < 3; i++) result.nodes[i] = (*nodes)[i];

  result.a = config->get_as<double>("a").value_or(defaults.a);

  result.calc_current = config->get_as<bool>("calc_current")
      .value_or(defaults.calc_current);
  result.clean_ep = config->get_as<bool>("clean_ep")
      .value_or(defaults.clean_ep);
  result.check_egb = config->get_as<bool>("check_egb")
      .value_or(defaults.check_egb);

  auto pml = config->get_array_of<int64_t>("pml");
  if (pml)
    for (int i = 0; i < 3; i++) result.pml[i] = (*pml)[i];

  result.pmllen = config->get_as<int>("pmllen").value_or(defaults.pmllen);
  result.sigpml = config->get_as<double>("sigpml").value_or(defaults.sigpml);

  return result;
}

}  // namespace Coffee
