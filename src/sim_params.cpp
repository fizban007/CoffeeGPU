#include "sim_params.h"

#include "cpptoml.h"

namespace Coffee {

sim_params parse_config(const std::string& filename) {
  auto config = cpptoml::parse_file(filename);
  sim_params defaults, result;

  result.dt = config->get_as<double>("dt").value_or(defaults.dt);
  result.max_steps =
      config->get_as<uint64_t>("max_steps").value_or(defaults.max_steps);
  result.data_interval =
      config->get_as<int>("data_interval").value_or(defaults.data_interval);
  result.snapshot_interval = config->get_as<int>("snapshot_interval")
                                 .value_or(defaults.snapshot_interval);
  result.downsample =
      config->get_as<int>("downsample").value_or(defaults.downsample);

  result.slice_x = config->get_as<bool>("slice_x").value_or(defaults.slice_x);
  result.slice_y = config->get_as<bool>("slice_y").value_or(defaults.slice_y);
  result.slice_z = config->get_as<bool>("slice_z").value_or(defaults.slice_z);
  result.slice_xy = config->get_as<bool>("slice_xy").value_or(defaults.slice_xy);
  result.slice_x_pos = config->get_as<double>("slice_x_pos").value_or(defaults.slice_x_pos);
  result.slice_y_pos = config->get_as<double>("slice_y_pos").value_or(defaults.slice_y_pos);
  result.slice_z_pos = config->get_as<double>("slice_z_pos").value_or(defaults.slice_z_pos);
  result.slice_interval =
      config->get_as<int>("slice_interval").value_or(defaults.slice_interval);
  result.downsample_2d =
      config->get_as<int>("downsample_2d").value_or(defaults.downsample_2d);

  auto periodic_boundary = config->get_array_of<bool>("periodic_boundary");
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

  result.shift_ghost =
      config->get_as<int>("shift_ghost").value_or(defaults.shift_ghost);

  result.a = config->get_as<double>("a").value_or(defaults.a);

  result.calc_current =
      config->get_as<bool>("calc_current").value_or(defaults.calc_current);
  result.clean_ep =
      config->get_as<bool>("clean_ep").value_or(defaults.clean_ep);
  result.check_egb =
      config->get_as<bool>("check_egb").value_or(defaults.check_egb);

  auto pml = config->get_array_of<int64_t>("pml");
  if (pml)
    for (int i = 0; i < 3; i++) result.pml[i] = (*pml)[i];

  result.pmllen = config->get_as<int>("pmllen").value_or(defaults.pmllen);
  result.sigpml = config->get_as<double>("sigpml").value_or(defaults.sigpml);
  result.damp_gamma =
      config->get_as<double>("damp_gamma").value_or(defaults.damp_gamma);
  result.use_edotb_damping = config->get_as<bool>("use_edotb_damping")
                                 .value_or(defaults.use_edotb_damping);

  result.divB_clean =
      config->get_as<bool>("divB_clean").value_or(defaults.divB_clean);
  result.ch2 = config->get_as<double>("ch2").value_or(defaults.ch2);
  result.tau = config->get_as<double>("tau").value_or(defaults.tau);
  result.KOeps = config->get_as<double>("KOeps").value_or(defaults.KOeps);
  result.KO_geometry =
      config->get_as<bool>("KO_geometry").value_or(defaults.KO_geometry);

  result.pulsar = config->get_as<bool>("pulsar").value_or(defaults.pulsar);
  result.radius = config->get_as<double>("radius").value_or(defaults.radius);
  result.omega = config->get_as<double>("omega").value_or(defaults.omega);
  result.b0 = config->get_as<double>("b0").value_or(defaults.b0);
  // result.alpha =
  // config->get_as<double>("alpha").value_or(defaults.alpha);
  result.p1 = config->get_as<double>("p1").value_or(defaults.p1);
  result.p2 = config->get_as<double>("p2").value_or(defaults.p2);
  result.p3 = config->get_as<double>("p3").value_or(defaults.p3);
  result.q11 = config->get_as<double>("q11").value_or(defaults.q11);
  result.q12 = config->get_as<double>("q12").value_or(defaults.q12);
  result.q13 = config->get_as<double>("q13").value_or(defaults.q13);
  result.q22 = config->get_as<double>("q22").value_or(defaults.q22);
  result.q23 = config->get_as<double>("q23").value_or(defaults.q23);
  result.q_offset_x =
      config->get_as<double>("q_offset_x").value_or(defaults.q_offset_x);
  result.q_offset_y =
      config->get_as<double>("q_offset_y").value_or(defaults.q_offset_y);
  result.q_offset_z =
      config->get_as<double>("q_offset_z").value_or(defaults.q_offset_z);

  result.tp_start =
      config->get_as<double>("tp_start").value_or(defaults.tp_start);
  result.tp_end = config->get_as<double>("tp_end").value_or(defaults.tp_end);
  result.rpert1 = config->get_as<double>("rpert1").value_or(defaults.rpert1);
  result.rpert2 = config->get_as<double>("rpert2").value_or(defaults.rpert2);
  result.dw0 = config->get_as<double>("dw0").value_or(defaults.dw0);
  result.nT = config->get_as<double>("nT").value_or(defaults.nT);
  result.shear = config->get_as<double>("shear").value_or(defaults.shear);
  result.ph1 = config->get_as<double>("ph1").value_or(defaults.ph1);
  result.ph2 = config->get_as<double>("ph2").value_or(defaults.ph2);
  result.dph = config->get_as<double>("dph").value_or(defaults.dph);
  result.theta0 = config->get_as<double>("theta0").value_or(defaults.theta0);
  result.drpert = config->get_as<double>("drpert").value_or(defaults.drpert);

  result.tp_start1 =
      config->get_as<double>("tp_start1").value_or(defaults.tp_start1);
  result.tp_end1 = config->get_as<double>("tp_end1").value_or(defaults.tp_end1);
  result.rpert11 = config->get_as<double>("rpert11").value_or(defaults.rpert11);
  result.rpert21 = config->get_as<double>("rpert21").value_or(defaults.rpert21);
  result.dw1 = config->get_as<double>("dw1").value_or(defaults.dw1);
  result.nT1 = config->get_as<double>("nT1").value_or(defaults.nT1);
  result.shear1 = config->get_as<double>("shear1").value_or(defaults.shear1);
  result.ph11 = config->get_as<double>("ph11").value_or(defaults.ph11);
  result.ph21 = config->get_as<double>("ph21").value_or(defaults.ph21);
  result.dph1 = config->get_as<double>("dph1").value_or(defaults.dph1);

  result.pert_type = config->get_as<int>("pert_type").value_or(defaults.pert_type);

  return result;
}

}  // namespace Coffee
