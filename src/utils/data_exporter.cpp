#include "data_exporter.h"
#include "H5Cpp.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/sim_data.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "sim_env.h"
#include "sim_params.h"

#define ADD_GRID_OUTPUT(input, name, func, file)               \
  add_grid_output(input, name,                                 \
                  [](sim_data & data, multi_array<Scalar> & p, \
                     Index idx, Index idx_out) func,           \
                  file)

using namespace H5;

namespace Coffee {

template <typename Func>
void
sample_grid_quantity2d(sim_data& data, const Grid& g, int downsample,
                       multi_array<Scalar>& result, Func f) {
  const auto& ext = g.extent();
  for (int j = 0; j < ext.height(); j++) {
    for (int i = 0; i < ext.width(); i++) {
      Index idx_out(i, j, 0);
      Index idx_data(i * downsample + g.guard[0],
                     j * downsample + g.guard[1], 0);
      f(data, result, idx_data, idx_out);
    }
  }
}

template <typename Func>
void
sample_grid_quantity3d(sim_data& data, const Grid& g, int downsample,
                       multi_array<Scalar>& result, Func f) {
  const auto& ext = g.extent();
  for (int k = 0; k < ext.depth(); k++) {
    for (int j = 0; j < ext.height(); j++) {
      for (int i = 0; i < ext.width(); i++) {
        Index idx_out(i, j, k);
        Index idx_data(i * downsample + g.guard[0],
                       j * downsample + g.guard[1],
                       k * downsample + g.guard[2]);
        f(data, result, idx_data, idx_out);
      }
    }
  }
}

data_exporter::data_exporter(sim_environment& env, uint32_t& timestep)
    : m_env(env) {
  auto& grid = m_env.grid();
  auto ext = grid.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_data.resize(ext.width() / d, ext.height() / d,
                       ext.depth() / d);

  outputDirectory = "./Data/";
}

data_exporter::~data_exporter() {}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  if (m_thread && m_thread->joinable()) m_thread->join();

  data.sync_to_host();

  // Launch a new thread to handle the field output
  // m_thread.reset(new std::thread(&data_exporter::write_field_output,
  //                                this, std::ref(data), timestep,
  //                                time));
  write_field_output(data, timestep, time);
  std::cout << "Output written!" << std::endl;
}

void
data_exporter::sync() {
  if (m_thread && m_thread->joinable()) m_thread->join();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  H5File datafile(
      outputDirectory + std::string("fld") +
          std::to_string(timestep / m_env.params().data_interval) +
          std::string(".h5"),
      H5F_ACC_TRUNC);
  std::cout << outputDirectory + std::string("fld") +
                   std::to_string(timestep /
                                  m_env.params().data_interval) +
                   std::string(".h5")
            << std::endl;
  // add_grid_output(
  //     data, "E1",
  //     [](sim_data& data, multi_array<Scalar>& p, Index idx,
  //        Index idx_out) {
  //       p(idx_out) = data.E(0, idx) + data.Ebg(0, idx);
  //     },
  //     datafile);
  ADD_GRID_OUTPUT(
      data, "E1", { p(idx_out) = data.E(0, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "E2", { p(idx_out) = data.E(1, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "E3", { p(idx_out) = data.E(2, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "B1", { p(idx_out) = data.B(0, idx) + data.B0(0, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "B2", { p(idx_out) = data.B(1, idx) + data.B0(1, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "B3", { p(idx_out) = data.B(2, idx) + data.B0(2, idx); },
      datafile);

  datafile.close();
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, H5File& file) {
  if (data.env.grid().dim() == 3) {
    sample_grid_quantity3d(data, m_env.grid(),
                           m_env.params().downsample, tmp_grid_data, f);

    // Actually write the temp array to hdf
    hsize_t dims[3] = {(uint32_t)tmp_grid_data.width(),
                       (uint32_t)tmp_grid_data.height(),
                       (uint32_t)tmp_grid_data.depth()};
    DataSpace dataspace(3, dims);
    DataSet dataset =
        file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
    dataset.write(tmp_grid_data.host_ptr(), PredType::NATIVE_FLOAT);
  } else if (data.env.grid().dim() == 2) {
    sample_grid_quantity2d(data, m_env.grid(),
                           m_env.params().downsample, tmp_grid_data, f);

    // Actually write the temp array to hdf
    hsize_t dims[2] = {(uint32_t)tmp_grid_data.width(),
                       (uint32_t)tmp_grid_data.height()};
    DataSpace dataspace(2, dims);
    DataSet dataset =
        file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
    dataset.write(tmp_grid_data.host_ptr(), PredType::NATIVE_FLOAT);
  }
}

}  // namespace Coffee
