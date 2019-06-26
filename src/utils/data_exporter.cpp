#include "data_exporter.h"
// #include "H5Cpp.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/sim_data.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "sim_env.h"
#include "sim_params.h"
#include <iomanip>
#include <boost/filesystem.hpp>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#define ADD_GRID_OUTPUT(input, name, func, file)              \
  add_grid_output(input, name,                                \
                  [](sim_data & data, multi_array<float> & p, \
                     Index idx, Index idx_out) func,          \
                  file)

// using namespace H5;
using namespace HighFive;

namespace Coffee {

template <typename Func>
void
sample_grid_quantity2d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       boost::multi_array<float, 3>& out, Func f) {
  const auto& ext = g.extent();
  for (int j = 0; j < ext.height(); j++) {
    for (int i = 0; i < ext.width(); i++) {
      Index idx_out(i, j, 0);
      Index idx_data(i * downsample + g.guard[0],
                     j * downsample + g.guard[1], 0);
      f(data, result, idx_data, idx_out);

      out[0][j][i] = result(i, j, 0);
    }
  }
}

template <typename Func>
void
sample_grid_quantity3d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       boost::multi_array<float, 3>& out, Func f) {
  const auto& ext = result.extent();
  for (int k = 0; k < ext.depth(); k++) {
    for (int j = 0; j < ext.height(); j++) {
      for (int i = 0; i < ext.width(); i++) {
        Index idx_out(i, j, k);
        Index idx_data(i * downsample + g.guard[0],
                       j * downsample + g.guard[1],
                       k * downsample + g.guard[2]);
        // std::cout << idx_out << ", " << idx_data << std::endl;
        f(data, result, idx_data, idx_out);

        out[k][j][i] = result(i, j, k);
      }
    }
  }
}

data_exporter::data_exporter(sim_environment& env, uint32_t& timestep)
    : m_env(env) {
  auto& grid = m_env.grid();
  auto ext = grid.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_data = multi_array<float>(ext.width() / d, ext.height() / d,
                                     ext.depth() / d);
  m_output.resize(
      boost::extents[tmp_grid_data.depth()][tmp_grid_data.height()]
                    [tmp_grid_data.width()]);
  outputDirectory = "./Data/";
  m_thread = nullptr;
  boost::filesystem::path outPath(outputDirectory);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  std::string path = outputDirectory + "config.toml";
  boost::filesystem::copy_file(
      "config.toml", path,
      boost::filesystem::copy_option::overwrite_if_exists);
}

data_exporter::~data_exporter() {}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  if (m_thread != nullptr && m_thread->joinable()) m_thread->join();

  data.sync_to_host();

  // Launch a new thread to handle the field output
  m_thread.reset(new std::thread(&data_exporter::write_field_output,
                                 this, std::ref(data), timestep, time));
  // write_field_output(data, timestep, time);
  std::cout << "Output written!" << std::endl;
}

void
data_exporter::sync() {
  // std::cout << m_thread->joinable() << std::endl;
  if (m_thread != nullptr && m_thread->joinable()) m_thread->join();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0')
     << timestep / m_env.params().data_interval;
  std::string num = ss.str();
  File datafile(
      outputDirectory + std::string("fld.") + num + std::string(".h5"),
      File::ReadWrite | File::Create | File::Truncate,
      MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));
  // H5F_ACC_TRUNC);
  // H5F_ACC_RDWR);
  // add_grid_output(
  //     data, "E1",
  //     [](sim_data& data, multi_array<Scalar>& p, Index idx,
  //        Index idx_out) {
  //       p(idx_out) = data.E(0, idx) + data.Ebg(0, idx);
  //     },
  //     datafile);
  ADD_GRID_OUTPUT(
      data, "Ex",
      {
        p(idx_out) = 0.25 * (data.E(0, idx) +
                             data.E(0, idx.x, idx.y + 1, idx.z) +
                             data.E(0, idx.x, idx.y, idx.z + 1) +
                             data.E(0, idx.x, idx.y + 1, idx.z + 1));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "Ey",
      {
        p(idx_out) = 0.25 * (data.E(1, idx) +
                             data.E(1, idx.x + 1, idx.y, idx.z) +
                             data.E(1, idx.x, idx.y, idx.z + 1) +
                             data.E(1, idx.x + 1, idx.y, idx.z + 1));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "Ez",
      {
        p(idx_out) = 0.25 * (data.E(2, idx) +
                             data.E(2, idx.x + 1, idx.y, idx.z) +
                             data.E(2, idx.x, idx.y + 1, idx.z) +
                             data.E(2, idx.x + 1, idx.y + 1, idx.z));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "Bx",
      {
        p(idx_out) =
            0.5 * (data.B(0, idx) + data.B(0, idx.x - 1, idx.y, idx.z));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "By",
      {
        p(idx_out) =
            0.5 * (data.B(1, idx) + data.B(1, idx.x, idx.y - 1, idx.z));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "Bz",
      {
        p(idx_out) =
            0.5 * (data.B(2, idx) + data.B(2, idx.x, idx.y, idx.z - 1));
      },
      datafile);
  ADD_GRID_OUTPUT(
      data, "EdotB",
      {
        p(idx_out) = data.E(0, idx) * data.B(0, idx) +
                     data.E(1, idx) * data.B(1, idx) +
                     data.E(2, idx) * data.B(2, idx);
      },
      datafile);

  // datafile.close();
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, File& file) {
  // if (data.env.grid().dim() == 3) {
  int downsample = m_env.params().downsample;
  sample_grid_quantity3d(data, m_env.grid(), downsample, tmp_grid_data,
                         m_output, f);

  std::vector<size_t> dims(3);
  for (int i = 0; i < 3; i++) {
    dims[i] = m_env.params().N[2 - i];
    if (dims[i] > downsample) dims[i] /= downsample;
  }
  // Actually write the temp array to hdf
  DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));

  std::vector<size_t> out_dim(3);
  std::vector<size_t> offsets(3);
  for (int i = 0; i < 3; i++) {
    offsets[i] = m_env.grid().offset[2 - i] / downsample;
    out_dim[i] = tmp_grid_data.extent()[2 - i];
  }
  dataset.select(offsets, out_dim).write(m_output);
  // } else if (data.env.grid().dim() == 2) {
  //   sample_grid_quantity2d(data, m_env.grid(),
  //                          m_env.params().downsample, tmp_grid_data,
  //                          m_output, f);

  //   // Actually write the temp array to hdf
  //   DataSet dataset =
  //       file.createDataSet<float>(name, DataSpace::From(m_output));
  //   dataset.write(m_output);
  // }
}

}  // namespace Coffee
