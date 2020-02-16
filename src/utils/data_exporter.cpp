#include "data_exporter.h"
// #include "H5Cpp.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/sim_data.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "hdf_wrapper_impl.hpp"
#include "sim_env.h"
#include "sim_params.h"
// #include "utils/nvproftool.h"
//#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
//#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <iomanip>

#define ADD_GRID_OUTPUT(input, name, func, file)              \
  add_grid_output(input, name,                                \
                  [](sim_data & data, multi_array<float> & p, \
                     Index idx, Index idx_out) func,          \
                  file)

// using namespace H5;

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
  for (int i = 0; i < 3; i++) {
    if (ext[i] > d) ext[i] /= d;
  }
  tmp_grid_data =
      multi_array<float>(ext.width(), ext.height(), ext.depth());
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
  // RANGE_PUSH("Data output", CLR_BLUE);
  // if (m_thread != nullptr && m_thread->joinable()) m_thread->join();

  data.sync_to_host();

  write_field_output(data, timestep, time);
  // std::cout << "Output written!" << std::endl;
  // RANGE_POP;
}

void
data_exporter::sync() {
  // std::cout << m_thread->joinable() << std::endl;
  // if (m_thread != nullptr && m_thread->joinable()) m_thread->join();
}

void
data_exporter::save_snapshot(const std::string& filename,
                             sim_data& data, uint32_t step,
                             Scalar time) {
  // Do this regardless of cpu or gpu
  data.sync_to_host();

  // Open the snapshot file for writing
  // std::string filename = outputDirectory +
  // std::string("snapshot.h5");

  auto datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

  int rank = m_env.rank();
  auto& params = m_env.params();
  auto& grid = m_env.grid();
  Extent ext_total, ext;
  Index idx_dst, idx_src;
  for (int i = 0; i < grid.dim(); i++) {
    ext_total[i] = params.N[i] + 2 * params.guard[i];
    ext[i] = grid.reduced_dim(i);
    idx_dst[i] = grid.offset[i];
    idx_src[i] = 0;
    if (idx_dst[i] > 0) {
      idx_dst[i] += grid.guard[i];
      idx_src[i] += grid.guard[i];
    }
    if (m_env.neighbor_left(i) == NEIGHBOR_NULL) {
      ext[i] += grid.guard[i];
    }
    if (m_env.neighbor_right(i) == NEIGHBOR_NULL) {
      ext[i] += grid.guard[i];
    }
  }

  // Write to the datafile
  datafile.write_parallel(data.E.data(0), ext_total, idx_dst, ext,
                          idx_src, "Ex");
  datafile.write_parallel(data.E.data(1), ext_total, idx_dst, ext,
                          idx_src, "Ey");
  datafile.write_parallel(data.E.data(2), ext_total, idx_dst, ext,
                          idx_src, "Ez");
  datafile.write_parallel(data.B.data(0), ext_total, idx_dst, ext,
                          idx_src, "Bx");
  datafile.write_parallel(data.B.data(1), ext_total, idx_dst, ext,
                          idx_src, "By");
  datafile.write_parallel(data.B.data(2), ext_total, idx_dst, ext,
                          idx_src, "Bz");
  datafile.write_parallel(data.B0.data(0), ext_total, idx_dst, ext,
                          idx_src, "B0x");
  datafile.write_parallel(data.B0.data(1), ext_total, idx_dst, ext,
                          idx_src, "B0y");
  datafile.write_parallel(data.B0.data(2), ext_total, idx_dst, ext,
                          idx_src, "B0z");
  datafile.write_parallel(data.P, ext_total, idx_dst, ext, idx_src,
                          "P");
  datafile.write_parallel(data.divE, ext_total, idx_dst, ext, idx_src,
                          "divE");
  datafile.write_parallel(data.divB, ext_total, idx_dst, ext, idx_src,
                          "divB");

  // if (rank == 0) {
  datafile.write(step, "step");
  datafile.write(time, "time");
  // }
  datafile.close();
}

void
data_exporter::load_snapshot(const std::string& filename,
                             sim_data& data, uint32_t& step,
                             Scalar& time) {
  // Check whether filename exists
  if (!boost::filesystem::exists(filename)) {
    std::cout
        << "Can't find restart file, proceeding without loading it!"
        << std::endl;
    return;
  }

  // Open the snapshot file for reading
  H5File datafile(filename, H5OpenMode::read_parallel);

  int rank = m_env.rank();
  auto& params = m_env.params();
  auto& grid = m_env.grid();
  Extent ext;
  Index idx_dst, idx_src;
  for (int i = 0; i < grid.dim(); i++) {
    // ext_total[i] = params.N[i] + 2 * params.guard[i];
    ext[i] = grid.reduced_dim(i);
    idx_src[i] = grid.offset[i];
    idx_dst[i] = 0;
    if (idx_src[i] > 0) {
      idx_src[i] += grid.guard[i];
      idx_dst[i] += grid.guard[i];
    }
    if (m_env.neighbor_left(i) == NEIGHBOR_NULL) {
      ext[i] += grid.guard[i];
    }
    if (m_env.neighbor_right(i) == NEIGHBOR_NULL) {
      ext[i] += grid.guard[i];
    }
  }

  // Write to the datafile
  datafile.read_subset(data.E.data(0), "Ex", idx_src, ext, idx_dst);
  datafile.read_subset(data.E.data(1), "Ey", idx_src, ext, idx_dst);
  datafile.read_subset(data.E.data(2), "Ez", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(0), "Bx", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(1), "By", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(2), "Bz", idx_src, ext, idx_dst);
  datafile.read_subset(data.B0.data(0), "B0x", idx_src, ext, idx_dst);
  datafile.read_subset(data.B0.data(1), "B0y", idx_src, ext, idx_dst);
  datafile.read_subset(data.B0.data(2), "B0z", idx_src, ext, idx_dst);
  datafile.read_subset(data.P, "P", idx_src, ext, idx_dst);
  datafile.read_subset(data.divE, "divE", idx_src, ext, idx_dst);
  datafile.read_subset(data.divB, "divB", idx_src, ext, idx_dst);

  step = datafile.read_scalar<uint32_t>("step");
  time = datafile.read_scalar<Scalar>("time");
  datafile.close();
  data.sync_to_device();

  m_env.send_guard_cells(data);
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0')
     << timestep / m_env.params().data_interval;
  std::string num = ss.str();
  std::string filename =
      outputDirectory + std::string("fld.") + num + std::string(".h5");

  auto datafile = hdf_create(filename, H5CreateMode::trunc_parallel);
  // hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  // H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  // hid_t datafile =
  //     H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
  //     plist_id);
  // H5Pclose(plist_id);

  add_grid_output(data.E.data(0), "Ex", data.E.stagger(0), datafile);
  add_grid_output(data.E.data(1), "Ey", data.E.stagger(1), datafile);
  add_grid_output(data.E.data(2), "Ez", data.E.stagger(2), datafile);
  add_grid_output(data.B.data(0), "Bx", data.B.stagger(0), datafile);
  add_grid_output(data.B.data(1), "By", data.B.stagger(1), datafile);
  add_grid_output(data.B.data(2), "Bz", data.B.stagger(2), datafile);
  add_grid_output(data.B0.data(0), "Jx", data.B0.stagger(0), datafile);
  add_grid_output(data.B0.data(1), "Jy", data.B0.stagger(1), datafile);
  add_grid_output(data.B0.data(2), "Jz", data.B0.stagger(2), datafile);
  add_grid_output(data.P, "P", Stagger(0b111), datafile);
  add_grid_output(data.divB, "divB", Stagger(0b111), datafile);
  add_grid_output(data.divE, "divE", Stagger(0b111), datafile);

  datafile.close();
  // H5Fclose(datafile);
}

// template <typename Func>
// void
// data_exporter::add_grid_output(sim_data& data, const std::string&
// name,
//                                Func f, hid_t file) {
//   // if (data.env.grid().dim() == 3) {
//   int downsample = m_env.params().downsample;
//   sample_grid_quantity3d(data, m_env.grid(), downsample,
//   tmp_grid_data,
//                          m_output, f);

//   std::vector<size_t> dims(3);
//   for (int i = 0; i < 3; i++) {
//     dims[i] = m_env.params().N[2 - i];
//     if (dims[i] > downsample) dims[i] /= downsample;
//   }
//   // Actually write the temp array to hdf
//   DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));

//   std::vector<size_t> out_dim(3);
//   std::vector<size_t> offsets(3);
//   for (int i = 0; i < 3; i++) {
//     offsets[i] = m_env.grid().offset[2 - i] / downsample;
//     out_dim[i] = tmp_grid_data.extent()[2 - i];
//   }
//   dataset.select(offsets, out_dim).write(m_output);
//   // } else if (data.env.grid().dim() == 2) {
//   //   sample_grid_quantity2d(data, m_env.grid(),
//   //                          m_env.params().downsample,
//   tmp_grid_data,
//   //                          m_output, f);

//   //   // Actually write the temp array to hdf
//   //   DataSet dataset =
//   //       file.createDataSet<float>(name,
//   DataSpace::From(m_output));
//   //   dataset.write(m_output);
//   // }
// }

void
data_exporter::add_grid_output(multi_array<Scalar>& array,
                               const std::string& name, Stagger stagger,
                               // HighFive::File& file) {
                               H5File& file) {
  int downsample = m_env.params().downsample;
  array.downsample(downsample, tmp_grid_data,
                   Index(m_env.grid().guard[0], m_env.grid().guard[1],
                         m_env.grid().guard[2]),
                   stagger, m_output.data());
  auto& grid = m_env.grid();
  Extent dims;
  Index offsets;
  for (int i = 0; i < grid.dim(); i++) {
    dims[i] = m_env.params().N[i];
    if (dims[i] > downsample) dims[i] /= downsample;
    offsets[i] = m_env.grid().offset[i] / downsample;
  }

  file.write_parallel(tmp_grid_data, dims, offsets,
                      tmp_grid_data.extent(), Index(0, 0, 0), name);
}

// void
// data_exporter::write_multi_array(const multi_array<Scalar>& array,
//                                  const std::string& name,
//                                  hid_t file_id) {
//   auto& grid = m_env.grid();
//   hsize_t dims[3];
//   for (int i = 0; i < grid.dim(); i++) {
//     dims[i] = array.extent()[grid.dim() - 1 - i];
//     // if (dims[i] > downsample) dims[i] /= downsample;
//   }
//   // Actually write the temp array to hdf
//   auto filespace = H5Screate_simple(m_env.grid().dim(), dims, NULL);

//   auto memspace = H5Screate_simple(m_env.grid().dim(), dims, NULL);
//   // dataset.select(offsets, out_dim).write(m_output);
//   auto plist_id = H5Pcreate(H5P_DATASET_CREATE);
//   H5Pset_chunk(plist_id, m_env.grid().dim(), dims);
//   auto dset_id =
//       H5Dcreate(file_id, name.c_str(), H5T_NATIVE_FLOAT, filespace,
//                 H5P_DEFAULT, plist_id, H5P_DEFAULT);
//   H5Pclose(plist_id);
//   H5Sclose(filespace);

//   plist_id = H5Pcreate(H5P_DATASET_XFER);
//   H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

//   auto status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace,
//   H5S_ALL,
//                          plist_id, tmp_grid_data.host_ptr());

//   H5Dclose(dset_id);
//   // H5Sclose(filespace);
//   H5Sclose(memspace);
//   H5Pclose(plist_id);
// }

}  // namespace Coffee
