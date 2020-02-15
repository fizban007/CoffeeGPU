#ifndef _HDF_WRAPPER_IMPL_H_
#define _HDF_WRAPPER_IMPL_H_

#include "hdf_wrapper.h"

namespace Coffee {

template <typename T>
void
H5File::write(T value, const std::string& name) {
  auto dataspace_id = H5Screate(H5S_SCALAR);
  auto dataset_id =
      H5Dcreate(m_file_id, name.c_str(), h5datatype<T>(), dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  auto status = H5Dwrite(dataset_id, h5datatype<T>(), H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, &value);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}

template <typename T>
void
H5File::write(const multi_array<T>& array, const std::string& name) {
  hsize_t dims[3];
  for (int i = 0; i < array.dim(); i++) {
    dims[i] = array.extent()[array.dim() - 1 - i];
  }
  auto dataspace_id = H5Screate_simple(array.dim(), dims, NULL);
  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), dataspace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  auto status = H5Dwrite(dataset_id, h5datatype<T>(), H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, array.host_ptr());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}

template <typename T>
void
H5File::write_parallel(const multi_array<T>& array,
                       const Extent& ext_total, const Index& idx_dst,
                       const Extent& ext, const Index& idx_src,
                       const std::string& name) {
  hsize_t dims[3], array_dims[3];
  for (int i = 0; i < ext_total.dim(); i++) {
    dims[i] = ext_total[ext_total.dim() - 1 - i];
    array_dims[i] = array.extent()[array.dim() - 1 - i];
  }
  auto filespace_id = H5Screate_simple(ext_total.dim(), dims, NULL);
  auto memspace_id = H5Screate_simple(ext.dim(), array_dims, NULL);

  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), filespace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t offsets[3], offsets_l[3], out_dim[3], count[3], stride[3];
  for (int i = 0; i < 3; i++) {
    count[i] = 1;
    stride[i] = 1;
    offsets[i] = idx_dst[ext_total.dim() - i - 1];
    offsets_l[i] = idx_src[ext_total.dim() - i - 1];
    out_dim[i] = ext[ext_total.dim() - i - 1];
  }
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offsets, stride,
                      count, out_dim);
  H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  auto status = H5Dwrite(dataset_id, h5datatype<T>(), memspace_id,
                         filespace_id, plist_id, array.host_ptr());

  H5Dclose(dataset_id);
  H5Sclose(filespace_id);
  H5Sclose(memspace_id);
  H5Pclose(plist_id);
}

template <typename T>
multi_array<T>
H5File::read(const std::string& name) {
  Extent ext;

  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);

  hsize_t dims[3];
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  int dim = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  for (int i = 0; i < dim; i++) {
    ext[i] = dims[dim - i - 1];
  }

  multi_array<T> result(ext);
  H5Dread(dataset, h5datatype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
          result.host_ptr());

  H5Dclose(dataset);
  H5Sclose(dataspace);
  return result;
}

template <typename T>
T
H5File::read_scalar(const std::string& name) {
  T result;
  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */

  H5Dread(dataset, h5datatype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
          &result);

  H5Dclose(dataset);
  H5Sclose(dataspace);
  return result;
}

template <typename T>
void
H5File::read_subset(multi_array<T>& array, const std::string& name,
                    const Index& idx_src, const Extent& ext,
                    const Index& idx_dst) {
  hsize_t dims[3], array_dims[3];
  for (int i = 0; i < ext.dim(); i++) {
    // dims[i] = ext_total[ext_total.dim() - 1 - i];
    array_dims[i] = array.extent()[array.dim() - 1 - i];
  }
  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  int dim = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  auto memspace = H5Screate_simple(ext.dim(), array_dims, NULL);

  hsize_t offsets[3], offsets_l[3], out_dim[3], count[3], stride[3];
  for (int i = 0; i < 3; i++) {
    count[i] = 1;
    stride[i] = 1;
    offsets[i] = idx_src[dim - i - 1];
    offsets_l[i] = idx_dst[dim - i - 1];
    out_dim[i] = ext[dim - i - 1];
  }
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsets, stride, count,
                      out_dim);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  auto status = H5Dread(dataset, h5datatype<T>(), memspace, dataspace,
                        plist_id, array.host_ptr());

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
}

}  // namespace Coffee

#endif  // _HDF_WRAPPER_IMPL_H_
