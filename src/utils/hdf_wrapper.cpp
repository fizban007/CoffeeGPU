#include "hdf_wrapper.h"
#include "mpi.h"

namespace Coffee {

H5File::H5File() {}

H5File::H5File(hid_t file_id) : m_file_id(file_id) { m_is_open = true; }

H5File::H5File(const std::string& filename, H5OpenMode mode) {
  open(filename, mode);
}

H5File::H5File(H5File&& other) {
  m_file_id = other.m_file_id;
  m_is_open = true;
  other.m_is_open = false;
}

H5File::~H5File() { close(); }

H5File&
H5File::operator=(H5File&& other) {
  m_file_id = other.m_file_id;
  m_is_open = true;
  other.m_is_open = false;
  return *this;
}

void
H5File::open(const std::string& filename, H5OpenMode mode) {
  unsigned int h5mode;
  if (mode == H5OpenMode::read_write || mode == H5OpenMode::rw_parallel)
    h5mode = H5F_ACC_RDWR;
  else
    h5mode = H5F_ACC_RDONLY;

  hid_t plist_id = H5P_DEFAULT;
  // Enable mpio when writing
  if (mode == H5OpenMode::rw_parallel || mode == H5OpenMode::read_parallel) {
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  }

  m_file_id = H5Fopen(filename.c_str(), h5mode, plist_id);
  m_is_open = true;

  if (mode == H5OpenMode::rw_parallel) {
    H5Pclose(plist_id);
  }
}

void
H5File::close() {
  if (m_is_open) {
    H5Fclose(m_file_id);
    m_is_open = false;
  }
}

H5File
hdf_create(const std::string& filename, H5CreateMode mode) {
  // auto h5mode =
  //     (mode == H5CreateMode::trunc ? H5F_ACC_TRUNC : H5F_ACC_EXCL);
  unsigned int h5mode;
  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::trunc)
    h5mode = H5F_ACC_TRUNC;
  else
    h5mode = H5F_ACC_EXCL;

  hid_t plist_id = H5P_DEFAULT;
  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::excl_parallel) {
    // Enable mpio when writing
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  }
  hid_t datafile =
      H5Fcreate(filename.c_str(), h5mode, H5P_DEFAULT, plist_id);

  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::excl_parallel) {
    H5Pclose(plist_id);
  }

  H5File file(datafile);
  return file;
}

//// Explicitly specialize h5datatype functions
template <>
hid_t
h5datatype<char>() {
  return H5T_NATIVE_CHAR;
}

template <>
hid_t
h5datatype<float>() {
  return H5T_NATIVE_FLOAT;
}

template <>
hid_t
h5datatype<double>() {
  return H5T_NATIVE_DOUBLE;
}

template <>
hid_t
h5datatype<uint32_t>() {
  return H5T_NATIVE_UINT32;
}

template <>
hid_t
h5datatype<uint64_t>() {
  return H5T_NATIVE_UINT64;
}

}  // namespace Coffee
