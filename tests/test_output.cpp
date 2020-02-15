#include <iostream>
#include <string>
#include <hdf5.h>
#include <mpi.h>
#include "utils/hdf_wrapper_impl.hpp"

using namespace Coffee;

int main(int argc, char *argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  MPI_Comm m_world = MPI_COMM_WORLD;
  int m_rank, m_size;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);
  
  // Create data for every process
  // int N = 256;
  // hsize_t dims[3] = {N, N, N};
  // hsize_t local_dim[3] = {N / m_size, N, N};
  // hsize_t offsets[3] = {m_rank * local_dim[0], N, N};
  
  // std::string filename = "test.h5";

  // hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  // H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  // hid_t datafile =
  //     H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  // H5Pclose(plist_id);

  // H5Fclose(datafile);
  multi_array<float> arr(10, 10);
  arr.assign(m_rank);
  Extent total_ext(20, 10);

  auto file = hdf_create("test_mpi_output.h5", H5CreateMode::trunc_parallel);

  file.write_parallel(arr, total_ext, Index(10 * m_rank, 0, 0), arr.extent(), Index(0, 0, 0), "data");
  
  // file.close();

  multi_array<float> out(8, 8);
  out.assign(m_rank + 2.0);
  file.write_parallel(out, total_ext, Index(1 + 10 * m_rank, 0, 0), out.extent(), Index(0, 0, 0), "data2");
  // file = H5File("test_mpi_output.h5", H5OpenMode::read_parallel);
  // file.read_subset(out,
  //                  "data", Index(1 + 10 * m_rank, 1, 0), Extent(8, 8), Index(0, 0, 0));
  // std::cout << "rank " << m_rank << " has value " << out(2, 2) << "\n";
  file.close();
  return 0;
}
