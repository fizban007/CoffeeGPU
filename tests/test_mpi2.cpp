#include "data/multi_array.h"
#include <mpi.h>

using namespace Coffee;

int
main(int argc, char *argv[]) {
  int size = 4;
  int p = 4;
  int size1 = size * p;
  multi_array<float> arr(size, size, size);
  multi_array<float> arr2(size1, size1);

  MPI_Init(&argc, &argv);

  MPI_Comm m_world = MPI_COMM_WORLD;
  int m_rank, m_size;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

  arr.assign(m_rank);
  arr2.assign(-1.0);


  MPI_Datatype x_send, x_temp;
  MPI_Type_vector(size, 1, size,
                  MPI_FLOAT, &x_temp);
  MPI_Type_commit(&x_temp);
  MPI_Type_create_hvector(
      size, 1,
      sizeof(float) * size * size, x_temp, &x_send);
  MPI_Type_commit(&x_send);

  MPI_Datatype x_receive;
  MPI_Type_vector(size, size, size1, MPI_FLOAT, &x_receive);
  MPI_Type_commit(&x_receive);

  if (m_rank == 1) {
    for (int k = 0; k < size; k++) {
      for (int j = 0; j < size; j++) {
        std::cout << arr(size - 1, j, k) << " ";
      }
      std::cout << std::endl;
    }
  }

  MPI_Send(arr.host_ptr(), 1, x_send, 0, m_rank, m_world);
  if (m_rank == 0) {
    MPI_Status status;
    for (int i = 0; i < p * p; i++){
      offset = i / p * size * size1 + (i % p) * size;
      MPI_Recv(arr2.host_ptr() + offset, 1, x_receive, i, i, m_world, &status);
    }
  }

  arr.sync_to_host();
  arr2.sync_to_host();

  if (m_rank == 0) {
    for (int k = 0; k < size1; k++) {
      for (int j = 0; j < size1; j++) {
        std::cout << arr2(j, k) << " ";
      }
      std::cout << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
