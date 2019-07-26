#include "data/multi_array.h"
#include <mpi.h>

using namespace Coffee;

int
main(int argc, char *argv[]) {
  multi_array<float> arr(10, 10, 10);

  MPI_Init(&argc, &argv);

  MPI_Comm m_world = MPI_COMM_WORLD;
  int m_rank, m_size;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);

  arr.assign_dev(m_rank);
  arr.sync_to_host();

  MPI_Datatype y_type;
  MPI_Type_vector(10, 10 * 2, 10 * 10, MPI_FLOAT, &y_type);
  MPI_Type_commit(&y_type);

  MPI_Datatype x_type, x_temp;
  MPI_Type_vector(10, 2, 10,
                  MPI_FLOAT, &x_temp);
  MPI_Type_commit(&x_temp);
  MPI_Type_create_hvector(
      10, 1,
      sizeof(float) * 10 * 10, x_temp, &x_type);
  MPI_Type_commit(&x_type);

  if (m_rank == 1) {
    for (int k = 0; k < 10; k++) {
      for (int j = 0; j < 10; j++) {
        std::cout << arr(9, j, k) << " ";
      }
      std::cout << std::endl;
    }
  }
  if (m_rank == 0) {
    MPI_Send(arr.dev_ptr() + 1, 1, x_type, 1, 0, m_world);
  } else if (m_rank == 1) {
    MPI_Status status;
    MPI_Recv(arr.dev_ptr() + 8, 1, x_type, 0, 0, m_world, &status);
  }

  arr.sync_to_host();

  if (m_rank == 1) {
    for (int k = 0; k < 10; k++) {
      for (int j = 0; j < 10; j++) {
        std::cout << arr(7, j, k) << " ";
      }
      std::cout << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
