#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver.h"

namespace Coffee {

__global__ void
kernel_rk_push(const Scalar *ex, const Scalar *ey, const Scalar *ez,
               const Scalar *bx, const Scalar *by, const Scalar *bz,
               const Scalar *bx0, const Scalar *by0, const Scalar *bz0,
               Scalar *dex, Scalar *dey, Scalar *dez, Scalar *dbx,
               Scalar *dby, Scalar *dbz) {
  Scalar CCx = dev_params.dt * dev_grid.inv_delta[0];
  Scalar CCy = dev_params.dt * dev_grid.inv_delta[1];
  Scalar CCz = dev_params.dt * dev_grid.inv_delta[2];
  for (int k =
           threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2];
       k < dev_grid.dims[2] - dev_grid.guard[2];
       k += blockDim.z * gridDim.z) {
    for (int j =
             threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1];
         j < dev_grid.dims[1] - dev_grid.guard[1];
         j += blockDim.y * gridDim.y) {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x +
                   dev_grid.guard[0];
           i < dev_grid.dims[0] - dev_grid.guard[0];
           i += blockDim.x * gridDim.x) {
        size_t ijk = i + j * dev_grid.dims[0] +
                     k * dev_grid.dims[0] * dev_grid.dims[1];
        size_t iP1jk = (i + 1) + j * dev_grid.dims[0] +
                       k * dev_grid.dims[0] * dev_grid.dims[1];
        size_t iM1jk = (i - 1) + j * dev_grid.dims[0] +
                       k * dev_grid.dims[0] * dev_grid.dims[1];
        size_t ijP1k = i + (j + 1) * dev_grid.dims[0] +
                       k * dev_grid.dims[0] * dev_grid.dims[1];
        size_t ijM1k = i + (j - 1) * dev_grid.dims[0] +
                       k * dev_grid.dims[0] * dev_grid.dims[1];
        size_t ijkP1 = i + j * dev_grid.dims[0] +
                       (k + 1) * dev_grid.dims[0] * dev_grid.dims[1];
        size_t ijkM1 = i + j * dev_grid.dims[0] +
                       (k - 1) * dev_grid.dims[0] * dev_grid.dims[1];
        // push B-field
        dbx[ijk] = CCx * (ey[ijkP1] - ey[ijk] - ez[ijP1k] + ez[ijk]);
        dby[ijk] = CCy * (ez[iP1jk] - ez[ijk] - ez[ijkP1] + ex[ijk]);
        dbz[ijk] = CCz * (ex[ijP1k] - ex[ijk] - ez[iP1jk] + ey[ijk]);
        // push E-field
        dex[ijk] =
            CCx * ((by[ijkM1] - by[ijk] - bz[ijM1k] + bz[ijk]) -
                   (by0[ijkM1] - bz0[ijk] - bz0[ijM1k] + bz0[ijk]));
        dey[ijk] =
            CCy * ((bz[iM1jk] - bz[ijk] - bx[ijkM1] + bx[ijk]) -
                   (bz0[iM1jk] - bz0[ijk] - bx0[ijkM1] + bx0[ijk]));
        dez[ijk] =
            CCz * ((bx[ijM1k] - bx[ijk] - by[iM1jk] + by[ijk]) -
                   (bx0[ijM1k] - bx0[ijk] - by0[iM1jk] + by0[ijk]));
        // compute `jx, jy, jz` from interpolation
        // dex[ijk] -= jx;
        // dey[ijk] -= jy;
        // dez[ijk] -= jz;
      }
    }
  }
}

__global__ void
kernel_rk_update(Scalar *ex, Scalar *ey, Scalar *ez, Scalar *bx,
                 Scalar *by, Scalar *bz, const Scalar *enx,
                 const Scalar *eny, const Scalar *enz,
                 const Scalar *bnx, const Scalar *bny,
                 const Scalar *bnz, const Scalar *dex,
                 const Scalar *dey, const Scalar *dez,
                 const Scalar *dbx, const Scalar *dby,
                 const Scalar *dbz, Scalar rk_c1, Scalar rk_c2,
                 Scalar rk_c3) {
  for (int k =
           threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2];
       k < dev_grid.dims[2] - dev_grid.guard[2];
       k += blockDim.z * gridDim.z) {
    for (int j =
             threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1];
         j < dev_grid.dims[1] - dev_grid.guard[1];
         j += blockDim.y * gridDim.y) {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x +
                   dev_grid.guard[0];
           i < dev_grid.dims[0] - dev_grid.guard[0];
           i += blockDim.x * gridDim.x) {
        size_t ijk = i + j * dev_grid.dims[0] +
                     k * dev_grid.dims[0] * dev_grid.dims[1];
        // update E-field
        ex[ijk] = rk_c1 * enx[ijk] + rk_c2 * ex[ijk] + rk_c3 * dex[ijk];
        ey[ijk] = rk_c1 * eny[ijk] + rk_c2 * ey[ijk] + rk_c3 * dey[ijk];
        ez[ijk] = rk_c1 * enz[ijk] + rk_c2 * ez[ijk] + rk_c3 * dez[ijk];
        // update B-field
        bx[ijk] = rk_c1 * bnx[ijk] + rk_c2 * bx[ijk] + rk_c3 * dbx[ijk];
        by[ijk] = rk_c1 * bny[ijk] + rk_c2 * by[ijk] + rk_c3 * dby[ijk];
        bz[ijk] = rk_c1 * bnz[ijk] + rk_c2 * bz[ijk] + rk_c3 * dbz[ijk];
      }
    }
  }
}

__global__ void
kernel_clean_epar() {}

field_solver::field_solver(sim_data &mydata) : m_data(mydata) {
  En = vector_field<Scalar>(m_data.env.grid());
  dE = vector_field<Scalar>(m_data.env.grid());
  En.copy_stagger(m_data.E);
  dE.copy_stagger(m_data.E);

  Bn = vector_field<Scalar>(m_data.env.grid());
  dB = vector_field<Scalar>(m_data.env.grid());
  Bn.copy_stagger(m_data.B);
  dB.copy_stagger(m_data.B);
}

void
field_solver::evolve_fields() {
  // `En = E`, `Bn = B`:
  copy_fields();

  // `dB = -curl E`
  // `dE = curl B - curl B0 - j`
  // `E = c1 En + c2 E + c3 dE`
  // `B = c1 Bn + c2 B + c3 dB`
  rk_push();
  rk_update(1.0, 0.0, 1.0);
  rk_push();
  rk_update(0.75, 0.25, 0.25);
  rk_push();
  rk_update(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);

  // clean `E || B`
  clean_epar();
  // boundary call
}

void
field_solver::copy_fields() {
  En.copy_from(m_data.E);
  Bn.copy_from(m_data.B);
}

void
field_solver::rk_push() {
  dim3 gridSize(16, 16, 16);
  dim3 blockSize(8, 8, 8);
  kernel_rk_push<<<gridSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2));
}

void
field_solver::rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3) {
  dim3 gridSize(16, 16, 16);
  dim3 blockSize(8, 8, 8);
  kernel_rk_update<<<gridSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      En.dev_ptr(0), En.dev_ptr(1), En.dev_ptr(2), Bn.dev_ptr(0),
      Bn.dev_ptr(1), Bn.dev_ptr(2), dE.dev_ptr(0), dE.dev_ptr(1),
      dE.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2), rk_c1,
      rk_c2, rk_c3);
}

void
field_solver::clean_epar() {
  dim3 gridSize(16, 16, 16);
  dim3 blockSize(8, 8, 8);
  kernel_clean_epar<<<gridSize, blockSize>>>();
}

}  // namespace Coffee
