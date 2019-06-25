#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver.h"

namespace Coffee {

static dim3 gridSize(16, 16, 16);
static dim3 blockSize(8, 8, 8);

__global__ void
kernel_compute_rho(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                   Scalar *rho) {
  size_t ijk, iM1jk, ijM1k, ijkM1;
  for (int k =
           threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - 1;
       k < dev_grid.dims[2] - dev_grid.guard[2] + 1;
       k += blockDim.z * gridDim.z) {
    for (int j =
             threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - 1;
         j < dev_grid.dims[1] - dev_grid.guard[1] + 1;
         j += blockDim.y * gridDim.y) {
      for (int i = threadIdx.x + blockIdx.x * blockDim.x +
                   dev_grid.guard[0] - 1;
           i < dev_grid.dims[0] - dev_grid.guard[0] + 1;
           i += blockDim.x * gridDim.x) {
        ijk = i + j * dev_grid.dims[0] +
              k * dev_grid.dims[0] * dev_grid.dims[1];
        iM1jk = ijk - 1;
        ijM1k = ijk - dev_grid.dims[0];
        ijkM1 = ijk - dev_grid.dims[0] * dev_grid.dims[1];
        rho[ijk] = dev_grid.inv_delta[0] * (ex[ijk] - ex[iM1jk]) +
                   dev_grid.inv_delta[1] * (ey[ijk] - ey[ijM1k]) +
                   dev_grid.inv_delta[2] * (ez[ijk] - ez[ijkM1]);
      }
    }
  }
}

__global__ void
kernel_rk_push(const Scalar *ex, const Scalar *ey, const Scalar *ez,
               const Scalar *bx, const Scalar *by, const Scalar *bz,
               const Scalar *bx0, const Scalar *by0, const Scalar *bz0,
               Scalar *dex, Scalar *dey, Scalar *dez, Scalar *dbx,
               Scalar *dby, Scalar *dbz, Scalar *rho) {
  Scalar CCx = dev_params.dt * dev_grid.inv_delta[0];
  Scalar CCy = dev_params.dt * dev_grid.inv_delta[1];
  Scalar CCz = dev_params.dt * dev_grid.inv_delta[2];
  Scalar intex, intey, intez, intbx, intby, intbz, intrho;
  Scalar jx, jy, jz;
  size_t ijk, iP1jk, iM1jk, ijP1k, ijM1k, ijkP1, ijkM1;
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
        ijk = i + j * dev_grid.dims[0] +
              k * dev_grid.dims[0] * dev_grid.dims[1];
        iP1jk = ijk + 1;
        iM1jk = ijk - 1;
        ijP1k = ijk + dev_grid.dims[0];
        ijM1k = ijk - dev_grid.dims[0];
        ijkP1 = ijk + dev_grid.dims[0] * dev_grid.dims[1];
        ijkM1 = ijk - dev_grid.dims[0] * dev_grid.dims[1];
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
        // computing currents
        //   `j_x`:
        intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        jx = CCx * intrho * (intey * intbz - intby * intez) / (intbx * intbx + intby * intby + intbz * intbz);
        //   `j_y`:
        intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        jy = CCy * intrho * (intez * intbx - intex * intbz) / (intbx * intbx + intby * intby + intbz * intbz);
        //   `j_z`:
        intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        jz = CCz * intrho * (intex * intby - intbx * intey) / (intbx * intbx + intby * intby + intbz * intbz);

        dex[ijk] -= jx;
        dey[ijk] -= jy;
        dez[ijk] -= jz;
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
  size_t ijk;
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
        ijk = i + j * dev_grid.dims[0] +
              k * dev_grid.dims[0] * dev_grid.dims[1];
        // update E-field
        ex[ijk] = rk_c1 * enx[ijk] + rk_c2 * ex[ijk] + rk_c3 * dex[ijk];
        ey[ijk] = rk_c1 * eny[ijk] + rk_c2 * ey[ijk] + rk_c3 * dey[ijk];
        ez[ijk] = rk_c1 * enz[ijk] + rk_c2 * ez[ijk] + rk_c3 * dez[ijk];
        dex[ijk] = ex[ijk];
        dey[ijk] = ey[ijk];
        dez[ijk] = ez[ijk];
        // update B-field
        bx[ijk] = rk_c1 * bnx[ijk] + rk_c2 * bx[ijk] + rk_c3 * dbx[ijk];
        by[ijk] = rk_c1 * bny[ijk] + rk_c2 * by[ijk] + rk_c3 * dby[ijk];
        bz[ijk] = rk_c1 * bnz[ijk] + rk_c2 * bz[ijk] + rk_c3 * dbz[ijk];
      }
    }
  }
}

__global__ void
kernel_clean_epar(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                  const Scalar *bx, const Scalar *by, const Scalar *bz,
                  Scalar *dex, Scalar *dey, Scalar *dez) {
  Scalar intex, intey, intez, intbx, intby, intbz;
  size_t ijk;
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
        ijk = i + j * dev_grid.dims[0] +
              k * dev_grid.dims[0] * dev_grid.dims[1];
        // x:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        dex[ijk] = ex[ijk] - (intex * intbx + intey * intby + intez * intbz) * intbx
                              / (intbx * intbx + intby * intby + intbz * intbz);

        // y:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        dey[ijk] = ey[ijk] - (intex * intbx + intey * intby + intez * intbz) * intby
                              / (intbx * intbx + intby * intby + intbz * intbz);

        // z:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        dez[ijk] = ez[ijk] - (intex * intbx + intey * intby + intez * intbz) * intbz
                              / (intbx * intbx + intby * intby + intbz * intbz);
      }
    }
  }
}

__global__ void
kernel_check_eGTb(const Scalar *dex, const Scalar *dey, const Scalar *dez
                  Scalar *ex, Scalar *ey, Scalar *ez,
                  const Scalar *bx, const Scalar *by, const Scalar *bz) {
  Scalar intex, intey, intez, intbx, intby, intbz, emag, bmag, temp;
  size_t ijk;
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
        ijk = i + j * dev_grid.dims[0] +
              k * dev_grid.dims[0] * dev_grid.dims[1];
        // x:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        emag = intex * intex + intey * intey + intez * intez;
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b011), dev_grid.dims[0], dev_grid.dims[1]);
        bmag = intbx * intbx + intby * intby + intbz * intbz;
        if (emag > bmag) {
          temp = sqrt(bmag / emag);
        } else {
          temp = 1.0;
        }
        ex[ijk] = temp * dex[ijk]

        // y:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        emag = intex * intex + intey * intey + intez * intez;
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b101), dev_grid.dims[0], dev_grid.dims[1]);
        bmag = intbx * intbx + intby * intby + intbz * intbz;
        if (emag > bmag) {
          temp = sqrt(bmag / emag);
        } else {
          temp = 1.0;
        }
        ey[ijk] = temp * dey[ijk]

        // z:
        intex = interpolate(ex, ijk, Stagger(0b011), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intez = interpolate(ez, ijk, Stagger(0b110), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        emag = intex * intex + intey * intey + intez * intez;
        intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        intbz = interpolate(bz, ijk, Stagger(0b001), Stagger(0b110), dev_grid.dims[0], dev_grid.dims[1]);
        bmag = intbx * intbx + intby * intby + intbz * intbz;
        if (emag > bmag) {
          temp = sqrt(bmag / emag);
        } else {
          temp = 1.0;
        }
        ez[ijk] = temp * dez[ijk]
      }
    }
  }
}

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

field_solver::~field_solver() {}

void
field_solver::evolve_fields() {
  copy_fields();

  // substep #1:
  rk_push();
  rk_update(1.0, 0.0, 1.0);
  check_eGTb();

  // substep #2:
  rk_push();
  rk_update(0.75, 0.25, 0.25);
  check_eGTb();

  // substep #3:
  rk_push();
  rk_update(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  clean_epar();
  check_eGTb();

  // boundary call
  CudaSafeCall(cudaDeviceSynchronize());
}

void
field_solver::copy_fields() {
  // `En = E, Bn = B`:
  En.copy_from(m_data.E);
  Bn.copy_from(m_data.B);
}

void
field_solver::rk_push() {
  // `rho = div E`
  kernel_compute_rho<<<gridSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr);
  // `dE = curl B - curl B0 - j, dB = -curl E`
  kernel_rk_push<<<gridSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr());
}

void
field_solver::rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3) {
  // `E = c1 En + c2 E + c3 dE, B = c1 Bn + c2 B + c3 dB`
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
  // clean `E || B`
  kernel_clean_epar<<<gridSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2));
}

void
field_solver::check_eGTb() {
  // renormalizing `E > B`
  kernel_check_eGTb<<<gridSize, blockSize>>>(
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2),
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2));
}

}  // namespace Coffee
