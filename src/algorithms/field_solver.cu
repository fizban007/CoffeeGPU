#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver.h"
#include "interpolation.h"
#include "utils/timer.h"
#include "utils/nvproftool.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4

#define SHIFT_GHOST 2

#define TINY 1e-7

namespace Coffee {

static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

__global__ void
kernel_compute_rho_thread(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                          Scalar *rho, int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    rho[ijk] = dev_grid.inv_delta[0] * (ex[ijk] - ex[ijk - 1]) +
               dev_grid.inv_delta[1] * (ey[ijk] - ey[ijk - dev_grid.dims[0]]) +
               dev_grid.inv_delta[2] * (ez[ijk] - ez[ijk - dev_grid.dims[0] * dev_grid.dims[1]]);
  }
}

__global__ void
kernel_rk_push_thread(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                      const Scalar *bx, const Scalar *by, const Scalar *bz,
                      const Scalar *bx0, const Scalar *by0,
                      const Scalar *bz0, Scalar *dex, Scalar *dey,
                      Scalar *dez, Scalar *dbx, Scalar *dby, Scalar *dbz,
                      Scalar *rho, int shift) {
  Scalar CCx = dev_params.dt * dev_grid.inv_delta[0];
  Scalar CCy = dev_params.dt * dev_grid.inv_delta[1];
  Scalar CCz = dev_params.dt * dev_grid.inv_delta[2];
  Scalar intex, intey, intez, intbx, intby, intbz, intrho;
  Scalar jx, jy, jz;
  size_t ijk, iP1jk, iM1jk, ijP1k, ijM1k, ijkP1, ijkM1;

  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    iP1jk = ijk + 1;
    iM1jk = ijk - 1;
    ijP1k = ijk + dev_grid.dims[0];
    ijM1k = ijk - dev_grid.dims[0];
    ijkP1 = ijk + dev_grid.dims[0] * dev_grid.dims[1];
    ijkM1 = ijk - dev_grid.dims[0] * dev_grid.dims[1];
    // push B-field
    dbx[ijk] = CCz * (ey[ijkP1] - ey[ijk]) - CCy * (ez[ijP1k] - ez[ijk]);
    dby[ijk] = CCx * (ez[iP1jk] - ez[ijk]) - CCz * (ex[ijkP1] - ex[ijk]);
    dbz[ijk] = CCy * (ex[ijP1k] - ex[ijk]) - CCx * (ey[iP1jk] - ey[ijk]);
    // push E-field
    dex[ijk] = (CCz * (by[ijkM1] - by[ijk]) - CCy * (bz[ijM1k] - bz[ijk])) -
                      (CCz * (by0[ijkM1] - by0[ijk]) - CCy * (bz0[ijM1k] - bz0[ijk]));
    dey[ijk] = (CCx * (bz[iM1jk] - bz[ijk]) - CCz * (bx[ijkM1] - bx[ijk])) -
                      (CCx * (bz0[iM1jk] - bz0[ijk]) - CCz * (bx0[ijkM1] - bx0[ijk]));
    dez[ijk] = (CCy * (bx[ijM1k] - bx[ijk]) - CCx * (by[iM1jk] - by[ijk])) -
                      (CCy * (bx0[ijM1k] - bx0[ijk]) - CCx * (by0[iM1jk] - by0[ijk]));
    // if (i == 10 && j == 10 && k == 10)
      // printf("%d, %d, %d\n", dev_grid.dims[0], dev_grid.dims[1], dev_grid.dims[2]);
      // printf("%f, %f, %f\n", dex[ijk], dey[ijk], dez[ijk]);
      // printf("%lu, %lu, %lu\n", ijkM1, ijM1k, iM1jk);

    // computing currents
    //   `j_x`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b110),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jx = dev_params.dt * intrho * (intey * intbz - intby * intez) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    //   `j_y`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b101),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jy = dev_params.dt * intrho * (intez * intbx - intex * intbz) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);
    //   `j_z`:
    intrho = interpolate(rho, ijk, Stagger(0b111), Stagger(0b011),
                         dev_grid.dims[0], dev_grid.dims[1]);
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                       dev_grid.dims[0], dev_grid.dims[1]);
    jz = dev_params.dt * intrho * (intex * intby - intbx * intey) /
         (intbx * intbx + intby * intby + intbz * intbz + TINY);

    dex[ijk] -= jx;
    dey[ijk] -= jy;
    dez[ijk] -= jz;
  }
}

__global__ void
kernel_rk_update_thread(Scalar *ex, Scalar *ey, Scalar *ez, Scalar *bx,
                        Scalar *by, Scalar *bz, const Scalar *enx,
                        const Scalar *eny, const Scalar *enz,
                        const Scalar *bnx, const Scalar *bny,
                        const Scalar *bnz, Scalar *dex, Scalar *dey,
                        Scalar *dez, const Scalar *dbx, const Scalar *dby,
                        const Scalar *dbz, Scalar rk_c1, Scalar rk_c2,
                        Scalar rk_c3, int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
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

__global__ void
kernel_clean_epar_thread(const Scalar *ex, const Scalar *ey, const Scalar *ez,
                         const Scalar *bx, const Scalar *by, const Scalar *bz,
                         Scalar *dex, Scalar *dey, Scalar *dez, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // x:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dex[ijk] = ex[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intbx /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // y:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dey[ijk] = ey[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intby /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);

    // z:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    dez[ijk] = ez[ijk] -
               (intex * intbx + intey * intby + intez * intbz) *
                   intbz /
                   (intbx * intbx + intby * intby + intbz * intbz + TINY);
  }
}

__global__ void
kernel_check_eGTb_thread(const Scalar *dex, const Scalar *dey,
                         const Scalar *dez, Scalar *ex, Scalar *ey, Scalar *ez,
                         const Scalar *bx, const Scalar *by,
                         const Scalar *bz, int shift) {
  Scalar intex, intey, intez, intbx, intby, intbz, emag, bmag, temp;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];
    // x:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b110),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ex[ijk] = temp * dex[ijk];

    // y:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b101),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ey[ijk] = temp * dey[ijk];

    // z:
    intex = interpolate(ex, ijk, Stagger(0b110), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intey = interpolate(ey, ijk, Stagger(0b101), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intez = interpolate(ez, ijk, Stagger(0b011), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    emag = intex * intex + intey * intey + intez * intez + TINY;
    intbx = interpolate(bx, ijk, Stagger(0b001), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intby = interpolate(by, ijk, Stagger(0b010), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    intbz = interpolate(bz, ijk, Stagger(0b100), Stagger(0b011),
                        dev_grid.dims[0], dev_grid.dims[1]);
    bmag = intbx * intbx + intby * intby + intbz * intbz + TINY;
    if (emag > bmag) {
      temp = sqrt(bmag / emag);
    } else {
      temp = 1.0;
    }
    ez[ijk] = temp * dez[ijk];
  }
}


field_solver::field_solver(sim_data &mydata, sim_environment& env) : m_data(mydata), m_env(env) {
  En = vector_field<Scalar>(m_data.env.grid());
  dE = vector_field<Scalar>(m_data.env.grid());
  En.copy_stagger(m_data.E);
  dE.copy_stagger(m_data.E);
  En.initialize();
  dE.initialize();

  Bn = vector_field<Scalar>(m_data.env.grid());
  dB = vector_field<Scalar>(m_data.env.grid());
  Bn.copy_stagger(m_data.B);
  dB.copy_stagger(m_data.B);
  Bn.initialize();
  dB.initialize();

  rho = multi_array<Scalar>(m_data.env.grid().extent());
  rho.assign_dev(0.0);

  blockGroupSize = dim3((m_data.env.grid().reduced_dim(0) + SHIFT_GHOST * 2 + blockSize.x - 1) / blockSize.x,
                        (m_data.env.grid().reduced_dim(1) + SHIFT_GHOST * 2 + blockSize.y - 1) / blockSize.y,
                        (m_data.env.grid().reduced_dim(2) + SHIFT_GHOST * 2 + blockSize.z - 1) / blockSize.z);
  std::cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
  std::cout << blockGroupSize.x << ", " << blockGroupSize.y << ", " << blockGroupSize.z << std::endl;
}

field_solver::~field_solver() {}

void
field_solver::evolve_fields() {
    RANGE_PUSH("Compute", CLR_GREEN);
    copy_fields();

  // substep #1:
  rk_push();
  rk_update(1.0, 0.0, 1.0);
  clean_epar();
  check_eGTb();
    CudaSafeCall(cudaDeviceSynchronize());
    RANGE_POP;
    m_env.send_guard_cells(m_data);

  // substep #2:
    RANGE_PUSH("Compute", CLR_GREEN);
  rk_push();
  rk_update(0.75, 0.25, 0.25);
  clean_epar();
  check_eGTb();
    CudaSafeCall(cudaDeviceSynchronize());
    RANGE_POP;
    m_env.send_guard_cells(m_data);

  // substep #3:
    RANGE_PUSH("Compute", CLR_GREEN);
  rk_push();
  rk_update(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  clean_epar();
  check_eGTb();
    CudaSafeCall(cudaDeviceSynchronize());
    RANGE_POP;
    m_env.send_guard_cells(m_data);
}

void 
field_solver::impose_bc(uint32_t timestep) {
  Scalar time = timestep * m_env.params().dt;
  m_data.E.impose_bc_z1(0, [&](Scalar x, Scalar y, Scalar z) {
    Scalar rx = (x - m_env.params().xc);
    Scalar ry1 = (y - m_env.params().yc1);
    Scalar ry2 = (y - m_env.params().yc2);
    Scalar rz = (z - m_env.params().zc);

    Scalar ry = (y - m_env.params().yc1);
    Scalar rr = sqrt(rx * rx + ry * ry);

    Scalar r1 = sqrt(rx * rx + ry1 * ry1 + rz * rz);
    Scalar r2 = sqrt(rx * rx + ry2 * ry2 + rz * rz);
    
    Scalar Bz = rz / (r1 * r1 * r1) - rz / (r2 * r2 * r2);

    Scalar ampl = exp(-rr / m_env.params().r0) * (rr / m_env.params().r0);
    return ampl * (rx / rr) * Bz * m_env.params().v_twist;
  });
  m_data.E.impose_bc_z1(1, [&](Scalar x, Scalar y, Scalar z) {
    Scalar rx = (x - m_env.params().xc);
    Scalar ry1 = (y - m_env.params().yc1);
    Scalar ry2 = (y - m_env.params().yc2);
    Scalar rz = (z - m_env.params().zc);

    Scalar ry = (y - m_env.params().yc1);
    Scalar rr = sqrt(rx * rx + ry * ry);

    Scalar r1 = sqrt(rx * rx + ry1 * ry1 + rz * rz);
    Scalar r2 = sqrt(rx * rx + ry2 * ry2 + rz * rz);
    
    Scalar Bz = rz / (r1 * r1 * r1) - rz / (r2 * r2 * r2);

    Scalar ampl = exp(-rr / m_env.params().r0) * (rr / m_env.params().r0);
    return ampl * (ry / rr) * Bz * m_env.params().v_twist;
  });
  m_data.B.impose_bc_z1(2, [&](Scalar x, Scalar y, Scalar z) {
    Scalar rx = (x - m_env.params().xc);
    Scalar ry1 = (y - m_env.params().yc1);
    Scalar ry2 = (y - m_env.params().yc2);
    Scalar rz = (z - m_env.params().zc);

    Scalar r1 = sqrt(rx * rx + ry1 * ry1 + rz * rz);
    Scalar r2 = sqrt(rx * rx + ry2 * ry2 + rz * rz);
    
    return rz / (r1 * r1 * r1) - rz / (r2 * r2 * r2);
  });
}

void
field_solver::copy_fields() {
  // `En = E, Bn = B`:
  En.copy_from(m_data.E);
  Bn.copy_from(m_data.B);
  dE.initialize();
  dB.initialize();
}

void
field_solver::rk_push() {
  // `rho = div E`
  kernel_compute_rho_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      rho.dev_ptr(), SHIFT_GHOST);
  CudaCheckError();
  // `dE = curl B - curl B0 - j, dB = -curl E`
  kernel_rk_push_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      m_data.B0.dev_ptr(0), m_data.B0.dev_ptr(1), m_data.B0.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), dB.dev_ptr(0),
      dB.dev_ptr(1), dB.dev_ptr(2), rho.dev_ptr(), SHIFT_GHOST);
  CudaCheckError();
}

void
field_solver::rk_update(Scalar rk_c1, Scalar rk_c2, Scalar rk_c3) {
  // `E = c1 En + c2 E + c3 dE, B = c1 Bn + c2 B + c3 dB`
  kernel_rk_update_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      En.dev_ptr(0), En.dev_ptr(1), En.dev_ptr(2), Bn.dev_ptr(0),
      Bn.dev_ptr(1), Bn.dev_ptr(2), dE.dev_ptr(0), dE.dev_ptr(1),
      dE.dev_ptr(2), dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2), rk_c1,
      rk_c2, rk_c3, SHIFT_GHOST);
  CudaCheckError();
}

void
field_solver::clean_epar() {
  // clean `E || B`
  kernel_clean_epar_thread<<<blockGroupSize, blockSize>>>(
      m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
      m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), SHIFT_GHOST);
  CudaCheckError();
}

void
field_solver::check_eGTb() {
  // renormalizing `E > B`
  kernel_check_eGTb_thread<<<blockGroupSize, blockSize>>>(
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2), m_data.E.dev_ptr(0),
      m_data.E.dev_ptr(1), m_data.E.dev_ptr(2), m_data.B.dev_ptr(0),
      m_data.B.dev_ptr(1), m_data.B.dev_ptr(2), SHIFT_GHOST);
  CudaCheckError();
}

}  // namespace Coffee
