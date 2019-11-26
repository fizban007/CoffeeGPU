#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cuda_utility.h"
#include "field_solver_EZ.h"
#include "utils/timer.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 2
#define BLOCK_SIZE_Z 2

#define TINY 1e-12

namespace Coffee {

static dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

static dim3 blockGroupSize;

__device__ inline Scalar diff1x4(const Scalar * f, int ijk){
	return (f[ijk - 2] - 8 * f[ijk - 1] + 8 * f[ijk + 1] - f[ijk + 2]) / 12.0;
} 

__device__ inline Scalar diff1y4(const Scalar * f, int ijk){
	int s = dev_grid.dims[0];
	return (f[ijk - 2 * s] - 8 * f[ijk - 1 * s] + 8 * f[ijk + 1 * s] - f[ijk + 2 * s]) / 12.0;
} 

__device__ inline Scalar diff1z4(const Scalar * f, int ijk){
	int s = dev_grid.dims[0] * dev_grid.dims[1];
	return (f[ijk - 2 * s] - 8 * f[ijk - 1 * s] + 8 * f[ijk + 1 * s] - f[ijk + 2 * s]) / 12.0;
} 

__device__ inline Scalar diff4x2(const Scalar * f, int ijk){
	return (f[ijk - 2] - 4 * f[ijk - 1] + 6 * f[ijk] - 4 * f[ijk + 1] + f[ijk + 2]);
}

__device__ inline Scalar diff4y2(const Scalar * f, int ijk){
	int s = dev_grid.dims[0];
	return (f[ijk - 2 * s] - 4 * f[ijk - 1 * s] + 6 * f[ijk] - 4 * f[ijk + 1 * s] + f[ijk + 2 * s]);
}

__device__ inline Scalar diff4z2(const Scalar * f, int ijk){
	int s = dev_grid.dims[0] * dev_grid.dims[1];
	return (f[ijk - 2 * s] - 4 * f[ijk - 1 * s] + 6 * f[ijk] - 4 * f[ijk + 1 * s] + f[ijk + 2 * s]);
}

__device__ inline Scalar dfdx(const Scalar * f, int ijk){
	return diff1x4(f, ijk) / dev_grid.delta[0];
}

__device__ inline Scalar dfdy(const Scalar * f, int ijk){
	return diff1y4(f, ijk) / dev_grid.delta[1];
}

__device__ inline Scalar dfdz(const Scalar * f, int ijk){
	return diff1z4(f, ijk) / dev_grid.delta[2];
}

__device__ inline Scalar KO(const Scalar * f, int ijk){
	return diff4x2(f, ijk) + diff4y2(f, ijk) + diff4z2(f, ijk);
}

__global__ void
kernel_rk_step1_thread(const Scalar *Ex, const Scalar *Ey, const Scalar *Ez,
                      const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                      Scalar *dEx, Scalar *dEy, Scalar *dEz, Scalar *dBx, Scalar *dBy, Scalar *dBz,
                      const Scalar *P, Scalar *dP, int shift, Scalar As) {
	size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

	  Scalar rotBx = dfdy(Bz, ijk) - dfdz(By, ijk);
	  Scalar rotBy = dfdz(Bx, ijk) - dfdz(Bz, ijk);
	  Scalar rotBz = dfdx(By, ijk) - dfdy(Bx, ijk);
	  Scalar rotEx = dfdy(Ez, ijk) - dfdz(Ey, ijk);
	  Scalar rotEy = dfdz(Ex, ijk) - dfdz(Ez, ijk);
	  Scalar rotEz = dfdx(Ey, ijk) - dfdy(Ex, ijk);

	  Scalar divE = dfdx(Ex, ijk) + dfdy(Ey, ijk) + dfdz(Ez, ijk);
	  Scalar divB = dfdx(Bx, ijk) + dfdy(By, ijk) + dfdz(Bz, ijk);

	  Scalar B2 = Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
	  if (B2 < TINY) B2 = TINY;

	  Scalar Jp = (Bx[ijk] * rotBx + By[ijk] * rotBy + Bz[ijk] * rotBz)
	 							-(Ex[ijk] * rotEx + Ey[ijk] * rotEy + Ez[ijk] * rotEz);
	 	Scalar Jx = (divE * (Ey[ijk] * Bz[ijk] - Ez[ijk] * By[ijk]) 
	 							+ Jp * Bx[ijk]) / B2;
	 	Scalar Jy = (divE * (Ez[ijk] * Bx[ijk] - Ex[ijk] * Bz[ijk]) 
	 							+ Jp * By[ijk]) / B2;
	 	Scalar Jz = (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) 
	 							+ Jp * Bz[ijk]) / B2; 

	  dBx[ijk] = As * dBx[ijk] - dev_params.dt * (rotEx + dfdx(P, ijk));
	  dBy[ijk] = As * dBy[ijk] - dev_params.dt * (rotEy + dfdy(P, ijk));
	  dBz[ijk] = As * dBz[ijk] - dev_params.dt * (rotEz + dfdz(P, ijk));

	  dEx[ijk] = As * dEx[ijk] + dev_params.dt * (rotBx - Jx);
	  dEy[ijk] = As * dEy[ijk] + dev_params.dt * (rotBy - Jy);
	  dEz[ijk] = As * dEz[ijk] + dev_params.dt * (rotBz - Jz);

	  dP[ijk] = As * dP[ijk] - dev_params.dt * (dev_params.ch2 * divB + P[ijk] / dev_params.tau);
	}

}

__global__ void
kernel_rk_step2_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx, Scalar *By, Scalar *Bz,
											 const Scalar *dEx, const Scalar *dEy, const Scalar *dEz,
											 const Scalar *dBx, const Scalar *dBy, const Scalar *dBz,
											 const Scalar *dP, Scalar *P, int shift, Scalar Bs){
	size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    Ex[ijk] = Ex[ijk] + Bs * dEx[ijk];
    Ey[ijk] = Ey[ijk] + Bs * dEy[ijk];
    Ez[ijk] = Ez[ijk] + Bs * dEz[ijk];

    Bx[ijk] = Bx[ijk] + Bs * dBx[ijk];
    By[ijk] = By[ijk] + Bs * dBy[ijk];
    Bz[ijk] = Bz[ijk] + Bs * dBz[ijk];

    P[ijk] = P[ijk] + Bs * dP[ijk];
  }
}

__global__ void
kernel_Epar_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez,
                  const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                  int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

  	Scalar B2 = Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
		if (B2 < TINY) B2 = TINY;
		Scalar EB = Ex[ijk] * Bx[ijk] + Ey[ijk] * By[ijk] + Ez[ijk] * Bz[ijk];

		Ex[ijk] = Ex[ijk] - EB / B2 * Bx[ijk];
		Ey[ijk] = Ey[ijk] - EB / B2 * By[ijk];
		Ez[ijk] = Ez[ijk] - EB / B2 * Bz[ijk];

  }

}

__global__ void
kernel_EgtB_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez,
                   const Scalar *Bx, const Scalar *By, const Scalar *Bz,
                   int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

  	Scalar B2 = Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
		if (B2 < TINY) B2 = TINY;
		Scalar E2 = Ex[ijk] * Ex[ijk] + Ey[ijk] * Ey[ijk] + Ez[ijk] * Ez[ijk];

		if (E2 > B2) {
			Scalar s = sqrt(B2/E2);
			Ex[ijk] *= s;
			Ey[ijk] *= s;
			Ez[ijk] *= s;
		}

  }

}

__global__ void
kernel_KO_step1_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx, Scalar *By, Scalar *Bz,
                 Scalar *Ex_tmp, Scalar *Ey_tmp, Scalar *Ez_tmp, Scalar *Bx_tmp, 
                 Scalar *By_tmp, Scalar *Bz_tmp, Scalar *P, Scalar *P_tmp, int shift) {
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    Ex_tmp[ijk] = KO(Ex, ijk);
    Ey_tmp[ijk] = KO(Ey, ijk);
    Ez_tmp[ijk] = KO(Ez, ijk);

    Bx_tmp[ijk] = KO(Bx, ijk);
    By_tmp[ijk] = KO(By, ijk);
    Bz_tmp[ijk] = KO(Bz, ijk);

    P_tmp[ijk] = KO(P, ijk);

  }

}

__global__ void
kernel_KO_step2_thread(Scalar *Ex, Scalar *Ey, Scalar *Ez, Scalar *Bx, Scalar *By, Scalar *Bz,
                 Scalar *Ex_tmp, Scalar *Ey_tmp, Scalar *Ez_tmp, Scalar *Bx_tmp, 
                 Scalar *By_tmp, Scalar *Bz_tmp, Scalar *P, Scalar *P_tmp, int shift) {

  Scalar KO_const = -1.0/16.0;
  size_t ijk;
  int i = threadIdx.x + blockIdx.x * blockDim.x + dev_grid.guard[0] - shift;
  int j = threadIdx.y + blockIdx.y * blockDim.y + dev_grid.guard[1] - shift;
  int k = threadIdx.z + blockIdx.z * blockDim.z + dev_grid.guard[2] - shift;
  if (i < dev_grid.dims[0] - dev_grid.guard[0] + shift &&
      j < dev_grid.dims[1] - dev_grid.guard[1] + shift &&
      k < dev_grid.dims[2] - dev_grid.guard[2] + shift) {
    ijk = i + j * dev_grid.dims[0] +
          k * dev_grid.dims[0] * dev_grid.dims[1];

    Ex[ijk] -= dev_params.KOeps * KO_const * Ex_tmp[ijk];
    Ey[ijk] -= dev_params.KOeps * KO_const * Ey_tmp[ijk];
    Ez[ijk] -= dev_params.KOeps * KO_const * Ez_tmp[ijk];

    Bx[ijk] -= dev_params.KOeps * KO_const * Bx_tmp[ijk];
    By[ijk] -= dev_params.KOeps * KO_const * By_tmp[ijk];
    Bz[ijk] -= dev_params.KOeps * KO_const * Bz_tmp[ijk];

    P[ijk] -= dev_params.KOeps * KO_const * P_tmp[ijk];

  }

}

void
field_solver_EZ::rk_step(Scalar As, Scalar Bs){
	kernel_rk_step1_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2),
      dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2),
      P.dev_ptr(), dP.dev_ptr(), m_env.params().shift_ghost, As);
	CudaCheckError();
	kernel_rk_step2_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
      dE.dev_ptr(0), dE.dev_ptr(1), dE.dev_ptr(2),
      dB.dev_ptr(0), dB.dev_ptr(1), dB.dev_ptr(2),
      P.dev_ptr(), dP.dev_ptr(), m_env.params().shift_ghost, Bs);
	CudaCheckError();
}

void 
field_solver_EZ::Kreiss_Oliger(){
	kernel_KO_step1_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
		  Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
		  Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
		  P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
	CudaCheckError();
	kernel_KO_step2_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
		  Etmp.dev_ptr(0), Etmp.dev_ptr(1), Etmp.dev_ptr(2),
		  Btmp.dev_ptr(0), Btmp.dev_ptr(1), Btmp.dev_ptr(2),
		  P.dev_ptr(), Ptmp.dev_ptr(), m_env.params().shift_ghost);
	CudaCheckError();
}

void
field_solver_EZ::clean_epar(){
	kernel_Epar_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
		  m_env.params().shift_ghost);
	CudaCheckError();
}

void
field_solver_EZ::check_eGTb(){
	kernel_EgtB_thread<<<blockGroupSize, blockSize>>>(
		  m_data.E.dev_ptr(0), m_data.E.dev_ptr(1), m_data.E.dev_ptr(2),
		  m_data.B.dev_ptr(0), m_data.B.dev_ptr(1), m_data.B.dev_ptr(2),
		  m_env.params().shift_ghost);
	CudaCheckError();
}

void 
field_solver_EZ::evolve_fields(){

	Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257};
	Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681};
	Scalar cs[5] = {0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306784};

  for (int i = 0; i < 5; ++i){
  	rk_step(As[i], Bs[i]);

  	if (m_env.params().clean_ep) clean_epar();
  	if (m_env.params().check_egb) check_eGTb();

    CudaSafeCall(cudaDeviceSynchronize());
    m_env.send_guard_cells(m_data);
  }

  Kreiss_Oliger();
  if (m_env.params().clean_ep) clean_epar();
	if (m_env.params().check_egb) check_eGTb();
	CudaSafeCall(cudaDeviceSynchronize());
  m_env.send_guard_cells(m_data);
}

field_solver_EZ::field_solver_EZ(sim_data &mydata, sim_environment& env) : m_data(mydata), m_env(env) {
  dE = vector_field<Scalar>(m_data.env.grid());
  Etmp = vector_field<Scalar>(m_data.env.grid());
  dE.copy_stagger(m_data.E);
  Etmp.copy_stagger(m_data.E);
  dE.initialize();
  Etmp.initialize();

  dB = vector_field<Scalar>(m_data.env.grid());
  Btmp = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  Btmp.copy_stagger(m_data.B);
  dB.initialize();
  Btmp.initialize();

  P = multi_array<Scalar>(m_data.env.grid().extent());
  P.assign_dev(0.0);
  dP = multi_array<Scalar>(m_data.env.grid().extent());
  dP.assign_dev(0.0);
  Ptmp = multi_array<Scalar>(m_data.env.grid().extent());
  Ptmp.assign_dev(0.0);

  blockGroupSize = dim3((m_data.env.grid().reduced_dim(0) + m_env.params().shift_ghost * 2 + blockSize.x - 1) / blockSize.x,
                        (m_data.env.grid().reduced_dim(1) + m_env.params().shift_ghost * 2 + blockSize.y - 1) / blockSize.y,
                        (m_data.env.grid().reduced_dim(2) + m_env.params().shift_ghost * 2 + blockSize.z - 1) / blockSize.z);
  std::cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
  std::cout << blockGroupSize.x << ", " << blockGroupSize.y << ", " << blockGroupSize.z << std::endl;
}

field_solver_EZ::~field_solver_EZ() {}

}  // namespace Coffee