#include "field_solver_EZ.h"
#include "algorithms/finite_diff.h"
#include "utils/timer.h"

namespace Coffee {

const Grid *l_grid;

template <typename T>
inline Scalar
dfdx(const multi_array<T> &f, int ijk) {
  return df1(f.host_ptr(), ijk, 1, l_grid->inv_delta[0]);
}

template <typename T>
inline Scalar
dfdy(const multi_array<T> &f, int ijk) {
  return df1(f.host_ptr(), ijk, l_grid->dims[0], l_grid->inv_delta[1]);
}

template <typename T>
inline Scalar
dfdz(const multi_array<T> &f, int ijk) {
  return df1(f.host_ptr(), ijk, l_grid->dims[0] * l_grid->dims[1],
             l_grid->inv_delta[2]);
}

field_solver_EZ::field_solver_EZ(sim_data &mydata, sim_environment &env)
    : m_data(mydata), m_env(env) {
  dE = vector_field<Scalar>(m_data.env.grid());
  Etmp = vector_field<Scalar>(m_data.env.grid());
  dE.copy_stagger(m_data.E);
  Etmp.copy_stagger(m_data.E);
  dE.initialize();
  Etmp.copy_from(m_data.E);

  dB = vector_field<Scalar>(m_data.env.grid());
  Btmp = vector_field<Scalar>(m_data.env.grid());
  dB.copy_stagger(m_data.B);
  Btmp.copy_stagger(m_data.B);
  dB.initialize();
  Btmp.copy_from(m_data.B);

  // P = multi_array<Scalar>(m_data.env.grid().extent());
  // P.assign(0.0);
  dP = multi_array<Scalar>(m_data.env.grid().extent());
  dP.assign(0.0);
  Ptmp = multi_array<Scalar>(m_data.env.grid().extent());
  Ptmp.assign(0.0);

  skymap = multi_array<Scalar>(env.params().skymap_Nth,
                               env.params().skymap_Nph);
  skymap.assign(0.0);
  // skymap.sync_to_host();
  l_grid = &env.grid();
}

field_solver_EZ::~field_solver_EZ() {}

void
field_solver_EZ::rk_step(Scalar As, Scalar Bs) {
  int shift = m_env.params().shift_ghost;
  auto &grid = m_env.grid();
  auto &params = m_env.params();
  size_t ijk;
  auto &Ex = m_data.E.data(0);
  auto &Ey = m_data.E.data(1);
  auto &Ez = m_data.E.data(2);
  auto &Bx = m_data.B.data(0);
  auto &By = m_data.B.data(1);
  auto &Bz = m_data.B.data(2);
  auto &dBx = dB.data(0);
  auto &dBy = dB.data(1);
  auto &dBz = dB.data(2);
  auto &dEx = dE.data(0);
  auto &dEy = dE.data(1);
  auto &dEz = dE.data(2);
  auto &jx = m_data.B0.data(0);
  auto &jy = m_data.B0.data(1);
  auto &jz = m_data.B0.data(2);

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Scalar rotBx = dfdy(Bz, ijk) - dfdz(By, ijk);
        Scalar rotBy = dfdz(Bx, ijk) - dfdx(Bz, ijk);
        Scalar rotBz = dfdx(By, ijk) - dfdy(Bx, ijk);
        Scalar rotEx = dfdy(Ez, ijk) - dfdz(Ey, ijk);
        Scalar rotEy = dfdz(Ex, ijk) - dfdx(Ez, ijk);
        Scalar rotEz = dfdx(Ey, ijk) - dfdy(Ex, ijk);

        Scalar divE = dfdx(Ex, ijk) + dfdy(Ey, ijk) + dfdz(Ez, ijk);
        Scalar divB = dfdx(Bx, ijk) + dfdy(By, ijk) + dfdz(Bz, ijk);

        Scalar B2 =
            Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
        if (B2 < TINY) B2 = TINY;

        Scalar Jp =
            (Bx[ijk] * rotBx + By[ijk] * rotBy + Bz[ijk] * rotBz) -
            (Ex[ijk] * rotEx + Ey[ijk] * rotEy + Ez[ijk] * rotEz);
        Scalar Jx = (divE * (Ey[ijk] * Bz[ijk] - Ez[ijk] * By[ijk]) +
                     Jp * Bx[ijk]) /
                    B2;
        Scalar Jy = (divE * (Ez[ijk] * Bx[ijk] - Ex[ijk] * Bz[ijk]) +
                     Jp * By[ijk]) /
                    B2;
        Scalar Jz = (divE * (Ex[ijk] * By[ijk] - Ey[ijk] * Bx[ijk]) +
                     Jp * Bz[ijk]) /
                    B2;

        // Scalar Px = dfdx(P, ijk);
        // Scalar Py = dfdy(P, ijk);
        // Scalar Pz = dfdz(P, ijk);
        Scalar Px = 0.0;
        Scalar Py = 0.0;
        Scalar Pz = 0.0;

        dBx[ijk] = As * dBx[ijk] - params.dt * (rotEx + Px);
        dBy[ijk] = As * dBy[ijk] - params.dt * (rotEy + Py);
        dBz[ijk] = As * dBz[ijk] - params.dt * (rotEz + Pz);

        dEx[ijk] = As * dEx[ijk] + params.dt * (rotBx - Jx);
        dEy[ijk] = As * dEy[ijk] + params.dt * (rotBy - Jy);
        dEz[ijk] = As * dEz[ijk] + params.dt * (rotBz - Jz);

        dP[ijk] = As * dP[ijk] -
                  params.dt *
                      (params.ch2 * divB + m_data.P[ijk] / params.tau);
        jx[ijk] = Jx;
        jy[ijk] = Jy;
        jz[ijk] = Jz;
        m_data.divB[ijk] = divB;
        m_data.divE[ijk] = divE;
      }
    }
  }

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Ex[ijk] = Ex[ijk] + Bs * dEx[ijk];
        Ey[ijk] = Ey[ijk] + Bs * dEy[ijk];
        Ez[ijk] = Ez[ijk] + Bs * dEz[ijk];

        Bx[ijk] = Bx[ijk] + Bs * dBx[ijk];
        By[ijk] = By[ijk] + Bs * dBy[ijk];
        Bz[ijk] = Bz[ijk] + Bs * dBz[ijk];

        m_data.P[ijk] = P[ijk] + Bs * dP[ijk];
      }
    }
  }
}

void
field_solver_EZ::Kreiss_Oliger() {
  int shift = m_env.params().shift_ghost;
  auto &grid = m_env.grid();
  auto &params = m_env.params();
  size_t ijk;
  auto &Ex = m_data.E.data(0);
  auto &Ey = m_data.E.data(1);
  auto &Ez = m_data.E.data(2);
  auto &Bx = m_data.B.data(0);
  auto &By = m_data.B.data(1);
  auto &Bz = m_data.B.data(2);
  auto &Ex_tmp = Etmp.data(0);
  auto &Ey_tmp = Etmp.data(1);
  auto &Ez_tmp = Etmp.data(2);
  auto &Bx_tmp = Btmp.data(0);
  auto &By_tmp = Btmp.data(1);
  auto &Bz_tmp = Btmp.data(2);

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Ex_tmp[ijk] = KO(Ex, ijk, grid);
        Ey_tmp[ijk] = KO(Ey, ijk, grid);
        Ez_tmp[ijk] = KO(Ez, ijk, grid);

        Bx_tmp[ijk] = KO(Bx, ijk, grid);
        By_tmp[ijk] = KO(By, ijk, grid);
        Bz_tmp[ijk] = KO(Bz, ijk, grid);

        P_tmp[ijk] = KO(P, ijk, grid);
      }
    }
  }

  Scalar KO_const = 0.0;

  switch (FFE_DISSIPATION_ORDER) {
    case 4:
      KO_const = -1. / 16;
      break;
    case 6:
      KO_const = -1. / 64;
      break;
  }

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Ex[ijk] -= params.KOeps * KO_const * Ex_tmp[ijk];
        Ey[ijk] -= params.KOeps * KO_const * Ey_tmp[ijk];
        Ez[ijk] -= params.KOeps * KO_const * Ez_tmp[ijk];

        Bx[ijk] -= params.KOeps * KO_const * Bx_tmp[ijk];
        By[ijk] -= params.KOeps * KO_const * By_tmp[ijk];
        Bz[ijk] -= params.KOeps * KO_const * Bz_tmp[ijk];

        P[ijk] -= params.KOeps * KO_const * P_tmp[ijk];
      }
    }
  }
}

void
field_solver_EZ::clean_epar() {
  int shift = m_env.params().shift_ghost;
  auto &grid = m_env.grid();
  size_t ijk;
  auto &Ex = m_data.E.data(0);
  auto &Ey = m_data.E.data(1);
  auto &Ez = m_data.E.data(2);
  auto &Bx = m_data.B.data(0);
  auto &By = m_data.B.data(1);
  auto &Bz = m_data.B.data(2);

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Scalar B2 =
            Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
        if (B2 < TINY) B2 = TINY;
        Scalar EB =
            Ex[ijk] * Bx[ijk] + Ey[ijk] * By[ijk] + Ez[ijk] * Bz[ijk];

        Ex[ijk] = Ex[ijk] - EB / B2 * Bx[ijk];
        Ey[ijk] = Ey[ijk] - EB / B2 * By[ijk];
        Ez[ijk] = Ez[ijk] - EB / B2 * Bz[ijk];
      }
    }
  }
}

void
field_solver_EZ::check_eGTb() {
  int shift = m_env.params().shift_ghost;
  auto &grid = m_env.grid();
  size_t ijk;
  auto &Ex = m_data.E.data(0);
  auto &Ey = m_data.E.data(1);
  auto &Ez = m_data.E.data(2);
  auto &Bx = m_data.B.data(0);
  auto &By = m_data.B.data(1);
  auto &Bz = m_data.B.data(2);

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Scalar B2 =
            Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
        if (B2 < TINY) B2 = TINY;
        Scalar E2 =
            Ex[ijk] * Ex[ijk] + Ey[ijk] * Ey[ijk] + Ez[ijk] * Ez[ijk];

        if (E2 > B2) {
          Scalar s = sqrt(B2 / E2);
          Ex[ijk] *= s;
          Ey[ijk] *= s;
          Ez[ijk] *= s;
        }
      }
    }
  }
}

void
field_solver_EZ::boundary_pulsar(Scalar t) {
  
}


}  // namespace Coffee
