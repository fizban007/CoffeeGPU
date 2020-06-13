#include "field_solver_EZ.h"
#include "algorithms/damping_boundary.h"
#include "algorithms/finite_diff_simd.h"
#include "algorithms/pulsar.h"
#include "utils/simd.h"
#include "utils/timer.h"
#include <omp.h>

namespace Coffee {

using namespace simd;

const Grid *l_grid;

Vec_f_t
dfdx(const multi_array<Scalar> &f, int ijk) {
  return df1_simd(f.host_ptr(), ijk, 1, l_grid->inv_delta[0]);
}

Vec_f_t
dfdy(const multi_array<Scalar> &f, int ijk) {
  return df1_simd(f.host_ptr(), ijk, l_grid->dims[0],
                  l_grid->inv_delta[1]);
}

Vec_f_t
dfdz(const multi_array<Scalar> &f, int ijk) {
  return df1_simd(f.host_ptr(), ijk, l_grid->dims[0] * l_grid->dims[1],
                  l_grid->inv_delta[2]);
}

#ifdef USE_SIMD
inline Vec_f_t
KO(const multi_array<Scalar> &f, int ijk) {
  return KO_simd(f.host_ptr(), ijk, *l_grid);
}
#else
inline Scalar
KO(const multi_array<Scalar> &f, int ijk) {
  return KO(f.host_ptr(), ijk, *l_grid);
}
#endif

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

  Bbg = vector_field<Scalar>(m_data.env.grid());
  Bbg.copy_stagger(m_data.B);
  // If damp to vacuum background field
  Bbg.copy_from(m_data.B);
  // If damp to zero
  // Bbg.initialize();

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
      // #pragma omp simd simdlen(8)
      // TODO: Need to consider case where iteration is not a multiple
      // of vec_width
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i += vec_width) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        auto rotBx = dfdy(Bz, ijk) - dfdz(By, ijk);
        auto rotBy = dfdz(Bx, ijk) - dfdx(Bz, ijk);
        auto rotBz = dfdx(By, ijk) - dfdy(Bx, ijk);
        auto rotEx = dfdy(Ez, ijk) - dfdz(Ey, ijk);
        auto rotEy = dfdz(Ex, ijk) - dfdx(Ez, ijk);
        auto rotEz = dfdx(Ey, ijk) - dfdy(Ex, ijk);

        auto divE = dfdx(Ex, ijk) + dfdy(Ey, ijk) + dfdz(Ez, ijk);
        auto divB = dfdx(Bx, ijk) + dfdy(By, ijk) + dfdz(Bz, ijk);

        divE.store(m_data.divE.host_ptr() + ijk);
        divB.store(m_data.divB.host_ptr() + ijk);

        Vec_f_t bxvec, byvec, bzvec;
        bxvec.load(Bx.host_ptr() + ijk);
        byvec.load(By.host_ptr() + ijk);
        bzvec.load(Bz.host_ptr() + ijk);

        Vec_f_t exvec, eyvec, ezvec;
        exvec.load(Ex.host_ptr() + ijk);
        eyvec.load(Ey.host_ptr() + ijk);
        ezvec.load(Ez.host_ptr() + ijk);

        Vec_f_t Jx, Jy, Jz;
        // Scalar B2 =
        //     Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] *
        //     Bz[ijk];
        auto B2 = bxvec * bxvec + byvec * byvec + bzvec * bzvec;
        // if (B2 < TINY) B2 = TINY;
        B2 = max(B2, TINY);

        if (params.use_edotb_damping) {
          auto E2 = exvec * exvec + eyvec * eyvec + ezvec * ezvec;
          auto chi2 = B2 - E2;
          auto EdotB = exvec * bxvec + eyvec * byvec + ezvec * bzvec;
          auto E02 =
              0.5 * (sqrt(chi2 * chi2 + 4.0 * EdotB * EdotB) - chi2) +
              B2;

          auto Jp = (bxvec * rotBx + byvec * rotBy + bzvec * rotBz) -
                    (exvec * rotEx + eyvec * rotEy + ezvec * rotEz) +
                    EdotB * (params.damp_gamma / params.dt);
          Jx = divE * (eyvec * bzvec - ezvec * byvec) / E02 +
               Jp * bxvec / B2;
          Jy = divE * (ezvec * bxvec - exvec * bzvec) / E02 +
               Jp * byvec / B2;
          Jz = divE * (exvec * byvec - eyvec * bxvec) / E02 +
               Jp * bzvec / B2;
        } else {
          auto Jp = (bxvec * rotBx + byvec * rotBy + bzvec * rotBz) -
                    (exvec * rotEx + eyvec * rotEy + ezvec * rotEz);
          Jx = (divE * (eyvec * bzvec - ezvec * byvec) + Jp * bxvec) /
               B2;
          Jy = (divE * (ezvec * bxvec - exvec * bzvec) + Jp * byvec) /
               B2;
          Jz = (divE * (exvec * byvec - eyvec * bxvec) + Jp * bzvec) /
               B2;
        }

        Jx.store(jx.host_ptr() + ijk);
        Jy.store(jy.host_ptr() + ijk);
        Jz.store(jz.host_ptr() + ijk);
        // Scalar Px = dfdx(P, ijk);
        // Scalar Py = dfdy(P, ijk);
        // Scalar Pz = dfdz(P, ijk);
        // Scalar Px = 0.0;
        // Scalar Py = 0.0;
        // Scalar Pz = 0.0;
        Vec_f_t dbxvec, dbyvec, dbzvec;
        dbxvec.load(dBx.host_ptr() + ijk);
        dbyvec.load(dBy.host_ptr() + ijk);
        dbzvec.load(dBz.host_ptr() + ijk);

        // dBx[ijk] = As * dBx[ijk] - params.dt * rotEx;
        // dBy[ijk] = As * dBy[ijk] - params.dt * rotEy;
        // dBz[ijk] = As * dBz[ijk] - params.dt * rotEz;

        // dEx[ijk] = As * dEx[ijk] + params.dt * (rotBx - Jx);
        // dEy[ijk] = As * dEy[ijk] + params.dt * (rotBy - Jy);
        // dEz[ijk] = As * dEz[ijk] + params.dt * (rotBz - Jz);
        dbxvec = dbxvec * As - rotEx * params.dt;
        dbyvec = dbyvec * As - rotEy * params.dt;
        dbzvec = dbzvec * As - rotEz * params.dt;
        dbxvec.store(dBx.host_ptr() + ijk);
        dbyvec.store(dBy.host_ptr() + ijk);
        dbzvec.store(dBz.host_ptr() + ijk);

        Vec_f_t dexvec, deyvec, dezvec;
        dexvec.load(dEx.host_ptr() + ijk);
        deyvec.load(dEy.host_ptr() + ijk);
        dezvec.load(dEz.host_ptr() + ijk);

        dexvec = dexvec * As + (rotBx - Jx) * params.dt;
        deyvec = deyvec * As + (rotBy - Jy) * params.dt;
        dezvec = dezvec * As + (rotBz - Jz) * params.dt;
        dexvec.store(dEx.host_ptr() + ijk);
        deyvec.store(dEy.host_ptr() + ijk);
        dezvec.store(dEz.host_ptr() + ijk);

        // dP[ijk] = As * dP[ijk] -
        //           params.dt *
        //               (params.ch2 * divB + m_data.P[ijk] /
        //               params.tau);
        // jx[ijk] = Jx;
        // jy[ijk] = Jy;
        // jz[ijk] = Jz;
        // m_data.divB[ijk] = divB;
        // m_data.divE[ijk] = divE;
      }
    }
  }

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      // #pragma omp simd simdlen(8)
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i += vec_width) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Vec_f_t dexvec, deyvec, dezvec;
        dexvec.load(dEx.host_ptr() + ijk);
        deyvec.load(dEy.host_ptr() + ijk);
        dezvec.load(dEz.host_ptr() + ijk);

        Vec_f_t exvec, eyvec, ezvec;
        exvec.load(Ex.host_ptr() + ijk);
        eyvec.load(Ey.host_ptr() + ijk);
        ezvec.load(Ez.host_ptr() + ijk);

        exvec = mul_add(dexvec, Bs, exvec);
        eyvec = mul_add(deyvec, Bs, eyvec);
        ezvec = mul_add(dezvec, Bs, ezvec);

        exvec.store(Ex.host_ptr() + ijk);
        eyvec.store(Ey.host_ptr() + ijk);
        ezvec.store(Ez.host_ptr() + ijk);

        dexvec.load(dBx.host_ptr() + ijk);
        deyvec.load(dBy.host_ptr() + ijk);
        dezvec.load(dBz.host_ptr() + ijk);

        exvec.load(Bx.host_ptr() + ijk);
        eyvec.load(By.host_ptr() + ijk);
        ezvec.load(Bz.host_ptr() + ijk);

        exvec = mul_add(dexvec, Bs, exvec);
        eyvec = mul_add(deyvec, Bs, eyvec);
        ezvec = mul_add(dezvec, Bs, ezvec);

        exvec.store(Bx.host_ptr() + ijk);
        eyvec.store(By.host_ptr() + ijk);
        ezvec.store(Bz.host_ptr() + ijk);

        // Ex[ijk] += Bs * dEx[ijk];
        // Ey[ijk] += Bs * dEy[ijk];
        // Ez[ijk] += Bs * dEz[ijk];

        // Bx[ijk] += Bs * dBx[ijk];
        // By[ijk] += Bs * dBy[ijk];
        // Bz[ijk] += Bs * dBz[ijk];

        // m_data.P[ijk] = m_data.P[ijk] + Bs * dP[ijk];
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
      // #pragma omp simd simdlen(8)
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i += vec_width) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        auto extmp = KO(Ex, ijk);
        auto eytmp = KO(Ey, ijk);
        auto eztmp = KO(Ez, ijk);
        extmp.store(Ex_tmp.host_ptr() + ijk);
        eytmp.store(Ey_tmp.host_ptr() + ijk);
        eztmp.store(Ez_tmp.host_ptr() + ijk);
        // Ex_tmp[ijk] = KO(Ex, ijk);
        // Ey_tmp[ijk] = KO(Ey, ijk);
        // Ez_tmp[ijk] = KO(Ez, ijk);

        extmp = KO(Bx, ijk);
        eytmp = KO(By, ijk);
        eztmp = KO(Bz, ijk);
        extmp.store(Bx_tmp.host_ptr() + ijk);
        eytmp.store(By_tmp.host_ptr() + ijk);
        eztmp.store(Bz_tmp.host_ptr() + ijk);
        // Bx_tmp[ijk] = KO(Bx, ijk);
        // By_tmp[ijk] = KO(By, ijk);
        // Bz_tmp[ijk] = KO(Bz, ijk);

        // Ptmp[ijk] = KO(m_data.P, ijk);
      }
    }
  }

  Scalar KO_const = 0.0;

  switch (FFE_DISSIPATION_ORDER) {
    case 4:
      KO_const = 1.0 / 16.0;
      break;
    case 6:
      KO_const = -1.0 / 64.0;
      break;
  }

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i += vec_width) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Vec_f_t extmp, eytmp, eztmp;
        extmp.load(Ex_tmp.host_ptr() + ijk);
        eytmp.load(Ey_tmp.host_ptr() + ijk);
        eztmp.load(Ez_tmp.host_ptr() + ijk);

        Vec_f_t exvec, eyvec, ezvec;
        exvec.load(Ex.host_ptr() + ijk);
        eyvec.load(Ey.host_ptr() + ijk);
        ezvec.load(Ez.host_ptr() + ijk);

        exvec = exvec + extmp * (-params.KOeps * KO_const);
        eyvec = eyvec + eytmp * (-params.KOeps * KO_const);
        ezvec = ezvec + eztmp * (-params.KOeps * KO_const);

        exvec.store(Ex.host_ptr() + ijk);
        eyvec.store(Ey.host_ptr() + ijk);
        ezvec.store(Ez.host_ptr() + ijk);
        // Ex[ijk] -= params.KOeps * KO_const * Ex_tmp[ijk];
        // Ey[ijk] -= params.KOeps * KO_const * Ey_tmp[ijk];
        // Ez[ijk] -= params.KOeps * KO_const * Ez_tmp[ijk];

        extmp.load(Bx_tmp.host_ptr() + ijk);
        eytmp.load(By_tmp.host_ptr() + ijk);
        eztmp.load(Bz_tmp.host_ptr() + ijk);

        exvec.load(Bx.host_ptr() + ijk);
        eyvec.load(By.host_ptr() + ijk);
        ezvec.load(Bz.host_ptr() + ijk);

        exvec = exvec + extmp * (-params.KOeps * KO_const);
        eyvec = eyvec + eytmp * (-params.KOeps * KO_const);
        ezvec = ezvec + eztmp * (-params.KOeps * KO_const);

        exvec.store(Bx.host_ptr() + ijk);
        eyvec.store(By.host_ptr() + ijk);
        ezvec.store(Bz.host_ptr() + ijk);
        // Bx[ijk] -= params.KOeps * KO_const * Bx_tmp[ijk];
        // By[ijk] -= params.KOeps * KO_const * By_tmp[ijk];
        // Bz[ijk] -= params.KOeps * KO_const * Bz_tmp[ijk];

        // m_data.P[ijk] -= params.KOeps * KO_const * Ptmp[ijk];
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
#pragma omp simd simdlen(8)
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
#pragma omp simd simdlen(8)
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Scalar B2 =
            Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] * Bz[ijk];
        if (B2 < TINY) B2 = TINY;
        Scalar E2 =
            Ex[ijk] * Ex[ijk] + Ey[ijk] * Ey[ijk] + Ez[ijk] * Ez[ijk];

        if (E2 > B2) {
          Scalar s = std::sqrt(B2 / E2);
          Ex[ijk] *= s;
          Ey[ijk] *= s;
          Ez[ijk] *= s;
        }
      }
    }
  }
}

void
field_solver_EZ::clean_epar_check_eGTb() {
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
           i < grid.dims[0] - grid.guard[0] + shift; i += vec_width) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Vec_f_t bxvec, byvec, bzvec, exvec, eyvec, ezvec;
        bxvec.load(Bx.host_ptr() + ijk);
        byvec.load(By.host_ptr() + ijk);
        bzvec.load(Bz.host_ptr() + ijk);
        exvec.load(Ex.host_ptr() + ijk);
        eyvec.load(Ey.host_ptr() + ijk);
        ezvec.load(Ez.host_ptr() + ijk);

        auto B2 = bxvec * bxvec + byvec * byvec + bzvec * bzvec;
        B2 = max(B2, TINY);
        auto EB = bxvec * exvec + byvec * eyvec + bzvec * ezvec;

        exvec = exvec - EB * bxvec / B2;
        eyvec = eyvec - EB * byvec / B2;
        ezvec = ezvec - EB * bzvec / B2;

        // Scalar B2 = Bx[ijk] * Bx[ijk] + By[ijk] * By[ijk] + Bz[ijk] *
        // Bz[ijk]; if (B2 < TINY)
        //   B2 = TINY;
        // Scalar EB = Ex[ijk] * Bx[ijk] + Ey[ijk] * By[ijk] + Ez[ijk] *
        // Bz[ijk];

        auto E2 = exvec * exvec + eyvec * eyvec + ezvec * ezvec;
        auto s = sqrt(B2 / E2);

        auto egtb = E2 > B2;
        exvec = select(egtb, exvec * s, exvec);
        eyvec = select(egtb, eyvec * s, eyvec);
        ezvec = select(egtb, ezvec * s, ezvec);

        exvec.store(Ex.host_ptr() + ijk);
        eyvec.store(Ey.host_ptr() + ijk);
        ezvec.store(Ez.host_ptr() + ijk);
        // Ex[ijk] = Ex[ijk] - EB / B2 * Bx[ijk];
        // Ey[ijk] = Ey[ijk] - EB / B2 * By[ijk];
        // Ez[ijk] = Ez[ijk] - EB / B2 * Bz[ijk];
      }
    }
  }
}

void
field_solver_EZ::boundary_absorbing() {
  auto &params = m_env.params();
  auto &grid = m_env.grid();
  damping_boundary(Etmp, Btmp, Bbg, m_data.E, m_data.B, Ptmp, m_data.P,
                   params.shift_ghost, grid, params);
}

Scalar
wpert(Scalar t, Scalar r, Scalar th, Scalar tp_start, Scalar tp_end,
      Scalar dw0, Scalar nT, Scalar rpert1, Scalar rpert2) {
  Scalar th1 = acos(std::sqrt(1.0 - 1.0 / rpert1));
  Scalar th2 = acos(std::sqrt(1.0 - 1.0 / rpert2));
  if (th1 > th2) {
    Scalar tmp = th1;
    th1 = th2;
    th2 = tmp;
  }
  Scalar mu = (th1 + th2) / 2.0;
  Scalar s = (mu - th1) / 3.0;
  if (t >= tp_start && t <= tp_end && th >= th1 && th <= th2)
    return dw0 * exp(-0.5 * square((th - mu) / s)) *
           sin((t - tp_start) * 2.0 * M_PI * nT / (tp_end - tp_start));
  else
    return 0;
}

void
field_solver_EZ::boundary_pulsar(Scalar t) {
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

  Scalar rl = 2.0 * params.radius;
  Scalar ri = 0.5 * params.radius;
  Scalar scaleEpar = 0.5 * grid.delta[0];
  Scalar scaleEperp = 0.25 * grid.delta[0];
  Scalar scaleBperp = scaleEpar;
  Scalar scaleBpar = scaleBperp;
  Scalar d1 = 4.0 * grid.delta[0];
  Scalar d0 = 0;
  Scalar phase = params.omega * t;

  Scalar Bxnew, Bynew, Bznew, Exnew, Eynew, Eznew;

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      // #pragma omp simd simdlen(8)
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + (j + k * grid.dims[1]) * grid.dims[0];

        Scalar x = grid.pos(0, i, 1);
        Scalar y = grid.pos(1, j, 1);
        Scalar z = grid.pos(2, k, 1);
        Scalar r2 = x * x + y * y + z * z;
        if (r2 < TINY) r2 = TINY;
        Scalar r = std::sqrt(r2);
        Scalar th = acos(z / r);

        if (r < rl) {
          Scalar wpert0 =
              wpert(t, r, th, params.tp_start, params.tp_end, params.dw0,
                    params.nT, params.rpert1, params.rpert2);
          Scalar wpert1 = wpert(t, r, th, params.tp_start1,
                                params.tp_end1, params.dw1, params.nT1,
                                params.rpert11, params.rpert21);
          Scalar w = params.omega + wpert0 + wpert1;

          // Scalar bxn = params.b0 * cube(params.radius) *
          //              dipole_x(x, y, z, params.alpha, phase);
          // Scalar byn = params.b0 * cube(params.radius) *
          //              dipole_y(x, y, z, params.alpha, phase);
          // Scalar bzn = params.b0 * cube(params.radius) *
          //              dipole_z(x, y, z, params.alpha, phase);
          Scalar bxn =
              params.b0 *
              // quadru_dipole(x, y, z, params.p1, params.p2, params.p3,
              //               params.q11, params.q12, params.q13,
              //               params.q22, params.q23,
              //               params.q_offset_x, params.q_offset_y,
              //               params.q_offset_z, phase, 0);
              dipole2(x, y, z, params.p1, params.p2, params.p3, phase,
                      0);
          Scalar byn =
              params.b0 *
              // quadru_dipole(x, y, z, params.p1, params.p2, params.p3,
              //               params.q11, params.q12, params.q13,
              //               params.q22, params.q23,
              //               params.q_offset_x, params.q_offset_y,
              //               params.q_offset_z, phase, 1);
              dipole2(x, y, z, params.p1, params.p2, params.p3, phase,
                      1);
          Scalar bzn =
              params.b0 *
              // quadru_dipole(x, y, z, params.p1, params.p2, params.p3,
              //               params.q11, params.q12, params.q13,
              //               params.q22, params.q23,
              //               params.q_offset_x, params.q_offset_y,
              //               params.q_offset_z, phase, 2);
              dipole2(x, y, z, params.p1, params.p2, params.p3, phase,
                      1);
          Scalar s = shape(r, params.radius - d1, scaleBperp);
          Scalar bn_dot_r = bxn * x + byn * y + bzn * z;
          Scalar B_dot_r = Bx[ijk] * x + By[ijk] * y + Bz[ijk] * z;
          Bxnew = bn_dot_r * x / r2 * s + B_dot_r * x / r2 * (1 - s);
          Bynew = bn_dot_r * y / r2 * s + B_dot_r * y / r2 * (1 - s);
          Bznew = bn_dot_r * z / r2 * s + B_dot_r * z / r2 * (1 - s);
          s = shape(r, params.radius - d1, scaleBpar);
          Bxnew += (bxn - bn_dot_r * x / r2) * s +
                   (Bx[ijk] - B_dot_r * x / r2) * (1 - s);
          Bynew += (byn - bn_dot_r * y / r2) * s +
                   (By[ijk] - B_dot_r * y / r2) * (1 - s);
          Bznew += (bzn - bn_dot_r * z / r2) * s +
                   (Bz[ijk] - B_dot_r * z / r2) * (1 - s);

          Bx[ijk] = Bxnew;
          By[ijk] = Bynew;
          Bz[ijk] = Bznew;

          Scalar vx = -w * y;
          Scalar vy = w * x;
          Scalar exn = -vy * Bz[ijk];
          Scalar eyn = vx * Bz[ijk];
          Scalar ezn = -vx * By[ijk] + vy * Bx[ijk];
          Scalar en_dot_r = (exn * x + eyn * y + ezn * z);
          Scalar E_dot_r = (Ex[ijk] * x + Ey[ijk] * y + Ez[ijk] * z);
          s = shape(r, params.radius - d0, scaleEperp);
          Exnew = en_dot_r * x / r2 * s + E_dot_r * x / r2 * (1 - s);
          Eynew = en_dot_r * y / r2 * s + E_dot_r * y / r2 * (1 - s);
          Eznew = en_dot_r * z / r2 * s + E_dot_r * z / r2 * (1 - s);
          s = shape(r, params.radius - d0, scaleEpar);
          Exnew += (exn - en_dot_r * x / r2) * s +
                   (Ex[ijk] - E_dot_r * x / r2) * (1 - s);
          Eynew += (eyn - en_dot_r * y / r2) * s +
                   (Ey[ijk] - E_dot_r * y / r2) * (1 - s);
          Eznew += (ezn - en_dot_r * z / r2) * s +
                   (Ez[ijk] - E_dot_r * z / r2) * (1 - s);
          // Bx[ijk] = Bxnew;
          // By[ijk] = Bynew;
          // Bz[ijk] = Bznew;
          Ex[ijk] = Exnew;
          Ey[ijk] = Eynew;
          Ez[ijk] = Eznew;
          if (r < ri) {
            Bx[ijk] = bxn;
            By[ijk] = byn;
            Bz[ijk] = bzn;
            Ex[ijk] = exn;
            Ey[ijk] = eyn;
            Ez[ijk] = ezn;
          }
        }
      }
    }
  }
}

void
field_solver_EZ::evolve_fields(Scalar time) {
  Scalar As[5] = {0, -0.4178904745, -1.192151694643, -1.697784692471,
                  -1.514183444257};
  Scalar Bs[5] = {0.1496590219993, 0.3792103129999, 0.8229550293869,
                  0.6994504559488, 0.1530572479681};
  Scalar cs[5] = {0, 0.1496590219993, 0.3704009573644, 0.6222557631345,
                  0.9582821306784};

  Etmp.copy_from(m_data.E);
  Btmp.copy_from(m_data.B);
  Ptmp.copy_from(m_data.P);

  for (int i = 0; i < 5; ++i) {
    timer::stamp();
    rk_step(As[i], Bs[i]);

    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("rk_step", "ms");

    timer::stamp();
    // if (m_env.params().clean_ep)
    //   clean_epar();
    // if (m_env.params().check_egb)
    //   check_eGTb();
    clean_epar_check_eGTb();
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("clean/check", "ms");

    timer::stamp();
    boundary_pulsar(time + cs[i] * m_env.params().dt);
    if (i == 4) boundary_absorbing();

    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("boundary", "ms");

    timer::stamp();
    m_env.send_guard_cells(m_data);
    // m_env.send_guard_cell_array(P);
    if (m_env.rank() == 0)
      timer::show_duration_since_stamp("communication", "ms");
  }

  timer::stamp();
  Kreiss_Oliger();
  clean_epar_check_eGTb();
  // if (m_env.params().clean_ep)
  //   clean_epar();
  // if (m_env.params().check_egb)
  //   check_eGTb();
  boundary_pulsar(time + m_env.params().dt);

  m_env.send_guard_cells(m_data);
  // m_env.send_guard_cell_array(P);
  if (m_env.rank() == 0)
    timer::show_duration_since_stamp("Kreiss Oliger", "ms");
}

}  // namespace Coffee
