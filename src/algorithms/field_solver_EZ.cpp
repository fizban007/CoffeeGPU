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

}  // namespace Coffee
