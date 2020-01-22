#include "damping_boundary.h"

namespace Coffee {

void
damping_boundary(const vector_field<Scalar>& En,
                 const vector_field<Scalar>& Bn,
                 vector_field<Scalar>& E, vector_field<Scalar>& B,
                 multi_array<Scalar>& Pn, multi_array<Scalar>& P,
                 int shift, const Grid& grid,
                 const sim_params& params) {
  auto& ex = E.data(0);
  auto& ey = E.data(1);
  auto& ez = E.data(2);
  auto& enx = En.data(0);
  auto& eny = En.data(1);
  auto& enz = En.data(2);
  auto& bx = B.data(0);
  auto& by = B.data(1);
  auto& bz = B.data(2);
  auto& bnx = Bn.data(0);
  auto& bny = Bn.data(1);
  auto& bnz = Bn.data(2);
  Scalar x, y, z;
  Scalar sigx = 0.0, sigy = 0.0, sigz = 0.0, sig = 0.0;
  size_t ijk = 0;

  Scalar xh =
      params.lower[0] + params.size[0] - params.pml[0] * grid.delta[0];
  Scalar xl = params.lower[0] + params.pml[0] * grid.delta[0];
  Scalar yh =
      params.lower[1] + params.size[1] - params.pml[1] * grid.delta[1];
  Scalar yl = params.lower[1] + params.pml[1] * grid.delta[1];
  Scalar zh =
      params.lower[2] + params.size[2] - params.pml[2] * grid.delta[2];
  Scalar zl = params.lower[2] + params.pml[2] * grid.delta[2];

  for (int k = grid.guard[2] - shift;
       k < grid.dims[2] - grid.guard[2] + shift; k++) {
    for (int j = grid.guard[1] - shift;
         j < grid.dims[1] - grid.guard[1] + shift; j++) {
      for (int i = grid.guard[0] - shift;
           i < grid.dims[0] - grid.guard[0] + shift; i++) {
        ijk = i + j * grid.dims[0] + k * grid.dims[0] * grid.dims[1];
        x = grid.pos(0, i, 1);
        y = grid.pos(1, j, 1);
        z = grid.pos(2, k, 1);

        if (x > xh || x < xl || y > yh || y < yl || z > zh || z < zl) {
          // if (x > xh || y < yl || y > yh) {
          sigx = pmlsigma(x, xl, xh, params.pmllen * grid.delta[0],
                          params.sigpml);
          sigy = pmlsigma(y, yl, yh, params.pmllen * grid.delta[1],
                          params.sigpml);
          sigz = pmlsigma(z, zl, zh, params.pmllen * grid.delta[2],
                          params.sigpml);
          sig = sigx + sigy + sigz;
          // sig = sigx + sigy;
          if (sig > TINY) {
            ex[ijk] = exp(-sig) * enx[ijk] +
                      (1.0 - exp(-sig)) / sig * (ex[ijk] - enx[ijk]);
            ey[ijk] = exp(-sig) * eny[ijk] +
                      (1.0 - exp(-sig)) / sig * (ey[ijk] - eny[ijk]);
            ez[ijk] = exp(-sig) * enz[ijk] +
                      (1.0 - exp(-sig)) / sig * (ez[ijk] - enz[ijk]);
            bx[ijk] = exp(-sig) * bnx[ijk] +
                      (1.0 - exp(-sig)) / sig * (bx[ijk] - bnx[ijk]);
            by[ijk] = exp(-sig) * bny[ijk] +
                      (1.0 - exp(-sig)) / sig * (by[ijk] - bny[ijk]);
            bz[ijk] = exp(-sig) * bnz[ijk] +
                      (1.0 - exp(-sig)) / sig * (bz[ijk] - bnz[ijk]);
            // P[ijk] = exp(-sig) * Pn[ijk] +
            //          (1.0 - exp(-sig)) / sig * (P[ijk] - Pn[ijk]);
          }
        }
      }
    }
  }
}

}  // namespace Coffee
