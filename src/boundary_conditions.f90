#include "defs.f90"

REAL(kind = MPREC) FUNCTION lambda_wave(igl, jgl, kgl)
	! `lambda_wave` is `dt * lambda / 2`
	IMPLICIT NONE
	REAL(kind = MPREC), INTENT(IN) :: igl, jgl, kgl
	REAL(kind = MPREC) :: layer_size, dr, dr_Px, dr_Mx, dr_Py, dr_My, dr_Pz, dr_Mz

	layer_size = dr_layer

	! X direction
	IF (bc_Mx .EQ. 0) THEN
		dr_Mx = igl - 1
	ELSE
		dr_Mx = layer_size + 1
	END IF
	IF (bc_Px .EQ. 0) THEN
		dr_Px = mx0 - igl
	ELSE
		dr_Px = layer_size + 1
	END IF
	! Y direction
	IF (bc_My .EQ. 0) THEN
		dr_My = jgl - 1
	ELSE
		dr_My = layer_size + 1
	END IF
	IF (bc_Py .EQ. 0) THEN
		dr_Py = my0 - jgl
	ELSE
		dr_Py = layer_size + 1
	END IF
	! Z direction
	IF (bc_Mz .EQ. 0) THEN
		dr_Mz = kgl - 1
	ELSE
		dr_Mz = layer_size + 1
	END IF
	IF (bc_Pz .EQ. 0) THEN
		dr_Pz = mz0 - kgl
	ELSE
		dr_Pz = layer_size + 1
	END IF

	dr = MIN(dr_Px, dr_Mx, dr_Py, dr_My, dr_Pz, dr_Mz)

	IF (dr .LT. layer_size) THEN
		lambda_wave = ((layer_size - dr) / (layer_size))
	ELSE
		lambda_wave = 0
	END IF
END FUNCTION lambda_wave

REAL(kind = MPREC) FUNCTION sigmoid(lambda)
	IMPLICIT NONE
	REAL(kind = MPREC), INTENT(IN) :: lambda
	IF (lambda .GT. 0) THEN
		sigmoid = MIN(MAX((tanh(one_ / lambda + lambda) - tanh(2 * one_)) / (one_ - tanh(2 * one_)), zero_), one_)
	ELSE
		sigmoid = one_
	END IF
END FUNCTION sigmoid

SUBROUTINE boundary_conditions(sstep)
	IMPLICIT NONE
	INTEGER :: sstep
	INTEGER :: i, j, k, iglob, jglob, kglob, dd
	REAL(kind = MPREC) :: sx, sy, sz
	REAL(kind = MPREC) :: lam, lam_coeff

	DOUBLE PRECISION t0, t1

	! absorbing boundary conditions
	IF ((sstep .EQ. 3) .AND. (absorb)) THEN
		DO k = 1, mz
			DO j = 1, my
				DO i = 1, mx
					iglob = i
					jglob = j + MODULO(rank, sizey) * (my - 5)
					kglob = k + (rank / sizey) * (mz - 5)

					sx = REAL(iglob, MPREC)
					sy = REAL(jglob + 0.5, MPREC)
					sz = REAL(kglob + 0.5, MPREC)
					lam = lambda_wave(sx, sy, sz)
					f(i, j, k).bx = f(i, j, k).bx * sigmoid(lam) + f0(i, j, k).bx0 * (one_ - sigmoid(lam))
					sx = REAL(iglob + 0.5, MPREC)
					sy = REAL(jglob, MPREC)
					sz = REAL(kglob + 0.5, MPREC)
					lam = lambda_wave(sx, sy, sz)
					f(i, j, k).by = f(i, j, k).by * sigmoid(lam) + f0(i, j, k).by0 * (one_ - sigmoid(lam))
					sx = REAL(iglob + 0.5, MPREC)
					sy = REAL(jglob + 0.5, MPREC)
					sz = REAL(kglob, MPREC)
					lam = lambda_wave(sx, sy, sz)
					f(i, j, k).bz = f(i, j, k).bz * sigmoid(lam) + f0(i, j, k).bz0 * (one_ - sigmoid(lam))

					sx = REAL(iglob + 0.5, MPREC)
					sy = REAL(jglob, MPREC)
					sz = REAL(kglob, MPREC)
					lam = lambda_wave(sx, sy, sz)
					f(i, j, k).ex = f(i, j, k).ex * sigmoid(lam)
					sx = REAL(iglob, MPREC)
					sy = REAL(jglob + 0.5, MPREC)
					sz = REAL(kglob, MPREC)
					lam = lambda_wave(sx, sy, sz)
					f(i, j, k).ey = f(i, j, k).ey * sigmoid(lam)
					sx = REAL(iglob, MPREC)
					sy = REAL(jglob, MPREC)
					sz = REAL(kglob + 0.5, MPREC)
					f(i, j, k).ez = f(i, j, k).ez * sigmoid(lam)
				END DO
			END DO
		END DO
	END IF

	!exchange ghost zones

	IF (rank .EQ. 0) t0 = mpi_wtime()
	bcmult = 1 !to correctly copy periodic BC in presence of b0
	CALL copylayrx_f(f, mx, my, mz, 2, mx - 3, mx - 2, 3)
	CALL copylayrx_f(f, mx, my, mz, 1, mx - 4, mx - 1, 4)

	CALL copylayrz_f(f, mx, my, mz, 2, mz - 3, mz - 2, 3, rank, sizey, sizez)
	CALL copylayrz_f(f, mx, my, mz, 1, mz - 4, mz - 1, 4, rank, sizey, sizez)

	CALL copylayry_f(f, mx, my, mz, 2, my - 3, my - 2, 3, rank, sizey, sizez)
	CALL copylayry_f(f, mx, my, mz, 1, my - 4, my - 1, 4, rank, sizey, sizez)

	IF (rank .EQ. 0) PRINT *, "exchange f", mpi_wtime() - t0

	! radiation boundary conditions
	! setting dE / dn = 0 & dB / dn = 0
	IF (sstep .EQ. 3) THEN
		DO k = 1, mz
			DO j = 1, my
				DO i = 1, mx
					iglob = i
					jglob = j + MODULO(rank, sizey) * (my - 5)
					kglob = k + (rank / sizey) * (mz - 5)

					IF ((iglob .LE. 3) .AND. (bc_Mx .EQ. 0)) THEN
						f(i, j, k).bx = f(4, j, k).bx
						f(i, j, k).by = f(4, j, k).by
						f(i, j, k).bz = f(4, j, k).bz
						f(i, j, k).ex = f(4, j, k).ex
						f(i, j, k).ey = f(4, j, k).ey
						f(i, j, k).ez = f(4, j, k).ez
					ELSE IF ((iglob .GE. mx0 - 3) .AND. (bc_Px .EQ. 0)) THEN
						f(i, j, k).bx = f(mx - 4, j, k).bx
						f(i, j, k).by = f(mx - 4, j, k).by
						f(i, j, k).bz = f(mx - 4, j, k).bz
						f(i, j, k).ex = f(mx - 4, j, k).ex
						f(i, j, k).ey = f(mx - 4, j, k).ey
						f(i, j, k).ez = f(mx - 4, j, k).ez
					END IF

					IF ((jglob .LE. 3) .AND. (bc_My .EQ. 0)) THEN
						f(i, j, k).bx = f(i, 4, k).bx
						f(i, j, k).by = f(i, 4, k).by
						f(i, j, k).bz = f(i, 4, k).bz
						f(i, j, k).ex = f(i, 4, k).ex
						f(i, j, k).ey = f(i, 4, k).ey
						f(i, j, k).ez = f(i, 4, k).ez
					ELSE IF ((jglob .GE. my0 - 3) .AND. (bc_Px .EQ. 0)) THEN
						f(i, j, k).bx = f(i, my - 4, k).bx
						f(i, j, k).by = f(i, my - 4, k).by
						f(i, j, k).bz = f(i, my - 4, k).bz
						f(i, j, k).ex = f(i, my - 4, k).ex
						f(i, j, k).ey = f(i, my - 4, k).ey
						f(i, j, k).ez = f(i, my - 4, k).ez
					END IF

					IF ((kglob .LE. 3) .AND. (bc_Mz .EQ. 0)) THEN
						f(i, j, k).bx = f(i, j, 4).bx
						f(i, j, k).by = f(i, j, 4).by
						f(i, j, k).bz = f(i, j, 4).bz
						f(i, j, k).ex = f(i, j, 4).ex
						f(i, j, k).ey = f(i, j, 4).ey
						f(i, j, k).ez = f(i, j, 4).ez
					ELSE IF ((kglob .GE. mz0 - 3) .AND. (bc_Pz .EQ. 0)) THEN
						f(i, j, k).bx = f(i, j, mz - 4).bx
						f(i, j, k).by = f(i, j, mz - 4).by
						f(i, j, k).bz = f(i, j, mz - 4).bz
						f(i, j, k).ex = f(i, j, mz - 4).ex
						f(i, j, k).ey = f(i, j, mz - 4).ey
						f(i, j, k).ez = f(i, j, mz - 4).ez
					END IF
				END DO
			END DO
		END DO
	END IF

	IF (sstep .EQ. 3) THEN
		CALL user_bc()
	END IF

END SUBROUTINE boundary_conditions
