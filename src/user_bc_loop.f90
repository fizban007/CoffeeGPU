#include "defs.f90"

REAL(kind = MPREC) FUNCTION twist_profile(r, t)
	IMPLICIT NONE
	REAL(kind = MPREC), INTENT(IN) :: r
	INTEGER, INTENT(IN) :: t
	REAL(kind = MPREC) :: t_
	t_ = REAL(t, MPREC) / REAL(t_twist, MPREC)
	t_ = ((one_ - tanh(t_ - 5 * one_)) - 2 * (one_ - tanh(t_))) / 2
	! t_ = (t_) * exp(- t_)

	if ((t_ .GT. 10) .OR.&
		& (t_ .LT. 0)) then
		twist_profile = 0.0
	else
		twist_profile = (one_ - tanh((r - RR0 * LL0) / (dR0 * LL0))) * (t_)
	end if
END FUNCTION twist_profile

SUBROUTINE monopole_E(xc_, yc_, zc_, x_, y_, z_, ex_, ey_, ez_)
	IMPLICIT NONE
	REAL(kind = MPREC), INTENT(IN) 	:: xc_, yc_, zc_
	REAL(kind = MPREC), INTENT(IN) 	:: x_, y_, z_
	REAL(kind = MPREC), INTENT(OUT) :: ex_, ey_, ez_
	REAL(kind = MPREC) 							:: dx1_, dz1_, dy1_, rad_, er_

	dx1_ = x_ - xc_
	dy1_ = y_ - yc_
	dz1_ = z_ - zc_

	rad_ = SQRT(dx1_**2 + dy1_**2 + dz1_**2)
	er_ = twist_profile(rad_, lap - t0_twist)
	ex_ = er_ * dx1_ / (RR0 * LL0); ey_ = er_ * dy1_ / (RR0 * LL0); ez_ = er_ * dz1_ / (RR0 * LL0)
END SUBROUTINE

SUBROUTINE user_bc()
	IMPLICIT NONE
	REAL(kind = MPREC) 		:: sx, sy, sz
	INTEGER 	          :: i, j, k, kglob, jglob, iglob
	REAL(kind = MPREC) 		:: twist_xc, twist_yc, twist_zc, er_, dummy1_, dummy2_
	REAL(kind = MPREC) 		:: exn, eyn, ezn
	REAL(kind = MPREC) 		:: br1_, br2_
	LOGICAL             :: print_once
	print_once = .FALSE.

	twist_xc = mon1_center_x
	twist_yc = mon1_center_y
	twist_zc = 1

	DO k = 1, mz
		DO j = 1, my
			DO i = 1, mx
				iglob = i
				jglob = j + MODULO(rank, sizey) * (my - 5)
				kglob = k + (rank / sizey) * (mz - 5)

				IF (kglob .GT. 3) CYCLE
				IF (bc_Mz .NE. 2) CYCLE

				! set E_xy
				sx = REAL(iglob + 0.5, MPREC)
				sy = REAL(jglob, MPREC)
				sz = REAL(kglob, MPREC)
				call monopole_E(twist_xc, twist_yc, twist_zc, sx, sy, sz, er_, dummy1_, dummy2_)
				exn = er_ * E_twist * bb0
				f(i, j, k).ex = exn

				sx = REAL(iglob, MPREC)
				sy = REAL(jglob + 0.5, MPREC)
				sz = REAL(kglob, MPREC)
				call monopole_E(twist_xc, twist_yc, twist_zc, sx, sy, sz, dummy1_, er_, dummy2_)
				eyn = er_ * E_twist * bb0
				f(i, j, k).ey = eyn

				IF ((rank .EQ. 0) .AND. (.NOT. print_once)) THEN
					PRINT *, "setting conducting BC at", lap, i, j, k, exn, eyn, f(i, j, k).bx, f(i, j, k).by
					print_once = .TRUE.
				END IF
			END DO
		END DO
	END DO

END SUBROUTINE user_bc
