#include "defs.f90"

REAL FUNCTION twist_profile(r, t)
	IMPLICIT NONE
	REAL, INTENT(IN) :: r
	INTEGER, INTENT(IN) :: t
	REAL :: t_
	t_ = REAL(t) / REAL(t_twist)

	if (t_ .GT. 2 * t_twist) then
		twist_profile = 0.0
	else
		twist_profile = (1.0 - tanh((r - RR0 * LL0) / (dR0 * LL0))) * (t_) * exp(- t_)
	end if
END FUNCTION twist_profile

SUBROUTINE monopole_E(xc_, yc_, zc_, x_, y_, z_, ex_, ey_, ez_)
	IMPLICIT NONE
	REAL, INTENT(IN) 	:: xc_, yc_, zc_
	REAL, INTENT(IN) 	:: x_, y_, z_
	REAL, INTENT(OUT) :: ex_, ey_, ez_
	REAL 							:: dx1_, dz1_, dy1_, rad_, er_

	dx1_ = x_ - xc_
	dy1_ = y_ - yc_
	dz1_ = z_ - zc_

	rad_ = SQRT(dx1_**2 + dy1_**2 + dz1_**2)
	er_ = twist_profile(rad_, lap)
	ex_ = er_ * dx1_ / (RR0 * LL0); ey_ = er_ * dy1_ / (RR0 * LL0); ez_ = er_ * dz1_ / (RR0 * LL0)
END SUBROUTINE

SUBROUTINE user_bc()
	IMPLICIT NONE
	REAL 			:: sx, sy, sz
	INTEGER 	:: i, j, k, kglob, jglob, iglob
	REAL 			:: twist_xc, twist_yc, twist_zc, er_, dummy1_, dummy2_
	REAL 			:: exn, eyn, ezn

	twist_xc = ie0; twist_yc = je0; twist_zc = 1

	DO k = 1, mz
		DO j = 1, my
			DO i = 1, mx
				iglob = i
				kglob = k + (rank / sizey) * (mz - 5)
				jglob = j + MODULO(rank, sizey) * (my - 5)

				if (kglob .gt. 3) cycle

				! set E_xy
				sx = iglob + 0.5
				sy = jglob
				sz = kglob
				call monopole_E(twist_xc, twist_yc, twist_zc, sx, sy, sz, er_, dummy1_, dummy2_)
				exn = er_ * E_twist
				exv(i, j, k) = exn; exp1(i, j, k) = 0

				sx = iglob
				sy = jglob + 0.5
				sz = kglob
				call monopole_E(twist_xc, twist_yc, twist_zc, sx, sy, sz, dummy1_, er_, dummy2_)
				eyn = er_ * E_twist
				eyv(i, j, k) = eyn; eyp(i, j, k) = 0

				! set B_z
				bz0(i, j, k) = 1; bzv(i, j, k) = 0; bzp(i, j, k) = 0
			END DO
		END DO
	END DO

END SUBROUTINE user_bc
