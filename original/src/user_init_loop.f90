#include "defs.f90"

SUBROUTINE get_monopole(mon_center_x, mon_center_y, mon_center_z,&
											& mon_sign,&
											& sx, sy, sz, obx, oby, obz)
	IMPLICIT NONE

	REAL(kind = MPREC), INTENT(IN) 		:: mon_center_x, mon_center_y, mon_center_z
	REAL(kind = MPREC), INTENT(IN) 		:: sx, sz, sy
	REAL(kind = MPREC), INTENT(OUT) 	:: obx, oby, obz
	INTEGER, INTENT(IN) :: mon_sign
	REAL(kind = MPREC) 								:: dx1_, dz1_, dy1_, rad_, br_

	dx1_ = sx - mon_center_x
	dy1_ = sy - mon_center_y
	dz1_ = sz - mon_center_z

	rad_ = SQRT(dx1_**2 + dy1_**2 + dz1_**2)
	br_ = REAL(mon_sign) / rad_**2
	obx = br_ * dx1_ / rad_; oby = br_ * dy1_ / rad_; obz = br_ * dz1_ / rad_
END SUBROUTINE get_monopole

SUBROUTINE user_init_fields()
	IMPLICIT NONE
	REAL(kind = MPREC) 			:: sx, sz, sy
	INTEGER 							:: i, k, j, iglob, jglob, kglob
	REAL(kind = MPREC) 			:: br1_, br2_, dummy1_, dummy2_

	mon1_center_x = shiftX * (mx0 - 5) + 2
	mon2_center_x = mon1_center_x
	mon1_center_y = je0 - aa0 * LL0; 	mon2_center_y = je0 + aa0 * LL0
	mon1_center_z = -hh0 * LL0; 			mon2_center_z = -hh0 * LL0

	DO i = 1, mx
		DO j = 1, my
			DO k = 1, mz
				iglob = i
				kglob = k + (rank / sizey) * (mz - 5)
				jglob = j + MODULO(rank, sizey) * (my - 5)

				sx = iglob
				sy = jglob + 0.5
				sz = kglob + 0.5
				CALL get_monopole(mon1_center_x, mon1_center_y, mon1_center_z, +1,&
													& sx, sy, sz, br1_, dummy1_, dummy2_)
				CALL get_monopole(mon2_center_x, mon2_center_y, mon2_center_z, -1,&
													& sx, sy, sz, br2_, dummy1_, dummy2_)
				f0(i, j, k).bx0 = (br1_ + br2_) * bb0

				sx = iglob + 0.5
				sy = jglob
				sz = kglob + 0.5
				CALL get_monopole(mon1_center_x, mon1_center_y, mon1_center_z, +1,&
												& sx, sy, sz, dummy1_, br1_, dummy2_)
				CALL get_monopole(mon2_center_x, mon2_center_y, mon2_center_z, -1,&
												& sx, sy, sz, dummy1_, br2_, dummy2_)
				f0(i, j, k).by0 = (br1_ + br2_) * bb0

				sx = iglob + 0.5
				sy = jglob + 0.5
				sz = kglob
				CALL get_monopole(mon1_center_x, mon1_center_y, mon1_center_z, +1,&
												& sx, sy, sz, dummy1_, dummy2_, br1_)
				CALL get_monopole(mon2_center_x, mon2_center_y, mon2_center_z, -1,&
												& sx, sy, sz, dummy1_, dummy2_, br2_)
				f0(i, j, k).bz0 = (br1_ + br2_) * bb0
			ENDDO
		ENDDO
	ENDDO

END SUBROUTINE user_init_fields
