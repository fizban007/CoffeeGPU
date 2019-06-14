#include "defs.f90"

SUBROUTINE user_init_fields()
	IMPLICIT NONE
	INTEGER :: i, k, j, iglob, jglob, kglob
	DO i = 1, mx
		DO j = 1, my
			DO k = 1, mz
				iglob = i
				jglob = j + MODULO(rank, sizey) * (my - 5)
				kglob = k + (rank / sizey) * (mz - 5)

				f0(i, j, k).bz0 = bb0 * sin((jglob - 3 + 0.5) * 3.0 * 2.0 * M_PI / (my0 - 5))
			ENDDO
		ENDDO
	ENDDO
END SUBROUTINE user_init_fields
