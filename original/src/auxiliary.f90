#include "defs.f90"

SUBROUTINE copylayrx_f(arr, mx, my, mz, lt, ls, nt, ns)
	IMPLICIT NONE
	INTEGER mx, my, mz, lt, ls, nt, ns, k, j
	TYPE(field), INTENT(inout) :: arr(mx, my, mz)
	!our decomposition is in y and z, so in x this is local operation
	! on each processor

	DO k = 1, mz
		DO j = 1, my
			arr(lt, j, k).ex = arr(ls, j, k).ex
			arr(lt, j, k).ey = arr(ls, j, k).ey
			arr(lt, j, k).ez = arr(ls, j, k).ez

			arr(lt, j, k).bx = arr(ls, j, k).bx - f0(ls, j, k).bx0 * bcmult + f0(lt, j, k).bx0 * bcmult
			arr(lt, j, k).by = arr(ls, j, k).by - f0(ls, j, k).by0 * bcmult + f0(lt, j, k).by0 * bcmult
			arr(lt, j, k).bz = arr(ls, j, k).bz - f0(ls, j, k).bz0 * bcmult + f0(lt, j, k).bz0 * bcmult
		ENDDO
	ENDDO

	DO k = 1, mz
		DO j = 1, my
			arr(nt, j, k).ex = arr(ns, j, k).ex
			arr(nt, j, k).ey = arr(ns, j, k).ey
			arr(nt, j, k).ez = arr(ns, j, k).ez

			arr(nt, j, k).bx = arr(ns, j, k).bx - f0(ns, j, k).bx0 * bcmult + f0(nt, j, k).bx0 * bcmult
			arr(nt, j, k).by = arr(ns, j, k).by - f0(ns, j, k).by0 * bcmult + f0(nt, j, k).by0 * bcmult
			arr(nt, j, k).bz = arr(ns, j, k).bz - f0(ns, j, k).bz0 * bcmult + f0(nt, j, k).bz0 * bcmult
		ENDDO
	ENDDO
END SUBROUTINE copylayrx_f

SUBROUTINE copylayry_f(arr, mx, my, mz, lt, ls, nt, ns, rank, sizey, sizez)
	IMPLICIT NONE
	#ifdef MPI
	INCLUDE "mpif.h"
	INTEGER status(MPI_STATUS_SIZE)
	#endif
	INTEGER mx, my, mz, lt, ls, nt, ns, i, k
	INTEGER rgtrank, lftrank, iperiodic, comm, count, rgttag, lfttag &
	&    ,ierr, rank, sizey, sizez, mpi_read
	TYPE(field) arr(mx, my, mz), sendbuf(mx, mz), recvbuf(mx, mz)

	#ifdef MPI
	IF (MPREC .EQ. 4) THEN
		mpi_read = MPI_REAL
	ELSE
		mpi_read = MPI_DOUBLE_PRECISION
	END IF
	#endif
	!MPI_REAL or MPI_DOUBLE_PRECISION
	! MPI index for double - precision reals, use 13 for single
	! this variable used so that MPI code can be changed from
	! single to double precision by changing this variable.


	rgttag = 100
	lfttag = 200

	#ifdef MPI
	comm = MPI_Comm_world !shorthand notation
	#endif

	!processors are organized in a circle in y
	!find out who to send the data, and where from to receive it
	rgtrank = (rank / sizey) * sizey + MODULO(rank + 1, sizey)
	lftrank = (rank / sizey) * sizey + MODULO(rank - 1, sizey)
	count = mx * mz * 6

	!simulataneously send right and receive from left
	DO k = 1, mz
		DO i = 1, mx
			sendbuf(i, k) = arr(i, ls, k)

			sendbuf(i, k).bx = sendbuf(i, k).bx - f0(i, ls, k).bx0 * bcmult
			sendbuf(i, k).by = sendbuf(i, k).by - f0(i, ls, k).by0 * bcmult
			sendbuf(i, k).bz = sendbuf(i, k).bz - f0(i, ls, k).bz0 * bcmult

		ENDDO
	ENDDO


	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, rgtrank, rgttag, &
	&                  recvbuf, count, mpi_read, lftrank, rgttag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO k = 1, mz
		DO i = 1, mx
			arr(i, lt, k) = recvbuf(i, k)

			arr(i, lt, k).bx = arr(i, lt, k).bx + f0(i, lt, k).bx0 * bcmult
			arr(i, lt, k).by = arr(i, lt, k).by + f0(i, lt, k).by0 * bcmult
			arr(i, lt, k).bz = arr(i, lt, k).bz + f0(i, lt, k).bz0 * bcmult

		ENDDO
	ENDDO

	!send left and receive from right

	DO k = 1, mz
		DO i = 1, mx
			sendbuf(i, k) = arr(i, ns, k)

			sendbuf(i, k).bx = sendbuf(i, k).bx - f0(i, ns, k).bx0 * bcmult
			sendbuf(i, k).by = sendbuf(i, k).by - f0(i, ns, k).by0 * bcmult
			sendbuf(i, k).bz = sendbuf(i, k).bz - f0(i, ns, k).bz0 * bcmult

		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, lftrank, lfttag, &
	&                  recvbuf, count, mpi_read, rgtrank, lfttag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO k = 1, mz
		DO i = 1, mx
			arr(i, nt, k) = recvbuf(i, k)

			arr(i, nt, k).bx = arr(i, nt, k).bx + f0(i, nt, k).bx0 * bcmult
			arr(i, nt, k).by = arr(i, nt, k).by + f0(i, nt, k).by0 * bcmult
			arr(i, nt, k).bz = arr(i, nt, k).bz + f0(i, nt, k).bz0 * bcmult

		ENDDO

	ENDDO
END SUBROUTINE copylayry_f

SUBROUTINE copylayrz_f(arr, mx, my, mz, lt, ls, nt, ns, rank, sizey, sizez)
	IMPLICIT NONE
	#ifdef MPI
	INCLUDE "mpif.h"
	INTEGER status(MPI_STATUS_SIZE)
	#endif
	INTEGER mx, my, mz, lt, ls, nt, ns, i, j
	INTEGER dwnrank, uprank, comm, count, uptag, dwntag &
	&     ,ierr, rank, sizey, mpi_read, sizez

	TYPE(field) :: arr(mx, my, mz), sendbuf(mx, my), recvbuf(mx, my)

	#ifdef MPI
	IF (MPREC .EQ. 4) THEN
		mpi_read = MPI_REAL
	ELSE
		mpi_read = MPI_DOUBLE_PRECISION
	END IF
	#endif

	!or MPI_DOUBLE_PRECISION
	!MPI index for double - precision reals, use 13 for single
	! this variable used so that MPI code can be changed from
	! single to double precision by changing this variable.


	uptag = 100
	dwntag = 200
	#ifdef MPI
	comm = MPI_Comm_world !shorthand notation
	#endif
	!processors are organized in a circle in y
	!find out who to send the data, and where from to receive it

	uprank = MODULO((rank / sizey + 1), sizez) * sizey + MODULO(rank, sizey)
	dwnrank = MODULO((rank / sizey - 1), sizez) * sizey + MODULO(rank, sizey)
	count = mx * my * 6

	!simulataneously send up and receive from below
	DO j = 1, my
		DO i = 1, mx
			sendbuf(i, j) = arr(i, j, ls)

			sendbuf(i, j).bx = sendbuf(i, j).bx - f0(i, j, ls).bx0 * bcmult
			sendbuf(i, j).by = sendbuf(i, j).by - f0(i, j, ls).by0 * bcmult
			sendbuf(i, j).bz = sendbuf(i, j).bz - f0(i, j, ls).bz0 * bcmult

		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, uprank, uptag, &
	&                  recvbuf, count, mpi_read, dwnrank, uptag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO j = 1, my
		DO i = 1, mx
			arr(i, j, lt) = recvbuf(i, j)

			arr(i, j, lt).bx = arr(i, j, lt).bx + f0(i, j, lt).bx0 * bcmult
			arr(i, j, lt).by = arr(i, j, lt).by + f0(i, j, lt).by0 * bcmult
			arr(i, j, lt).bz = arr(i, j, lt).bz + f0(i, j, lt).bz0 * bcmult

		ENDDO
	ENDDO

	DO j = 1, my
		DO i = 1, mx
			sendbuf(i, j) = arr(i, j, ns)

			sendbuf(i, j).bx = sendbuf(i, j).bx - f0(i, j, ns).bx0 * bcmult
			sendbuf(i, j).by = sendbuf(i, j).by - f0(i, j, ns).by0 * bcmult
			sendbuf(i, j).bz = sendbuf(i, j).bz - f0(i, j, ns).bz0 * bcmult

		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, dwnrank, dwntag, &
	&                  recvbuf, count, mpi_read, uprank, dwntag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO j = 1, my
		DO i = 1, mx
			arr(i, j, nt) = recvbuf(i, j)

			arr(i, j, nt).bx = arr(i, j, nt).bx + f0(i, j, nt).bx0 * bcmult
			arr(i, j, nt).by = arr(i, j, nt).by + f0(i, j, nt).by0 * bcmult
			arr(i, j, nt).bz = arr(i, j, nt).bz + f0(i, j, nt).bz0 * bcmult

		ENDDO
	ENDDO
END SUBROUTINE copylayrz_f

SUBROUTINE find_rho()
	IMPLICIT NONE
	INTEGER i, j, k
	DO k = 2, mz
		DO j = 2, my
			DO i = i0, i1 !2, mx
				rho(i, j, k) = (f(i, j, k).ex - f(i - 1, j, k).ex) +&
				& (f(i, j, k).ey - f(i, j - 1, k).ey) +&
				& (f(i, j, k).ez - f(i, j, k - 1).ez)
			ENDDO
		ENDDO
	ENDDO
END SUBROUTINE find_rho


SUBROUTINE compute_ffree_current()
	IMPLICIT NONE
	INTEGER :: i, j, k
	REAL(kind = MPREC) :: intrho, intex, intey, intez, intbx, intby, intbz, temp1_, temp2_
	REAL(kind = MPREC) :: bmag, emag, int_curlBx, int_curlBy, int_curlBz
	type(current_field), allocatable :: curlE(:,:,:), curlB(:,:,:)
	allocate(curlE(mx, my, mz), curlB(mx, my, mz))
	curlE(:,:,:).jx = 0
	curlE(:,:,:).jy = 0
	curlE(:,:,:).jz = 0
	curlB(:,:,:).jx = 0
	curlB(:,:,:).jy = 0
	curlB(:,:,:).jz = 0
	ff_curr(:,:,:).jx = 0
	ff_curr(:,:,:).jy = 0
	ff_curr(:,:,:).jz = 0

	CALL find_rho()

	! ff_curr
	DO  k = 2, mz - 1
		DO  j = 2, my - 1
			DO  i = 2, mx - 1
				curlE(i, j, k).jx = (f(i, j, k).ez - f(i, j - 1, k).ez) - (f(i, j, k).ey - f(i, j, k - 1).ey)
				curlE(i, j, k).jy = -(f(i, j, k).ez - f(i - 1, j, k).ez) + (f(i, j, k).ex - f(i, j, k - 1).ex)
				curlE(i, j, k).jz = (f(i, j, k).ey - f(i - 1, j, k).ey) - (f(i, j, k).ex - f(i, j - 1, k).ex)

				curlB(i, j, k).jx = (f(i, j, k).bz - f(i, j - 1, k).bz) - (f(i, j, k).by - f(i, j, k - 1).by)
				curlB(i, j, k).jy = -(f(i, j, k).bz - f(i - 1, j, k).bz) + (f(i, j, k).bx - f(i, j, k - 1).bx)
				curlB(i, j, k).jz = (f(i, j, k).by - f(i - 1, j, k).by) - (f(i, j, k).bx - f(i, j - 1, k).bx)

				!--------X current-------------------------------------
				!-------interpolate rho_x---------------------------
				intrho = 0.5 * (rho(i + 1, j, k) + rho(i, j, k))

				!-------interpolate E x-----------------------------
				intex = f(i, j, k).ex
				intey = 0.25 * (f(i, j, k).ey + f(i + 1, j, k).ey + f(i, j - 1, k).ey &
										& + f(i + 1, j - 1, k).ey)
				intez = 0.25 * (f(i, j, k).ez + f(i + 1, j, k).ez + f(i, j, k - 1).ez &
										& + f(i + 1, j, k - 1).ez)
				!---------------------------------------------------
				!-------interpolate B x-----------------------------
				intbx = 0.125 * (f(i, j, k).bx + f(i, j - 1, k).bx + f(i + 1, j - 1, k).bx + f(i + 1, j, k).bx + &
											 & f(i, j, k - 1).bx + f(i, j - 1, k - 1).bx + f(i + 1, j - 1, k - 1).bx + f(i + 1, j, k - 1).bx)
				intby = 0.5 * (f(i, j, k).by + f(i, j, k - 1).by)
				intbz = 0.5 * (f(i, j, k).bz + f(i, j - 1, k).bz)
				!---------------------------------------------------

				bmag = intbx**2 + intby**2 + intbz**2
				emag = intex**2 + intey**2 + intez**2
				IF (emag .GT. bmag) THEN
					arr = SQRT(bmag / emag)
					intey = intey * arr
					intez = intez * arr
				ENDIF

				ff_curr(i, j, k).jx = (intrho * (intey * intbz - intby * intez)) / (bmag + delta0)
				!--------Y current--------------------------------------
				!-------interpolate rho_y---------------------------
				intrho = 0.5 * (rho(i, j + 1, k) + rho(i, j, k))

				!-------interpolate E y-----------------------------

				intex = 0.25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j + 1, k).ex &
										& + f(i - 1, j + 1, k).ex)
				intey = f(i, j, k).ey
				intez = 0.25 * (f(i, j, k).ez + f(i, j + 1, k).ez + f(i, j, k - 1).ez &
										& + f(i, j + 1, k - 1).ez)

				!---------------------------------------------------
				!-------interpolate B y-----------------------------
				intbx = 0.5 * (f(i, j, k).bx + f(i, j, k - 1).bx)
				intby = 0.125 * (f(i, j, k).by + f(i - 1, j, k).by + f(i - 1, j + 1, k).by + f(i, j + 1, k).by + &
											 & f(i, j, k - 1).by + f(i - 1, j, k - 1).by + f(i - 1, j + 1, k - 1).by + f(i, j + 1, k - 1).by)
				intbz = 0.5 * (f(i, j, k).bz + f(i - 1, j, k).bz)
				!---------------------------------------------------

				bmag = intbx**2 + intby**2 + intbz**2
				emag = intex**2 + intey**2 + intez**2
				IF (emag .GT. bmag) THEN
					arr = SQRT(bmag / emag)
					intex = intex * arr
					intez = intez * arr
				ENDIF

				ff_curr(i, j, k).jy = (intrho * (intez * intbx - intex * intbz)) / (bmag + delta0)


				!--------Z current--------------------------------------
				!-------interpolate rho_z---------------------------
				intrho = 0.5 * (rho(i, j, k) + rho(i, j, k + 1))

				!-------interpolate E z-----------------------------
				intex = 0.25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j, k + 1).ex &
										& + f(i - 1, j, k + 1).ex)
				intey = 0.25 * (f(i, j, k).ey + f(i, j - 1, k).ey + f(i, j, k + 1).ey &
										& + f(i, j - 1, k + 1).ey)
				intez = f(i, j, k).ez
				!---------------------------------------------------
				!-------interpolate B z-----------------------------
				intbx = 0.5 * (f(i, j, k).bx + f(i, j - 1, k).bx)
				intby = 0.5 * (f(i, j, k).by + f(i - 1, j, k).by)
				intbz = 0.125 * (f(i, j, k).bz + f(i - 1, j, k).bz + f(i - 1, j - 1, k).bz + f(i, j - 1, k).bz + &
											 & f(i, j, k + 1).bz + f(i - 1, j, k + 1).bz + f(i - 1, j - 1, k + 1).bz + f(i, j - 1, k + 1).bz)
				!---------------------------------------------------

				bmag = intbx**2 + intby**2 + intbz**2
				emag = intex**2 + intey**2 + intez**2
				IF (emag .GT. bmag) THEN
					arr = SQRT(bmag / emag)
					intex = intex * arr
					intey = intey * arr
				ENDIF
				ff_curr(i, j, k).jz = (intrho * (intex * intby - intbx * intey)) / (bmag + delta0)
			ENDDO
		ENDDO
	ENDDO

	CALL copylayrx_ffcurr(ff_curr, mx, my, mz, 1, mx - 4, mx - 1, 4)
	CALL copylayry_ffcurr(ff_curr, mx, my, mz, 1, my - 4, my - 1, 4, rank, sizey, sizez)
	CALL copylayrz_ffcurr(ff_curr, mx, my, mz, 1, mz - 4, mz - 1, 4, rank, sizey, sizez)

	CALL copylayrx_ffcurr(curlE, mx, my, mz, 1, mx - 4, mx - 1, 4)
	CALL copylayry_ffcurr(curlE, mx, my, mz, 1, my - 4, my - 1, 4, rank, sizey, sizez)
	CALL copylayrz_ffcurr(curlE, mx, my, mz, 1, mz - 4, mz - 1, 4, rank, sizey, sizez)

	CALL copylayrx_ffcurr(curlB, mx, my, mz, 1, mx - 4, mx - 1, 4)
	CALL copylayry_ffcurr(curlB, mx, my, mz, 1, my - 4, my - 1, 4, rank, sizey, sizez)
	CALL copylayrz_ffcurr(curlB, mx, my, mz, 1, mz - 4, mz - 1, 4, rank, sizey, sizez)

	! ff_curr.j = rho * E x B / B^2 (computed on edges)

	DO  k = 2, mz - 1
		DO  j = 2, my - 1
			DO  i = 2, mx - 1
				! interpolating to nodes
				ff_curr(i, j, k).jx = 0.5 * (ff_curr(i, j, k).jx + ff_curr(i + 1, j, k).jx)
				ff_curr(i, j, k).jy = 0.5 * (ff_curr(i, j, k).jy + ff_curr(i, j + 1, k).jy)
				ff_curr(i, j, k).jz = 0.5 * (ff_curr(i, j, k).jz + ff_curr(i, j, k + 1).jz)

				! interpolating onto the faces
				intex = 0.125 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j, k + 1).ex + f(i - 1, j, k + 1).ex + &
											 & f(i, j + 1, k).ex + f(i - 1, j + 1, k).ex + f(i, j + 1, k + 1).ex + f(i - 1, j + 1, k + 1).ex)
				intey = 0.125 * (f(i, j, k).ey + f(i, j - 1, k).ey + f(i, j, k + 1).ey + f(i, j - 1, k + 1).ey + &
											 & f(i + 1, j, k).ey + f(i + 1, j - 1, k).ey + f(i + 1, j, k + 1).ey + f(i + 1, j - 1, k + 1).ey)
				intez = 0.125 * (f(i, j, k).ez + f(i, j - 1, k).ez + f(i, j, k + 1).ez + f(i, j - 1, k + 1).ez + &
											 & f(i + 1, j, k).ez + f(i + 1, j - 1, k).ez + f(i + 1, j, k + 1).ez + f(i + 1, j - 1, k + 1).ez)
				! E . curl E:
				temp1_ = intex * curlE(i, j, k).jx + intey * curlE(i, j, k).jy + intez * curlE(i, j, k).jz

				int_curlBx = 0.125 * (curlB(i, j, k).jx + curlB(i - 1, j, k).jx + curlB(i, j, k + 1).jx + curlB(i - 1, j, k + 1).jx + &
														& curlB(i, j + 1, k).jx + curlB(i - 1, j + 1, k).jx + curlB(i, j + 1, k + 1).jx + curlB(i - 1, j + 1, k + 1).jx)
				int_curlBy = 0.125 * (curlB(i, j, k).jy + curlB(i, j - 1, k).jy + curlB(i, j, k + 1).jy + curlB(i, j - 1, k + 1).jy + &
														& curlB(i + 1, j, k).jy + curlB(i + 1, j - 1, k).jy + curlB(i + 1, j, k + 1).jy + curlB(i + 1, j - 1, k + 1).jy)
				int_curlBz = 0.125 * (curlB(i, j, k).jz + curlB(i, j - 1, k).jz + curlB(i, j, k + 1).jz + curlB(i, j - 1, k + 1).jz + &
														& curlB(i + 1, j, k).jz + curlB(i + 1, j - 1, k).jz + curlB(i + 1, j, k + 1).jz + curlB(i + 1, j - 1, k + 1).jz)
				! B . curl B:
				temp2_ = f(i, j, k).bx * int_curlBx + f(i, j, k).by * int_curlBy + f(i, j, k).bz * int_curlBz

				! now curlE will contain the new current (stored on faces)
				bmag = f(i, j, k).bx**2 + f(i, j, k).by**2 + f(i, j, k).bz**2
				curlE(i, j, k).jx = (temp2_ - temp1_) * f(i, j, k).bx / (bmag + delta0)
				curlE(i, j, k).jy = (temp2_ - temp1_) * f(i, j, k).by / (bmag + delta0)
				curlE(i, j, k).jz = (temp2_ - temp1_) * f(i, j, k).bz / (bmag + delta0)
			ENDDO
		ENDDO
	ENDDO

	CALL copylayrx_ffcurr(curlE, mx, my, mz, 1, mx - 4, mx - 1, 4)
	CALL copylayry_ffcurr(curlE, mx, my, mz, 1, my - 4, my - 1, 4, rank, sizey, sizez)
	CALL copylayrz_ffcurr(curlE, mx, my, mz, 1, mz - 4, mz - 1, 4, rank, sizey, sizez)

	DO  k = 2, mz - 1
		DO  j = 2, my - 1
			DO  i = 2, mx - 1
				! interpolating from faces to nodes
				curlE(i, j, k).jx = 0.25 * (curlE(i, j, k).jx + curlE(i, j - 1, k).jx + &
																	& curlE(i, j, k - 1).jx + curlE(i, j - 1, k - 1).jx)

				curlE(i, j, k).jy = 0.25 * (curlE(i, j, k).jy + curlE(i - 1, j, k).jy + &
																	& curlE(i, j, k - 1).jy + curlE(i - 1, j, k - 1).jy)
				curlE(i, j, k).jz = 0.25 * (curlE(i, j, k).jz + curlE(i - 1, j, k).jz + &
																	& curlE(i, j - 1, k).jz + curlE(i - 1, j - 1, k).jz)
				ff_curr(i, j, k).jx = ff_curr(i, j, k).jx + curlE(i, j, k).jx
				ff_curr(i, j, k).jy = ff_curr(i, j, k).jy + curlE(i, j, k).jy
				ff_curr(i, j, k).jz = ff_curr(i, j, k).jz + curlE(i, j, k).jz
			ENDDO
		ENDDO
	ENDDO

	deallocate(curlE, curlB)

END SUBROUTINE compute_ffree_current

SUBROUTINE copylayrx_ffcurr(temparr, mx, my, mz, lt, ls, nt, ns)
	IMPLICIT NONE
	TYPE(current_field), INTENT(INOUT) :: temparr(mx,my,mz)
	INTEGER :: mx, my, mz, lt, ls, nt, ns, k, j
	DO k = 1, mz
		DO j = 1, my
			temparr(lt, j, k).jx = temparr(ls, j, k).jx
			temparr(lt, j, k).jy = temparr(ls, j, k).jy
			temparr(lt, j, k).jz = temparr(ls, j, k).jz
		ENDDO
	ENDDO

	DO k = 1, mz
		DO j = 1, my
			temparr(nt, j, k).jx = temparr(ns, j, k).jx
			temparr(nt, j, k).jy = temparr(ns, j, k).jy
			temparr(nt, j, k).jz = temparr(ns, j, k).jz
		ENDDO
	ENDDO
END SUBROUTINE copylayrx_ffcurr

SUBROUTINE copylayry_ffcurr(temparr, mx, my, mz, lt, ls, nt, ns, rank, sizey, sizez)
	IMPLICIT NONE
	#ifdef MPI
	INCLUDE "mpif.h"
	INTEGER status(MPI_STATUS_SIZE)
	#endif
	TYPE(current_field), INTENT(INOUT) :: temparr(mx,my,mz)
	INTEGER :: mx, my, mz, lt, ls, nt, ns, i, k
	INTEGER :: rgtrank, lftrank, iperiodic, comm, count, rgttag, lfttag,&
					& ierr, rank, sizey, sizez, mpi_read
	TYPE(current_field) :: sendbuf(mx, mz), recvbuf(mx, mz)

	#ifdef MPI
	IF (MPREC .EQ. 4) THEN
		mpi_read = MPI_REAL
	ELSE
		mpi_read = MPI_DOUBLE_PRECISION
	END IF
	#endif

	rgttag = 100
	lfttag = 200

	#ifdef MPI
	comm = MPI_Comm_world !shorthand notation
	#endif

	!processors are organized in a circle in y
	!find out who to send the data, and where from to receive it
	rgtrank = (rank / sizey) * sizey + MODULO(rank + 1, sizey)
	lftrank = (rank / sizey) * sizey + MODULO(rank - 1, sizey)
	count = mx * mz * 3

	!simulataneously send right and receive from left
	DO k = 1, mz
		DO i = 1, mx
			sendbuf(i, k) = temparr(i, ls, k)
		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, rgtrank, rgttag, &
	&                  recvbuf, count, mpi_read, lftrank, rgttag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO k = 1, mz
		DO i = 1, mx
			temparr(i, lt, k) = recvbuf(i, k)
		ENDDO
	ENDDO

	!send left and receive from right

	DO k = 1, mz
		DO i = 1, mx
			sendbuf(i, k) = temparr(i, ns, k)
		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, lftrank, lfttag, &
	&                  recvbuf, count, mpi_read, rgtrank, lfttag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO k = 1, mz
		DO i = 1, mx
			temparr(i, nt, k) = recvbuf(i, k)
		ENDDO
	ENDDO
END SUBROUTINE copylayry_ffcurr

SUBROUTINE copylayrz_ffcurr(temparr, mx, my, mz, lt, ls, nt, ns, rank, sizey, sizez)
	IMPLICIT NONE
	#ifdef MPI
	INCLUDE "mpif.h"
	INTEGER status(MPI_STATUS_SIZE)
	#endif
	TYPE(current_field), INTENT(INOUT) :: temparr(mx,my,mz)
	INTEGER :: mx, my, mz, lt, ls, nt, ns, i, j
	INTEGER :: dwnrank, uprank, comm, count, uptag, dwntag, &
	& ierr, rank, sizey, mpi_read, sizez

	TYPE(current_field) :: sendbuf(mx, my), recvbuf(mx, my)

	#ifdef MPI
	IF (MPREC .EQ. 4) THEN
		mpi_read = MPI_REAL
	ELSE
		mpi_read = MPI_DOUBLE_PRECISION
	END IF
	#endif

	uptag = 100
	dwntag = 200
	#ifdef MPI
	comm = MPI_Comm_world !shorthand notation
	#endif
	!processors are organized in a circle in y
	!find out who to send the data, and where from to receive it

	uprank = MODULO((rank / sizey + 1), sizez) * sizey + MODULO(rank, sizey)
	dwnrank = MODULO((rank / sizey - 1), sizez) * sizey + MODULO(rank, sizey)
	count = mx * my * 3

	!simulataneously send up and receive from below
	DO j = 1, my
		DO i = 1, mx
			sendbuf(i, j) = temparr(i, j, ls)
		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, uprank, uptag, &
	&                  recvbuf, count, mpi_read, dwnrank, uptag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO j = 1, my
		DO i = 1, mx
			temparr(i, j, lt) = recvbuf(i, j)
		ENDDO
	ENDDO

	DO j = 1, my
		DO i = 1, mx
			sendbuf(i, j) = temparr(i, j, ns)
		ENDDO
	ENDDO

	#ifdef MPI
	CALL MPI_SendRecv(sendbuf, count, mpi_read, dwnrank, dwntag, &
	&                  recvbuf, count, mpi_read, uprank, dwntag, &
	&                  MPI_Comm_World, status, ierr)
	#else
	recvbuf = sendbuf
	#endif

	DO j = 1, my
		DO i = 1, mx
			temparr(i, j, nt) = recvbuf(i, j)
		ENDDO
	ENDDO
END SUBROUTINE copylayrz_ffcurr
