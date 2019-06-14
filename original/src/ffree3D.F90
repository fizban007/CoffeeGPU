#include "defs.f90"

MODULE EM_solver
	IMPLICIT NONE
	#ifdef MPI
	INCLUDE "mpif.h"
	#endif
	INTEGER mpi_read
	INTEGER mclock
	EXTERNAL mclock
	INTEGER rank, size0, sizey, sizez, ierr, statsize
	INTEGER clean_ep, normalize_e, check_e, nstart, nend, calc_curr, istep1, curr_form, subsamp

	INTEGER mx, my, mz, mx0, my0, mz0, n, i, j, k, ie0, je0, ke0
	INTEGER ieloc, jeloc, keloc, i0, i1
	INTEGER fnum

	REAL(kind = MPREC) delta0, c, dt, tch
	INTEGER substep
	REAL(kind = MPREC) radius, eta , eta1
	REAL(kind = MPREC) bcmult !to correctly copy periodic BC in presence of f0
	INTEGER lap, interval, hrsizex, hrsizey, hrsizez, writehr, writerest
	LOGICAL debug

	TYPE field
		REAL(kind = MPREC) :: ex, ey, ez, bx, by, bz
	END TYPE field

	TYPE supplement_field
		REAL(kind = MPREC) :: bx0, by0, bz0
	END TYPE supplement_field

	TYPE current_field
		REAL(kind = MPREC) :: jx, jy, jz
	END TYPE current_field

	TYPE(field), ALLOCATABLE :: f(:,:,:), fn(:,:,:), df(:,:,:)
	TYPE(current_field), ALLOCATABLE :: ff_curr(:,:,:)

	TYPE(supplement_field), ALLOCATABLE :: f0(:,:,:)
	REAL(kind = MPREC), ALLOCATABLE :: rho(:,:,:)

	REAL(kind = MPREC) intex, intey, intez, intbx, intby, intbz, intrho, jx, jy, jz, arr, emag, bmag
	REAL(kind = MPREC) epar, jxperp, jyperp, jzperp, jxpar, jypar, jzpar

	! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
	! problem specific:
	! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
	REAL(kind = MPREC) :: bb0, LL0, aa0, hh0
	REAL(kind = MPREC) :: shiftX, RR0, dR0, E_twist
	INTEGER :: t0_twist, t_twist, dr_layer
	INTEGER :: bc_Px, bc_Mx, bc_Py, bc_My, bc_Pz, bc_Mz
	LOGICAL :: absorb
	REAL(kind = MPREC) :: mon1_center_x, mon1_center_y, mon1_center_z
	REAL(kind = MPREC) :: mon2_center_x, mon2_center_y, mon2_center_z

CONTAINS
	SUBROUTINE alloc_EMsolver()
		ALLOCATE(f(mx, my, mz), fn(mx, my, mz), f0(mx, my, mz), df(mx, my, mz), rho(mx, my, mz), ff_curr(mx, my, mz))
	END SUBROUTINE alloc_EMsolver

	SUBROUTINE read_input()
		OPEN(unit = 10, file="input",form='formatted')
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) mx0, my0, mz0
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) sizey, c, dt, istep1
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) nstart, nend, tch, interval
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) bb0
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) shiftX, LL0, aa0, hh0, RR0, dR0
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) t0_twist, t_twist, E_twist
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) bc_Px, bc_Mx, bc_Py, bc_My, bc_Pz, bc_Mz
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) absorb, dr_layer
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) normalize_e, check_e, clean_ep, calc_curr
		READ(unit = 10, fmt='(/)', err = 998, END = 999)
		READ(unit = 10, fmt=*,     err = 998, END = 999) debug

		IF (RANK .EQ. 0) THEN
			PRINT *, "PRINTING READ_INPUT >>>"
			PRINT *, mx0, my0, mz0
			PRINT *, sizey, c, dt, istep1
			PRINT *, nstart, nend, tch, interval
			PRINT *, bb0
			PRINT *, shiftX, LL0, aa0, hh0, RR0, dR0
			PRINT *, t0_twist, t_twist, E_twist
			PRINT *, bc_Px, bc_Mx, bc_Py, bc_My, bc_Pz, bc_Mz
			PRINT *, absorb, dr_layer
			PRINT *, normalize_e, check_e, clean_ep, calc_curr
			PRINT *, debug
			PRINT *, "using precision:", MPREC
			PRINT *, "<<<"
		END IF

		GOTO 1997
		998 PRINT *,"error reading the input file"
		999 PRINT *,"reached the end of the input file"
		1997 CONTINUE

		CLOSE(10)

	END SUBROUTINE read_input

	SUBROUTINE init_EMsolver()
		IMPLICIT NONE

		INTEGER n, k, j, i

		#ifdef HDF5
		#ifndef MPI
		PRINT *, "ERROR, HDF has to be turned on with MPI. MPI not defined"
		STOP
		#endif
		#endif

		#ifdef MPI
		CALL MPI_Init(ierr)
		CALL MPI_Comm_rank(MPI_Comm_world, rank, ierr)
		CALL MPI_Comm_size(MPI_Comm_world, size0, ierr)
		statsize = MPI_STATUS_SIZE
		#else
		rank = 0
		size0 = 1
		#endif

		CALL read_input()

		delta0 = 1e-7

		#ifndef MPI
		sizey = 1
		#endif
		sizez = size0 / sizey

		PRINT *, "rank", rank, ":", " sizez=", sizez, "size0=", size0, "sizey=", sizey
		IF (size0 .LT. sizey) THEN
			PRINT *, rank, ":", "not enough processors for sizey"
			STOP
		ENDIF

		IF (sizez * sizey .NE. size0) THEN
			PRINT *, rank, ":", "Error: sizex * sizey ne size0"
			STOP
		ENDIF
		IF (MODULO(my0 - 5, sizey) .NE. 0) THEN
			PRINT *, rank, ":", "my indivisible by number of processors in the y direction", sizey, my
			STOP
		ENDIF
		IF (MODULO(mz0 - 5, sizez) .NE. 0) THEN
			PRINT *, rank, ":", "mz indivisible by number of processors in the z direction", sizez, mz
			STOP
		ENDIF

		mx = mx0
		my = (my0 - 5) / sizey + 5
		mz = (mz0 - 5) / sizez + 5

		ie0 = (mx0 - 5) / 2 + 2
		je0 = (my0 - 5) / 2 + 2
		ke0 = (mz0 - 5) / 2 + 2

		CALL alloc_EMsolver()
		CALL initialize_fields()
	END SUBROUTINE init_EMsolver

	INCLUDE "user_init_loop.f90"
	INCLUDE "user_bc_loop.f90"
	! INCLUDE "user_init_uniform.f90"

	SUBROUTINE initialize_fields()
		INTEGER  kglob, jglob
		f0(:,:,:).bx0 = 0
		f0(:,:,:).by0 = 0
		f0(:,:,:).bz0 = 0
		f(:,:,:).ex = 0
		f(:,:,:).ey = 0
		f(:,:,:).ez = 0

		CALL user_init_fields()

		f(:,:,:).bx = f0(:,:,:).bx0
		f(:,:,:).by = f0(:,:,:).by0
		f(:,:,:).bz = f0(:,:,:).bz0
	END SUBROUTINE initialize_fields

	INCLUDE "algorithm.f90"
	INCLUDE "boundary_conditions.f90"
	INCLUDE "auxiliary.f90"
	INCLUDE "output.f90"

END MODULE EM_solver

PROGRAM ffree_dipole
	USE EM_solver
	IMPLICIT NONE
	CHARACTER frestartfld * 30
	DOUBLE PRECISION tstart, tfin

	fnum = 0

	CALL init_EMsolver()

	DO lap = nstart, nend
		tstart = mpi_wtime()
		IF (rank .EQ. 0) PRINT *,"lap=",lap
		CALL EM_step_rk3()

		IF (MODULO(lap, interval).EQ.0) THEN
			#ifdef HDF5
			CALL write_hdf()
			#endif
		ENDIF

		tfin = mpi_wtime()
		IF (rank .EQ. 0) PRINT *, "Step time=", tfin - tstart, " seconds"
	ENDDO

END PROGRAM ffree_dipole
