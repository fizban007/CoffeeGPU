#include "defs.f90"

#ifdef HDF5
SUBROUTINE write_hdf()
	USE hdf5
	INTEGER :: is, kglob, kloc, kstart, kfinish, nx, nz, fh, info
	INTEGER :: jstart, jfinish, ny, iv, istart, ifinish

	REAL * 4, ALLOCATABLE:: temporary1(:,:,:)

	INTEGER error, error_n  ! Error flags
	INTEGER nvars, midvars, stride
	INTEGER all_counts(size0) ! array to know the sizes of other proc array
	INTEGER all_ions(size0), all_lecs(size0)
	CHARACTER(len = 9) dsetname(20)
	CHARACTER(kind = 1, len = 20) fname
	INTEGER(HID_T) :: file_id ! File identifier
	INTEGER(HID_T) :: dset_id(20) ! Dataset identifier
	INTEGER(HID_T) :: filespace(20) ! Dataspace identifier in file
	INTEGER(HID_T) :: memspace ! Dataspace identifier in memory
	INTEGER(HID_T) :: plist_id ! Property list identifier
	INTEGER(HSIZE_T) :: dimsf(3)
	INTEGER(HSIZE_T) :: dimsfi(7)
	INTEGER skipflag, datarank
	INTEGER(HSIZE_T), DIMENSION(3) :: count
	INTEGER(HSSIZE_T), DIMENSION(3) :: offset
	INTEGER(HSIZE_T), DIMENSION(1) :: countpart
	INTEGER(HSSIZE_T), DIMENSION(1) :: offsetpart
	INTEGER mx1, my1, mz1
	REAL(kind = MPREC) :: intbb

	fnum = fnum + 1
	CALL compute_ffree_current()

	WRITE (fname, "(a12, i3.3)") "output/fout.", fnum
	IF (rank .EQ. 0) WRITE(*,*) rank,": name", fname

	IF (rank .EQ. 0) THEN
		OPEN(unit = 11, file = fname, form='unformatted')
		CLOSE(11)
	ENDIF

	skipflag = 0
	dimsf(1) = (mx0 - 5) / istep1
	dimsf(2) = (my0 - 5) / istep1
	dimsf(3) = (mz0 - 5) / istep1

	dimsfi(1:3) = dimsf(1:3)
	dimsfi(4:7) = 0
	datarank = 3

	nvars = 10

	IF (debug .AND. rank .EQ. 0) PRINT *, "in write_hdf"

	istart = 3
	ifinish = mx - 3
	310 IF (MODULO(istart - 2, istep1).EQ.0) GOTO 320
	istart = istart + 1
	GOTO 310
	320 CONTINUE

	330 IF (MODULO(ifinish - 2, istep1).EQ.0) GOTO 340
	ifinish = ifinish - 1
	GOTO 330
	340 CONTINUE
	mx1 = (ifinish - istart) / istep1 + 1

	kstart = 3
	kfinish = mz - 3

	110 IF (MODULO(kstart + (rank / sizey) * (mz - 5) - 2, istep1) .EQ. 0) GOTO 120
	kstart = kstart + 1
	GOTO 110
	120 CONTINUE

	130 IF (MODULO(kfinish + (rank / sizey) * (mz - 5) - 2, istep1) .EQ. 0) GOTO 140
	kfinish = kfinish - 1
	GOTO 130
	140 CONTINUE
	mz1 = (kfinish - kstart) / istep1 + 1

	jstart = 3
	jfinish = my - 3

	210 IF (MODULO(jstart + MODULO(rank, sizey) * (my - 5) - 2, istep1) .EQ. 0) &
	&     GOTO 220
	jstart = jstart + 1
	GOTO 210
	220 CONTINUE

	230 IF (MODULO(jfinish + MODULO(rank, sizey) * (my - 5) - 2, istep1)  &
	& .EQ. 0) GOTO 240
	jfinish = jfinish - 1
	GOTO 230
	240 CONTINUE

	my1 = (jfinish - jstart) / istep1 + 1

	ALLOCATE(temporary1(mx1, my1, mz1))

	!  Initialize FORTRAN predefined datatypes
	CALL h5open_f(error)
	! Setup file access property list with parallel I / O access.
	CALL h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
	CALL h5pset_fapl_mpio_f(plist_id, mpi_comm_world, mpi_info_null, &
	&     error)

	dsetname(1)="ex"
	dsetname(2)="ey"
	dsetname(3)="ez"
	dsetname(4)="bx"
	dsetname(5)="by"
	dsetname(6)="bz"
	dsetname(7)="jx"
	dsetname(8)="jy"
	dsetname(9)="jz"
	dsetname(10)="jpar"

	! Create the file collectively.
	CALL h5fcreate_f(fname, H5F_ACC_TRUNC_F, file_id, error, &
	&     access_prp = plist_id)
	CALL h5pclose_f(plist_id, error)
	! Create the data space for the  dataset.
	DO i = 1, nvars
		CALL h5screate_simple_f(datarank, dimsf, filespace(i), error)
	ENDDO

	! Create the dataset with default properties.
	DO i = 1, nvars
		CALL h5dcreate_f(file_id, dsetname(i), H5T_NATIVE_REAL, &
		&     filespace(i), dset_id(i), error)
		CALL h5sclose_f(filespace(i), error)
	ENDDO

	! Each process defines dataset in memory and writes it to the hyperslab
	! in the file.
	COUNT(1) = mx1 !dimsf(1)
	COUNT(2) = my1 !dimsf(2)
	COUNT(3) = mz1

	!need to communicate with others to find out the offset
	CALL mpi_allgather(mz1, 1, mpi_integer, all_counts, 1, mpi_integer,&
								   & mpi_comm_world, error)
	IF (rank .EQ. 0) PRINT *, "allcounts mz1=",all_counts
	offset(1) = 0
	offset(2) = 0
	offset(3) = 0
	!z offset
	IF (rank / sizey .GT. 0) THEN
		offset(3) = SUM(all_counts(1:(rank / sizey) * sizey:sizey))
		IF (debug) PRINT *, rank, ":", "offset3=", offset(3), "count=", count
	ENDIF
	!now get the y offset
	CALL mpi_allgather(my1, 1, mpi_integer, all_counts, 1, mpi_integer,&
									 & mpi_comm_world, error)

	IF (MODULO(rank, sizey) .GT. 0) THEN
		offset(2) = SUM(all_counts((rank / sizey) * sizey + 1 : rank))
	ENDIF
	IF (debug) PRINT *, rank, ":", "offset2=", offset(2)

	IF (debug) PRINT *, rank, ": offsets", offset
	CALL h5screate_simple_f(datarank, count, memspace, error)

	! Create property list for collective dataset write
	CALL h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
	CALL h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

	! Select hyperslab in the file.
	DO iv = 1, nvars
		CALL h5dget_space_f(dset_id(iv), filespace(iv), error)

		IF (debug) PRINT *, rank, ": t1"
		CALL h5sselect_hyperslab_f (filespace(iv), H5S_SELECT_SET_F, offset, &
															& count, error)
		IF (debug) PRINT *, rank, ": t2"
		! Write the dataset collectively.

		DO k = kstart, kfinish, istep1 !1, mz / istep1
			nz = (k - kstart) / istep1 + 1
			DO j = jstart, jfinish, istep1 !1, my / istep1
				ny = (j - jstart) / istep1 + 1
				DO i = istart, ifinish, istep1 !mx / istep1
					nx = (i - istart) / istep1 + 1
					IF ((i .NE. 1) .AND. (j .NE. 1) .AND. (k .NE. 1)) THEN
						!---------------------interpolate E to node-----------------
						intex = 0.5 * (f(i, j, k).ex + f(i - 1, j, k).ex)
						intey = 0.5 * (f(i, j, k).ey + f(i, j - 1, k).ey)
						intez = 0.5 * (f(i, j, k).ez + f(i, j, k - 1).ez)
						!--------------------interpolate B to node------------------
						intbx = 0.25 * (f(i, j, k).bx + f(i, j - 1, k).bx + f(i, j, k - 1).bx + f(i, j - 1, k - 1).bx)
						intby = 0.25 * (f(i, j, k).by + f(i - 1, j, k).by + f(i, j, k - 1).by + f(i - 1, j, k - 1).by)
						intbz = 0.25 * (f(i, j, k).bz + f(i - 1, j, k).bz + f(i - 1, j - 1, k).bz + f(i, j - 1, k).bz)
						!-----------------------------------------------------------
					ELSE
						intex = 0
						intey = 0
						intez = 0
						intbx = 0
						intby = 0
						intbz = 0
					ENDIF
					intbb = sqrt(intbx**2 + intby**2 + intbz**2)

					IF (iv .EQ. 1) temporary1(nx, ny, nz) = REAL(intex, 4)
					IF (iv .EQ. 2) temporary1(nx, ny, nz) = REAL(intey, 4)
					IF (iv .EQ. 3) temporary1(nx, ny, nz) = REAL(intez, 4)
					IF (iv .EQ. 4) temporary1(nx, ny, nz) = REAL(intbx, 4)
					IF (iv .EQ. 5) temporary1(nx, ny, nz) = REAL(intby, 4)
					IF (iv .EQ. 6) temporary1(nx, ny, nz) = REAL(intbz, 4)
					! current is already inteprolated to nodes
					IF (iv .EQ. 7) temporary1(nx, ny, nz) = REAL(ff_curr(i, j, k).jx, 4)
					IF (iv .EQ. 8) temporary1(nx, ny, nz) = REAL(ff_curr(i, j, k).jy, 4)
					IF (iv .EQ. 9) temporary1(nx, ny, nz) = REAL(ff_curr(i, j, k).jz, 4)
					! current is already inteprolated to nodes
					IF (iv .EQ. 10) temporary1(nx, ny, nz) = REAL((ff_curr(i, j, k).jx * intbx +&
																										   & ff_curr(i, j, k).jy * intby +&
																											 & ff_curr(i, j, k).jz * intbz) / intbb, 4)
				ENDDO
			ENDDO
		ENDDO
		CALL h5dwrite_real_1(dset_id(iv), H5T_NATIVE_REAL, temporary1(:,:,:),&
											 & dimsfi, error, file_space_id = filespace(iv), mem_space_id = memspace,&
											 & xfer_prp = plist_id)
	ENDDO ! ivar
	DO iv = 1, nvars
		CALL h5sclose_f(filespace(iv), error)
	ENDDO
	CALL h5sclose_f(memspace, error)
	DO iv = 1, nvars
		CALL h5dclose_f(dset_id(iv), error)
	ENDDO
	CALL h5pclose_f(plist_id, error)
	CALL h5fclose_f(file_id, error)
	CALL h5close_f(error)
	DEALLOCATE(temporary1)

	IF (debug) PRINT *, rank,": finished writing fields"

END SUBROUTINE write_hdf
#endif
