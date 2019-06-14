#include "defs.f90"

SUBROUTINE EM_step_rk3()
  INTEGER i, j, k
  REAL tstep(3)
  DOUBLE PRECISION t0, t1, t2

  tstep(1) = 0
  tstep(2) = 1
  tstep(3) = 0.5

  fn = f

  i0 = 2
  i1 = mx - 1

  DO substep = 1, 3
    IF (rank .EQ. 0) PRINT *,"substep:", substep

    IF (rank .EQ. 0) t0 = mpi_wtime()

    IF (calc_curr .EQ. 1) CALL find_rho()

    DO  k = 2, mz - 1
      DO  j = 2, my - 1
        DO  i = i0, i1 !2, mx - 1

          df(i, j, k).bx = c * (f(i, j, k + 1).ey - f(i, j, k).ey - f(i, j + 1, k).ez + f(i, j, k).ez)
          df(i, j, k).by = c * (f(i + 1, j, k).ez - f(i, j, k).ez - f(i, j, k + 1).ex + f(i, j, k).ex)
          df(i, j, k).bz = c * (f(i, j + 1, k).ex - f(i, j, k).ex - f(i + 1, j, k).ey + f(i, j, k).ey)

          df(i, j, k).ex= c * (f(i, j, k - 1).by - f(i, j, k).by - f(i, j - 1, k).bz + f(i, j, k).bz &
                           & -(f0(i, j, k - 1).by0 - f0(i, j, k).by0 - f0(i, j - 1, k).bz0 + f0(i, j, k).bz0))


          df(i, j, k).ey= c * (f(i - 1, j, k).bz - f(i, j, k).bz - f(i, j, k - 1).bx + f(i, j, k).bx &
                           & -(f0(i - 1, j, k).bz0 - f0(i, j, k).bz0 - f0(i, j, k - 1).bx0 + f0(i, j, k).bx0))


          df(i, j, k).ez= c * (f(i, j - 1, k).bx - f(i, j, k).bx - f(i - 1, j, k).by + f(i, j, k).by &
                           & -(f0(i, j - 1, k).bx0 - f0(i, j, k).bx0 - f0(i - 1, j, k).by0 + f0(i, j, k).by0))

          IF (calc_curr .EQ. 1) THEN
            !--------X current-------------------------------------
            !-------interpolate rho_x---------------------------
            intrho = .5 * (rho(i + 1, j, k) + rho(i, j, k))

            !-------interpolate E x-----------------------------
            intex = f(i, j, k).ex
            intey = .25 * (f(i, j, k).ey + f(i + 1, j, k).ey + f(i, j - 1, k).ey &
            &           +f(i + 1, j - 1, k).ey)
            intez = .25 * (f(i, j, k).ez + f(i + 1, j, k).ez + f(i, j, k - 1).ez &
            &           +f(i + 1, j, k - 1).ez)
            !---------------------------------------------------
            !-------interpolate B x-----------------------------
            intbx = .125 * (f(i, j, k).bx + f(i, j - 1, k).bx + f(i + 1, j - 1, k).bx + f(i + 1, j, k).bx+&
            & f(i, j, k - 1).bx + f(i, j - 1, k - 1).bx + f(i + 1, j - 1, k - 1).bx + f(i + 1, j, k - 1).bx)
            intby = .5 * (f(i, j, k).by + f(i, j, k - 1).by)
            intbz = .5 * (f(i, j, k).bz + f(i, j - 1, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            IF (emag .GT. bmag) THEN
              arr = SQRT(bmag / emag)
              !         intex = intex * arr
              intey = intey * arr
              intez = intez * arr
            ENDIF

            jx = c / (bmag + delta0) * (intrho * (intey * intbz - intby * intez))
            !--------Y current--------------------------------------
            !-------interpolate rho_y---------------------------
            intrho = .5 * (rho(i, j + 1, k) + rho(i, j, k))

            !-------interpolate E y-----------------------------

            intex = .25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j + 1, k).ex &
            &           +f(i - 1, j + 1, k).ex)
            intey = f(i, j, k).ey
            intez = .25 * (f(i, j, k).ez + f(i, j + 1, k).ez + f(i, j, k - 1).ez &
            &           +f(i, j + 1, k - 1).ez)

            !---------------------------------------------------
            !-------interpolate B y-----------------------------
            intbx = .5 * (f(i, j, k).bx + f(i, j, k - 1).bx)
            intby = .125 * (f(i, j, k).by + f(i - 1, j, k).by + f(i - 1, j + 1, k).by + f(i, j + 1, k).by+&
            & f(i, j, k - 1).by + f(i - 1, j, k - 1).by + f(i - 1, j + 1, k - 1).by + f(i, j + 1, k - 1).by )
            intbz = .5 * (f(i, j, k).bz + f(i - 1, j, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            IF (emag .GT. bmag) THEN
              arr = SQRT(bmag / emag)
              intex = intex * arr
              !         intey = intey * arr
              intez = intez * arr
            ENDIF

            jy = c / (bmag + delta0) * (intrho * (intez * intbx -intex * intbz))


            !--------Z current--------------------------------------
            !-------interpolate rho_z---------------------------
            intrho = .5 * (rho(i, j, k) + rho(i, j, k + 1))

            !-------interpolate E z-----------------------------
            intex = .25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j, k + 1).ex &
            &           +f(i - 1, j, k + 1).ex)
            intey = .25 * (f(i, j, k).ey + f(i, j - 1, k).ey + f(i, j, k + 1).ey &
            &           +f(i, j - 1, k + 1).ey)
            intez = f(i, j, k).ez
            !---------------------------------------------------
            !-------interpolate B z-----------------------------
            intbx = .5 * (f(i, j, k).bx + f(i, j - 1, k).bx)
            intby = .5 * (f(i, j, k).by + f(i - 1, j, k).by)
            intbz = .125 * (f(i, j, k).bz + f(i - 1, j, k).bz + f(i - 1, j - 1, k).bz + f(i, j - 1, k).bz+&
            &  f(i, j, k + 1).bz + f(i - 1, j, k + 1).bz + f(i - 1, j - 1, k + 1).bz + f(i, j - 1, k + 1).bz )
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            IF (emag .GT. bmag) THEN
              arr = SQRT(bmag / emag)
              intex = intex * arr
              intey = intey * arr
              !         intez = intez * arr
            ENDIF

            jz = c / (bmag + delta0) * (intrho * (intex * intby -intbx * intey))

            df(i, j, k).ex = df(i, j, k).ex - jx
            df(i, j, k).ey = df(i, j, k).ey - jy
            df(i, j, k).ez = df(i, j, k).ez - jz
          ENDIF
        ENDDO
      ENDDO
    ENDDO

    IF (substep .EQ. 1) THEN
      DO k = 2, mz - 1
        DO j = 2, my - 2
          DO i = i0, i1 !2, mx - 1
            f(i, j, k).ex = fn(i, j, k).ex + df(i, j, k).ex
            f(i, j, k).ey = fn(i, j, k).ey + df(i, j, k).ey
            f(i, j, k).ez = fn(i, j, k).ez + df(i, j, k).ez
            f(i, j, k).bx = fn(i, j, k).bx + df(i, j, k).bx
            f(i, j, k).by = fn(i, j, k).by + df(i, j, k).by
            f(i, j, k).bz = fn(i, j, k).bz + df(i, j, k).bz

            df(i, j, k).ex = f(i, j, k).ex
            df(i, j, k).ey = f(i, j, k).ey
            df(i, j, k).ez = f(i, j, k).ez
            df(i, j, k).bx = f(i, j, k).bx
            df(i, j, k).by = f(i, j, k).by
            df(i, j, k).bz = f(i, j, k).bz
          ENDDO
        ENDDO
      ENDDO
    ENDIF
    IF (substep .EQ. 2) THEN
      DO k = 2, mz - 1
        DO j = 2, my - 2
          DO i = i0, i1 !2, mx - 1
            f(i, j, k).ex = .25 * (3 * fn(i, j, k).ex + f(i, j, k).ex + df(i, j, k).ex)
            f(i, j, k).ey = .25 * (3 * fn(i, j, k).ey + f(i, j, k).ey + df(i, j, k).ey)
            f(i, j, k).ez = .25 * (3 * fn(i, j, k).ez + f(i, j, k).ez + df(i, j, k).ez)
            f(i, j, k).bx = .25 * (3 * fn(i, j, k).bx + f(i, j, k).bx + df(i, j, k).bx)
            f(i, j, k).by = .25 * (3 * fn(i, j, k).by + f(i, j, k).by + df(i, j, k).by)
            f(i, j, k).bz = .25 * (3 * fn(i, j, k).bz + f(i, j, k).bz + df(i, j, k).bz)

            df(i, j, k).ex = f(i, j, k).ex
            df(i, j, k).ey = f(i, j, k).ey
            df(i, j, k).ez = f(i, j, k).ez
            df(i, j, k).bx = f(i, j, k).bx
            df(i, j, k).by = f(i, j, k).by
            df(i, j, k).bz = f(i, j, k).bz

          ENDDO
        ENDDO
      ENDDO
    ENDIF

    IF (substep .EQ. 3) THEN
      DO k = 2, mz - 1
        DO j = 2, my - 2
          DO i = i0, i1 !2, mx - 1
            f(i, j, k).ex = .3333333 * (fn(i, j, k).ex + 2 * f(i, j, k).ex + 2 * df(i, j, k).ex)
            f(i, j, k).ey = .3333333 * (fn(i, j, k).ey + 2 * f(i, j, k).ey + 2 * df(i, j, k).ey)
            f(i, j, k).ez = .3333333 * (fn(i, j, k).ez + 2 * f(i, j, k).ez + 2 * df(i, j, k).ez)
            f(i, j, k).bx = .3333333 * (fn(i, j, k).bx + 2 * f(i, j, k).bx + 2 * df(i, j, k).bx)
            f(i, j, k).by = .3333333 * (fn(i, j, k).by + 2 * f(i, j, k).by + 2 * df(i, j, k).by)
            f(i, j, k).bz = .3333333 * (fn(i, j, k).bz + 2 * f(i, j, k).bz + 2 * df(i, j, k).bz)

            ! because going into clean_epar which uses f, don't need df.e now.
            df(i, j, k).ex = f(i, j, k).ex
            df(i, j, k).ey = f(i, j, k).ey
            df(i, j, k).ez = f(i, j, k).ez
            df(i, j, k).bx = f(i, j, k).bx
            df(i, j, k).by = f(i, j, k).by
            df(i, j, k).bz = f(i, j, k).bz

          ENDDO
        ENDDO
      ENDDO
    ENDIF

    IF (rank .EQ. 0) PRINT *, "update",mpi_wtime() - t0

    IF (rank .EQ. 0) t0 = mpi_wtime()

    !clean on every step but only in the third substep

    IF (clean_ep .EQ. 1 .AND. substep .EQ. 3) THEN
      !        IF (clean_ep .eq. 1) then
      !      df = f !already copied in the update step

      DO  k = 2, mz - 1
        DO  j = 2, my - 1
          DO  i = i0, i1 !2, mx - 1

            !-------interpolate E x-----------------------------
            intex = f(i, j, k).ex
            intey = .25 * (f(i, j, k).ey + f(i + 1, j, k).ey + f(i, j - 1, k).ey &
            &           +f(i + 1, j - 1, k).ey)
            intez = .25 * (f(i, j, k).ez + f(i + 1, j, k).ez + f(i, j, k - 1).ez &
            &           +f(i + 1, j, k - 1).ez)
            !---------------------------------------------------
            !-------interpolate B x-----------------------------
            intbx = .125 * (f(i, j, k).bx + f(i, j - 1, k).bx + f(i + 1, j - 1, k).bx + f(i + 1, j, k).bx+&
            & f(i, j, k - 1).bx + f(i, j - 1, k - 1).bx + f(i + 1, j - 1, k - 1).bx + f(i + 1, j, k - 1).bx)
            intby = .5 * (f(i, j, k).by + f(i, j, k - 1).by)
            intbz = .5 * (f(i, j, k).bz + f(i, j - 1, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            epar = (intex * intbx + intey * intby +intez * intbz)
            df(i, j, k).ex = f(i, j, k).ex - epar * intbx / bmag


            !-------interpolate E y-----------------------------
            intex = .25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j + 1, k).ex &
            &           +f(i - 1, j + 1, k).ex)
            intey = f(i, j, k).ey
            intez = .25 * (f(i, j, k).ez + f(i, j + 1, k).ez + f(i, j, k - 1).ez &
            &           +f(i, j + 1, k - 1).ez)
            !---------------------------------------------------
            !-------interpolate B y-----------------------------
            intbx = .5 * (f(i, j, k).bx + f(i, j, k - 1).bx)
            intby = .125 * (f(i, j, k).by + f(i - 1, j, k).by + f(i - 1, j + 1, k).by + f(i, j + 1, k).by+&
            & f(i, j, k - 1).by + f(i - 1, j, k - 1).by + f(i - 1, j + 1, k - 1).by + f(i, j + 1, k - 1).by )
            intbz = .5 * (f(i, j, k).bz + f(i - 1, j, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            epar = (intex * intbx + intey * intby +intez * intbz)
            df(i, j, k).ey = f(i, j, k).ey - epar * intby / bmag


            !-------interpolate E z-----------------------------
            intex = .25 * (f(i, j, k).ex + f(i - 1, j, k).ex + f(i, j, k + 1).ex &
            &           +f(i - 1, j, k + 1).ex)
            intey = .25 * (f(i, j, k).ey + f(i, j - 1, k).ey + f(i, j, k + 1).ey &
            &           +f(i, j - 1, k + 1).ey)
            intez = f(i, j, k).ez
            !---------------------------------------------------
            !-------interpolate B z-----------------------------

            intbx = .5 * (f(i, j, k).bx + f(i, j - 1, k).bx)
            intby = .5 * (f(i, j, k).by + f(i - 1, j, k).by)
            intbz = .125 * (f(i, j, k).bz + f(i - 1, j, k).bz + f(i - 1, j - 1, k).bz + f(i, j - 1, k).bz+&
            &  f(i, j, k + 1).bz + f(i - 1, j, k + 1).bz + f(i - 1, j - 1, k + 1).bz + f(i, j - 1, k + 1).bz )

            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            epar = (intex * intbx + intey * intby +intez * intbz)
            df(i, j, k).ez = f(i, j, k).ez - epar * intbz / bmag

          ENDDO
        ENDDO
      ENDDO
    ENDIF
    !current array is in df
    IF (rank .EQ. 0) PRINT *, "clean_epar",mpi_wtime() - t0

    IF (rank .EQ. 0) t0 = mpi_wtime()
    IF ((lap .GT. tch) .AND. (check_e .EQ. 1)) THEN
      DO  k = 2, mz - 1
        DO  j = 2, my - 1
          DO  i = i0, i1 !2, mx - 1

            !-------interpolate E x-----------------------------
            intex = df(i, j, k).ex
            intey = .25 * (df(i, j, k).ey + df(i + 1, j, k).ey + df(i, j - 1, k).ey &
            &           +df(i + 1, j - 1, k).ey)
            intez = .25 * (df(i, j, k).ez + df(i + 1, j, k).ez + df(i, j, k - 1).ez &
            &           +df(i + 1, j, k - 1).ez)
            !---------------------------------------------------
            !-------interpolate B x-----------------------------
            intbx = .125 * (df(i, j, k).bx + df(i, j - 1, k).bx + df(i + 1, j - 1, k).bx + df(i + 1, j, k).bx+&
            & df(i, j, k - 1).bx + df(i, j - 1, k - 1).bx + df(i + 1, j - 1, k - 1).bx + df(i + 1, j, k - 1).bx)
            intby = .5 * (df(i, j, k).by + df(i, j, k - 1).by)
            intbz = .5 * (df(i, j, k).bz + df(i, j - 1, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            arr = 1.
            IF (emag .GT. bmag) THEN
              !         print*,'**************',lap, i, j + modulo(rank, sizey) * (my - 5), k + (rank / sizey) * (mz - 5)
              arr = SQRT(bmag / emag)
            ENDIF
            f(i, j, k).ex = df(i, j, k).ex * arr


            !-------interpolate E y-----------------------------
            intex = .25 * (df(i, j, k).ex + df(i - 1, j, k).ex + df(i, j + 1, k).ex &
            &           +df(i - 1, j + 1, k).ex)
            intey = df(i, j, k).ey
            intez = .25 * (df(i, j, k).ez + df(i, j + 1, k).ez + df(i, j, k - 1).ez &
            &           +df(i, j + 1, k - 1).ez)
            !---------------------------------------------------
            !-------interpolate B y-----------------------------
            intbx = .5 * (df(i, j, k).bx + df(i, j, k - 1).bx)
            intby = .125 * (df(i, j, k).by + df(i - 1, j, k).by + df(i - 1, j + 1, k).by + df(i, j + 1, k).by+&
            & df(i, j, k - 1).by + df(i - 1, j, k - 1).by + df(i - 1, j + 1, k - 1).by + df(i, j + 1, k - 1).by )
            intbz = .5 * (df(i, j, k).bz + df(i - 1, j, k).bz)
            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            arr = 1.
            IF (emag .GT. bmag) THEN
              !         print*,'**************',lap, i, j + modulo(rank, sizey) * (my - 5), k + (rank / sizey) * (mz - 5)
              arr = SQRT(bmag / emag)
            ENDIF
            f(i, j, k).ey = df(i, j, k).ey * arr

            !-------interpolate E z-----------------------------
            intex = .25 * (df(i, j, k).ex + df(i - 1, j, k).ex + df(i, j, k + 1).ex &
            &           +df(i - 1, j, k + 1).ex)
            intey = .25 * (df(i, j, k).ey + df(i, j - 1, k).ey + df(i, j, k + 1).ey &
            &           +df(i, j - 1, k + 1).ey)
            intez = df(i, j, k).ez
            !---------------------------------------------------
            !-------interpolate B z-----------------------------

            intbx = .5 * (df(i, j, k).bx + df(i, j - 1, k).bx)
            intby = .5 * (df(i, j, k).by + df(i - 1, j, k).by)
            intbz = .125 * (df(i, j, k).bz + df(i - 1, j, k).bz + df(i - 1, j - 1, k).bz + df(i, j - 1, k).bz+&
            &  df(i, j, k + 1).bz + df(i - 1, j, k + 1).bz + df(i - 1, j - 1, k + 1).bz + df(i, j - 1, k + 1).bz )

            !---------------------------------------------------

            bmag = intbx**2 + intby**2 + intbz**2
            emag = intex**2 + intey**2 + intez**2
            arr = 1.
            IF (emag .GT. bmag) THEN
              arr = SQRT(bmag / emag)
            ENDIF

            f(i, j, k).ez = df(i, j, k).ez * arr

          ENDDO
        ENDDO
      ENDDO
    ENDIF
    !current array is in f
    IF (rank .EQ. 0) PRINT *, "check_eb",mpi_wtime() - t0

    IF (rank .EQ. 0) t0 = mpi_wtime()
    IF (substep .EQ. 3) THEN
      IF (rank .EQ. 0)  PRINT *, "eta=",eta
      CALL add_diffusion()

    ENDIF !substep .eq.3
    IF (rank .EQ. 0) PRINT *, "diffusion",mpi_wtime() - t0


    IF (rank .EQ. 0) t0 = mpi_wtime()
    CALL boundary_conditions(substep)
    IF (rank .EQ. 0) PRINT *, "BC",mpi_wtime() - t0

  ENDDO

END SUBROUTINE EM_step_rk3


SUBROUTINE add_diffusion()
  DO  k = 2, mz - 1
    DO  j = 2, my - 1
      DO  i = i0, i1 !2, mx - 1
        df(i, j, k).ex = f(i + 1, j, k).ex + f(i - 1, j, k).ex+     &
        & f(i, j + 1, k).ex + f(i, j - 1, k).ex + f(i, j, k + 1).ex + f(i, j, k - 1).ex - 6 * f(i, j, k).ex

        df(i, j, k).ey = f(i + 1, j, k).ey + f(i - 1, j, k).ey+     &
        & f(i, j + 1, k).ey + f(i, j - 1, k).ey + f(i, j, k + 1).ey + f(i, j, k - 1).ey - 6 * f(i, j, k).ey

        df(i, j, k).ez = f(i + 1, j, k).ez + f(i - 1, j, k).ez+     &
        & f(i, j + 1, k).ez + f(i, j - 1, k).ez + f(i, j, k + 1).ez + f(i, j, k - 1).ez - 6 * f(i, j, k).ez
      ENDDO
    ENDDO
  ENDDO

  DO  k = 2, mz - 1
    DO  j = 2, my - 1
      DO  i = i0, i1 !2, mx - 1
        f(i, j, k).ex = f(i, j, k).ex + dt * c * eta * df(i, j, k).ex
        f(i, j, k).ey = f(i, j, k).ey + dt * c * eta * df(i, j, k).ey
        f(i, j, k).ez = f(i, j, k).ez + dt * c * eta * df(i, j, k).ez
      ENDDO
    ENDDO
  ENDDO
END SUBROUTINE add_diffusion
