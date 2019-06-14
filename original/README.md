# COmputational Force-FreE Electrodynamics [Coffee]

### Usage on `perseus`
Use the following modules when using the code on `perseus` cluster:
```bash
module load intel-mkl/11.3.4/4/64
module load intel/16.0/64/16.0.4.258
module load openmpi/intel-16.0/1.10.2/64
module load hdf5/intel-16.0/openmpi-1.10.2/1.8.16
```

Then `make` with `$ make all`, the executable is generated at the `exec/` directory.

Two user files (initialization and boundary conditions) are specified using the `INCLUDE` command, e.g.:
```fortran
INCLUDE "user_init_loop.f90"
INCLUDE "user_bc_loop.f90"
```

Two user defined subroutines called in the main loop are `user_init_fields()` and `user_bc()` (specified in the included `.f90` files). When adding new `.f90` files you also need to add them to the `Makefile` (in the `SRC_FILES` variable), so the precompiler knows about their existence.

### Tips
1. `#ifdef` statements can be inline, since the `Makefile` does an extra precompilation step and generates new files in the `src_/` directory.
2. Any problem specific variables should be defined in the `ffree3D.F90` file (and then properly read from the input in `read_input()` subroutine), e.g.:
  ```fortran
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
  ```
3. Force-free currents are computed at every output step and written into `jx`, `jy` and `jz` fields in the `hdf5`.
4. It is highly discouraged to add any problem specific routines into the main files (files that don't have the `user_` prescription).
5. `MPREC` is a global definition (defined in `defs.F90`) that is either `4` or `8` depending on whether the program should be compiled in a single or double precision.
6. General advices for the coding style in Fortran are applied, see [here](https://github.com/PrincetonUniversity/tristan-v2/#coding-style-advices).
