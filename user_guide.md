# CoffeeGPU User Guide

This guide focuses on two things:

1. What each field solver actually does during a timestep.
2. How each compiled executable is configured: which solver it uses, which
   problem it represents, which setup header initializes the fields, and which
   runtime parameters matter.

## Common Runtime Workflow

All executables follow the same high-level structure:

1. Construct `sim_environment env(&argc, &argv)`.
2. Parse `config.toml` into `sim_params`.
3. Build the local MPI-decomposed `Grid`.
4. Construct `sim_data data(env)`.
5. Construct a solver.
6. Apply one `user_*.hpp` setup block to initialize fields.
7. Enter the timestep loop:
   - write output every `data_interval`
   - optionally write restart snapshots every `snapshot_interval`
   - optionally write slices every `slice_interval`
   - call the solver's timestep routine

Common parameters used by almost every executable:

- `dt`
- `max_steps`
- `data_interval`
- `snapshot_interval`
- `N`
- `guard`
- `lower`
- `size`
- `nodes`
- `shift_ghost`
- `periodic_boundary`
- `downsample`
- `slice_x`, `slice_y`, `slice_z`, `slice_xy`
- `slice_interval`
- `downsample_2d`

Common solver controls:

- `clean_ep`
- `check_egb`
- `pml`
- `pmllen`
- `sigpml`

EZ-family solvers also use:

- `divB_clean`
- `ch2`
- `tau`
- `KOeps`
- `KO_geometry`
- `use_edotb_damping`
- `damp_gamma`

## Field Solvers

### `field_solver`

File:
[src/algorithms/field_solver.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver.h)

Implementation path:
[src/algorithms/field_solver.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver.cu)

This is the original Yee-style 3D solver. It advances `E` and `B` with a
three-stage SSP Runge-Kutta sequence.

Per timestep workflow:

1. Copy the current fields into stage storage (`En`, `Bn`) and zero the update
   accumulators.
2. Run substep 1:
   - compute one RK push
   - update with coefficients `(1, 0, 1)`
   - enforce `E^2 <= B^2` with `check_eGTb()`
   - apply pulsar boundary if `problem == 1`
   - exchange guard cells
3. Run substep 2:
   - compute one RK push
   - update with coefficients `(0.75, 0.25, 0.25)`
   - enforce `E^2 <= B^2`
   - apply pulsar boundary if `problem == 1`
   - exchange guard cells
4. Run substep 3:
   - compute one RK push
   - update with coefficients `(1/3, 2/3, 2/3)`
   - clean `E_parallel` with `clean_epar()`
   - enforce `E^2 <= B^2`
   - apply pulsar boundary if `problem == 1`
   - apply absorbing outer boundary
   - exchange guard cells

Boundary model:

- Pulsar boundary for `problem == 1`
- Absorbing boundary after the last substep
- MPI halo exchange after every substep

Notes:

- This solver is used as the non-EZ alternative for 3D pulsar and Alfven-from-
  pulsar applications.
- In the current build configuration, the implementation used here is the CUDA
  one.

### `field_solver_EZ`

Files:
[src/algorithms/field_solver_EZ.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ.h)
[src/algorithms/field_solver_EZ.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ.cpp)
[src/algorithms/field_solver_EZ.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ.cu)

This is the main extended EZ solver used by the 3D pulsar, Alfven-from-pulsar,
and periodic-box wave executables. It evolves `E`, `B`, and the scalar cleaning
field `P`.

Per timestep workflow:

1. Save copies of `E`, `B`, and `P` into temporary arrays.
2. Zero per-step diagnostic accumulators such as `dU_KO`, `dU_Epar`,
   `dU_EgtB`.
3. Run a five-stage low-storage Runge-Kutta update.
4. At each RK stage:
   - call `rk_step(As[i], Bs[i])`
   - clean and clamp force-free constraints
   - in the CPU path this is combined in `clean_epar_check_eGTb()`
   - apply application-specific boundary conditions
   - exchange guard cells
5. After the five RK stages:
   - apply Kreiss-Oliger dissipation
   - clean/clamp again
   - re-apply the application boundary
   - exchange guard cells again

Boundary model:

- `problem == 1`: `boundary_pulsar(t)`
- `problem == 2`: `boundary_alfven(t)`
- final RK stage: absorbing boundary

Important parameters:

- `clean_ep`
- `check_egb`
- `KOeps`
- `KO_geometry`
- `divB_clean`
- `ch2`
- `tau`
- `use_edotb_damping`
- `damp_gamma`
- `pml`, `pmllen`, `sigpml`

Notes:

- This is the most feature-complete explicit solver in the codebase.
- It assumes explicit MPI halo exchange between every RK stage.

### `field_solver_EZ_cylindrical`

Files:
[src/algorithms/field_solver_EZ_cylindrical.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ_cylindrical.h)
[src/algorithms/field_solver_EZ_cylindrical.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ_cylindrical.cu)

This is the axisymmetric cylindrical pulsar solver.

Per timestep workflow:

1. Save copies of `E` and `B`.
2. Run a five-stage low-storage RK update.
3. At each RK stage:
   - update fields with `rk_step`
   - optionally clean `E_parallel`
   - optionally enforce `E^2 <= B^2`
   - enforce the symmetry axis boundary
   - enforce the pulsar boundary
   - on the last stage apply the absorbing boundary
   - exchange field guard cells and the scalar `P`
4. After the five stages:
   - apply Kreiss-Oliger dissipation
   - repeat cleaning/clamping
   - re-apply axis and pulsar boundaries
   - exchange halos

Boundary model:

- `boundary_axis()`
- `boundary_pulsar(t)`
- `boundary_absorbing()`

Important parameters:

- all common EZ controls
- `radius`
- `b0`
- `omega`

### `field_solver_EZ_spherical`

Files:
[src/algorithms/field_solver_EZ_spherical.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ_spherical.h)
[src/algorithms/field_solver_EZ_spherical.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_EZ_spherical.cu)

This is the spherical-coordinate axisymmetric pulsar solver.

Per timestep workflow:

1. Apply the axis boundary before entering the RK loop.
2. Save copies of `E`, `B`, and `P`.
3. Run a five-stage low-storage RK update.
4. At each RK stage:
   - compute local metric-adapted fields with `get_ElBl()`
   - perform `rk_step`
   - optionally clean `E_parallel`
   - optionally enforce `E^2 <= B^2`
   - apply the pulsar boundary
   - on the last stage apply the absorbing boundary
   - re-apply the axis boundary
   - exchange guard cells
5. After the five stages:
   - apply Kreiss-Oliger dissipation
   - repeat cleaning/clamping
   - apply pulsar and axis boundaries
   - exchange guard cells

Boundary model:

- `boundary_axis()`
- `boundary_pulsar(t)`
- `boundary_absorbing()`

Important parameters:

- all common EZ controls
- `radius`
- `b0`
- `omega`
- coordinate geometry from the grid definition in `lower`, `size`, `N`

### `field_solver_gr`

Files:
[src/algorithms/field_solver_gr.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_gr.h)
[src/algorithms/field_solver_gr.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_gr.cu)

This is the general-relativistic Yee-like solver. It evolves the GR variables
using auxiliary `Ed` and `Hd` fields derived from the metric.

Per timestep workflow:

1. Copy current fields into stage storage.
2. For each of three RK substeps:
   - compute `E_gr`
   - compute `H_gr`
   - do one RK push
   - do one RK update
   - optionally enforce `E^2 <= B^2`
   - exchange guard cells
3. On the last substep:
   - optionally clean `E_parallel`
   - apply absorbing boundary
   - compute `divB`
   - exchange guard cells

Boundary model:

- absorbing outer boundary
- no pulsar boundary in the current GR path

Important parameters:

- `a` for black-hole spin
- `clean_ep`
- `check_egb`
- `pml`, `pmllen`, `sigpml`

### `field_solver_gr_EZ`

Files:
[src/algorithms/field_solver_gr_EZ.h](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_gr_EZ.h)
[src/algorithms/field_solver_gr_EZ.cu](/Volumes/yajiey/Active/codes/CoffeeGPU/src/algorithms/field_solver_gr_EZ.cu)

This is the general-relativistic EZ solver. It combines the five-stage EZ
update with GR metric conversions through `Ed` and `Hd`.

Per timestep workflow:

1. Save copies of the evolved electric-like field `D`, magnetic field `B`, and
   `P`.
2. Run a five-stage low-storage RK update.
3. At each RK stage:
   - compute `Ed` with `get_Ed()`
   - compute `Hd` with `get_Hd()`
   - advance the fields with `rk_step`
   - optionally clean `E_parallel`
   - optionally enforce `E^2 <= B^2`
   - on the final stage apply absorbing boundary
   - exchange guard cells
4. After the RK loop:
   - apply Kreiss-Oliger dissipation
   - repeat cleaning/clamping
   - exchange guard cells

Boundary model:

- absorbing outer boundary only

Important parameters:

- `a`
- all common EZ controls
- `pml`, `pmllen`, `sigpml`

## Executables and Applications

This section lists each compiled executable from
[src/CMakeLists.txt](/Volumes/yajiey/Active/codes/CoffeeGPU/src/CMakeLists.txt).

### `coffee`

Entrypoint:
[src/main.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main.cpp)

Build condition:
only when `use_cuda=ON`

Current application:

- GR EZ run with Wald-type initialization

Solver:

- `field_solver_gr_EZ`

Initialization include:

- `user_wald2.hpp`

Files:
[src/user_wald2.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_wald2.hpp)

Initial conditions:

- Sets all three `B` components to represent a uniform `Bz` background in curved
  spacetime.
- Initializes `E = 0`.
- Sets `B0 = 0`.
- Sets `P = 0`.
- Uses unstaggered `(0b111)` storage for `E`, `B`, and `B0`.

Boundary conditions:

- absorbing outer boundary from `field_solver_gr_EZ`
- MPI halo exchange every RK stage

Important parameters:

- `a`
- `clean_ep`
- `check_egb`
- `KOeps`
- `divB_clean`
- `pml`, `pmllen`, `sigpml`

### `coffee-gr`

Entrypoint:
[src/main-gr.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-gr.cpp)

Build condition:
only when `use_cuda=ON`

Current application:

- same application family as `coffee`
- GR EZ evolution with Wald-type initialization

Solver:

- `field_solver_gr_EZ`

Initialization include:

- `user_wald2.hpp`

Behavior:

- supports single-file snapshot restart through `env.is_restart()`
- writes periodic restart snapshots

Important parameters:

- same as `coffee`
- `snapshot_interval`

### `coffee-gr-Yee`

Entrypoint:
[src/main-gr-Yee.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-gr-Yee.cpp)

Build condition:
only when `use_cuda=ON`

Current application:

- GR Yee evolution of a Wald solution

Solver:

- `field_solver_gr`

Initialization include:

- `user_wald.hpp`

Files:
[src/user_wald.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_wald.hpp)

Initial conditions:

- builds a vector potential `A = solver.Dn`
- computes `B = curl(A) / sqrt(gamma)`
- computes an electrostatic potential `A0 = solver.rho`
- reconstructs `E` from metric coefficients, `A0`, and interpolated `B`
- initializes `B0 = 0`

Boundary conditions:

- absorbing outer boundary in the GR solver
- no pulsar boundary

Important parameters:

- `a`
- `clean_ep`
- `check_egb`
- `pml`, `pmllen`, `sigpml`
- `snapshot_interval`

### `coffee-wave`

Entrypoint:
[src/main-wave.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-wave.cpp)

Build condition:
only when `use_cuda=ON`

Application:

- periodic-box Alfven wave

Runtime check:

- requires `problem = 0`

Solver:

- `field_solver_EZ`

Initialization include:

- `user_alfven.hpp`

Files:
[src/user_alfven.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_alfven.hpp)

Initial conditions:

- builds a sinusoidal Alfven wave in a periodic box
- sets `E` from a plane-wave phase
- for EZ mode, initializes `B` directly
- for non-EZ mode, would initialize `B` from a vector potential
- sets `B0 = 0`
- in EZ mode forces all components to unstaggered `(0b111)`

Boundary conditions:

- intended to run with periodic boundaries
- no pulsar boundary
- absorbing boundary still exists in the solver, but the setup and executable
  are clearly meant for a periodic box

Important parameters:

- `problem = 0`
- `size`
- `N`
- `periodic_boundary`
- `clean_ep`
- `check_egb`

Parameter caveat:

- `user_alfven.hpp` uses `env.params().kn`, but `parse_config()` does not
  currently read `kn` from `config.toml`, so the code falls back to the default
  `kn = [1, 1, 0]` from `sim_params`.

### `coffee-tearing`

Entrypoint:
[src/main-tearing.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-tearing.cpp)

Build condition:
only when `use_cuda=ON`

Application:

- current-sheet tearing problem

Solver:

- currently `field_solver_EZ` because `EZ` is defined in the source
- the non-EZ alternative in the source is `field_solver`

Initialization include:

- `user_tearing.hpp`

Files:
[src/user_tearing.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_tearing.hpp)

Initial conditions:

- initializes `E = 0`
- initializes `P = 0`
- in EZ mode sets `E` and `B` to unstaggered `(0b111)`
- initializes a Harris-sheet-like magnetic configuration with a sinusoidal
  perturbation
- initializes `B0` as the unperturbed sheet field

Boundary conditions:

- no executable-level `problem` check is enforced in `main-tearing.cpp`
- runtime boundary behavior still comes from `field_solver_EZ`
- the solver's final-stage absorbing boundary still applies
- pulsar or Alfven-specific boundaries would only activate if `problem` were
  set to `1` or `2`

Important parameters:

- `dt`
- `max_steps`
- `data_interval`
- grid geometry through `N`, `guard`, `lower`, `size`, `nodes`
- EZ controls such as `clean_ep`, `check_egb`, `KOeps`, `divB_clean`

Code caveat:

- the tearing profile parameters are hardcoded in
  [src/user_tearing.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_tearing.hpp):
  `B0 = 1`, `a = 0.1`, `k = pi`, `amp = 1e-4`
- these values are not currently driven by `config.toml`

### `coffee-2d-cylindrical`

Entrypoint:
[src/main-pulsar2d-cylindrical.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-pulsar2d-cylindrical.cpp)

Build condition:
only when `use_cuda=ON`

Application:

- axisymmetric cylindrical pulsar

Solver:

- `field_solver_EZ_cylindrical`

Initialization include:

- `user_pulsar.hpp`

Files:
[src/user_pulsar.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_pulsar.hpp)

Initial conditions:

- sets all components to unstaggered `(0b111)`
- initializes `B_R` and `B_z` from a dipole field scaled by
  `b0 * radius^3`
- sets azimuthal magnetic field to zero
- initializes `E = 0`
- initializes `B0 = 0`

Boundary conditions:

- axis boundary
- pulsar boundary
- absorbing outer boundary

Important parameters:

- `radius`
- `b0`
- `omega`
- `clean_ep`
- `check_egb`
- cylindrical domain definition through `lower`, `size`, `N`

### `coffee-2d-spherical`

Entrypoint:
[src/main-pulsar2d-spherical.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-pulsar2d-spherical.cpp)

Build condition:
only when `use_cuda=ON`

Application:

- axisymmetric spherical pulsar

Solver:

- `field_solver_EZ_spherical`

Initialization include:

- `user_pulsar_sph.hpp`

Files:
[src/user_pulsar_sph.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_pulsar_sph.hpp)

Initial conditions:

- sets all field components to unstaggered `(0b111)`
- initializes `B` with a spherical dipole profile corrected by metric factors
- initializes `E = 0`
- initializes `B0 = 0`
- sets `P = 0`

Boundary conditions:

- axis boundary
- pulsar boundary
- absorbing outer boundary

Important parameters:

- `radius`
- `b0`
- `omega`
- `clean_ep`
- `check_egb`
- spherical domain layout through `lower`, `size`, `N`

### `coffee-3d`

Entrypoint:
[src/main-pulsar3d.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-pulsar3d.cpp)

Build condition:
always built

Application:

- 3D pulsar / multipole magnetosphere

Runtime check:

- requires `problem = 1`

Solver:

- currently `field_solver_EZ` because `EZ` is defined in the source
- the non-EZ alternative in the source is `field_solver`

Initialization include:

- `user_pulsar3d_EZ.hpp`

Alternative non-EZ include present in source:

- `user_pulsar3d.hpp`

Files:
[src/user_pulsar3d_EZ.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_pulsar3d_EZ.hpp)
[src/user_pulsar3d.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_pulsar3d.hpp)

Initial conditions:

- initializes `B` from `quadru_dipole(...)`
- uses dipole and quadrupole coefficients stored in `sim_params`
- EZ version sets `E`, `B`, `B0` to unstaggered `(0b111)` and zeroes `P`
- non-EZ version uses default staggered field storage and zero `E`

Boundary conditions:

- pulsar boundary from `field_solver_EZ`
- absorbing outer boundary
- MPI halo exchange after every RK stage

Important parameters:

- `problem = 1`
- `b0`
- `radius`
- `omega`
- dipole direction: `p1`, `p2`, `p3`
- quadrupole amplitudes: `q11`, `q12`, `q13`, `q22`, `q23`
- quadrupole offsets: `q_offset_x`, `q_offset_y`, `q_offset_z`
- `clean_ep`
- `check_egb`
- `KOeps`
- `pml`, `pmllen`, `sigpml`

### `coffee-alfven3d`

Entrypoint:
[src/main-alfven3d.cpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/main-alfven3d.cpp)

Build condition:
always built

Application:

- Alfven-wave perturbation launched from a pulsar background

Runtime check:

- requires `problem = 2`

Solver:

- currently `field_solver_EZ`

Initialization include:

- `user_pulsar3d_EZ.hpp`

Initial conditions:

- same background magnetic initialization as `coffee-3d`
- perturbation itself is imposed by the solver boundary through
  `boundary_alfven(t)`

Boundary conditions:

- `field_solver_EZ::boundary_alfven(t)` because `problem == 2`
- absorbing outer boundary
- MPI halo exchange after every RK stage

Important parameters:

- `problem = 2`
- all 3D pulsar background parameters
- perturbation timing: `tp_start`, `tp_end`, `tp_start1`, `tp_end1`
- perturbation radial/angular extents:
  `rpert1`, `rpert2`, `thp1`, `thp2`,
  `rpert11`, `rpert21`, `thp11`, `thp21`
- perturbation amplitudes and shape:
  `dw0`, `dw1`, `nT`, `nT1`, `shear`, `shear1`,
  `ph1`, `ph2`, `dph`, `ph11`, `ph21`, `dph1`,
  `theta0`, `drpert`, `pert_type`
- `snapshot_interval`

Restart/output behavior:

- supports restart through multi-file snapshots
- can write 2D slices via the `slice_*` parameters

## Auxiliary Setup Headers

These setup headers exist but are not currently selected by the compiled
executables in `src/CMakeLists.txt`:

- [src/user_init.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_init.hpp)
  provides a generic plane-wave initialization template.
- [src/user_wald1.hpp](/Volumes/yajiey/Active/codes/CoffeeGPU/src/user_wald1.hpp)
  provides an alternative Wald setup using `field_solver_gr_EZ`.

## Practical Setup Advice

When choosing an executable:

- use `coffee-wave` for periodic-box testing
- use `coffee-2d-cylindrical` or `coffee-2d-spherical` for axisymmetric pulsar
  studies
- use `coffee-3d` for full 3D pulsar evolution
- use `coffee-alfven3d` for pulsar-driven Alfven perturbations
- use `coffee-gr-Yee`, `coffee-gr`, or `coffee` for GR/Wald runs

When building a `config.toml`, start from `config.toml.example`, then add:

- the correct `problem` value for the executable
- the geometry and decomposition (`N`, `guard`, `lower`, `size`, `nodes`)
- the problem-specific magnetic and perturbation parameters
- the cleaning and output cadence controls

If you want a stricter one-to-one runbook next, the natural follow-up is a
table of executable -> required parameters -> recommended example values.
