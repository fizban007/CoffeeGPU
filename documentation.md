# CoffeeGPU Code Documentation

This document describes the codebase structure, coding conventions, and the common development workflow for CoffeeGPU.

## 1. Project Overview

CoffeeGPU is a C++/CUDA/MPI codebase for force-free electrodynamics simulations.

Core characteristics:
- MPI domain decomposition over a 3D Cartesian communicator.
- Optional CUDA backend (`use_cuda=ON`) with CPU fallback implementations.
- HDF5-based parallel output and restart snapshots.
- Multiple executable entrypoints for different physics setups.

## 2. Repository Structure

Top-level directories and files:
- `src/`: main simulation code (algorithms, data structures, CUDA integration, executables).
- `tests/`: Catch2 unit/integration tests and CUDA test targets.
- `deps/`: vendored third-party dependencies (`cpptoml`, `cxxopts`, `catch`, `vectorclass`).
- `original/`: original Fortran reference code.
- `python/`: post-processing and analysis scripts.
- `config.toml.example`: template runtime configuration.
- `CMakeLists.txt`: top-level build configuration.

`src/` module map:
- `src/data/`: core containers and domain objects.
  - `multi_array`: host/device array abstraction.
  - `vector_field`: 3-component field wrapper with stagger metadata.
  - `sim_data`: aggregate simulation state (`E`, `B`, `B0`, scalar fields).
  - `grid.h`, `stagger.h`, `vec3.h`, `typedefs.h`.
- `src/algorithms/`: field update schemes and physics kernels.
  - CPU and CUDA variants of solvers and boundary operators.
  - Solvers include `field_solver`, `field_solver_EZ`,
    `field_solver_EZ_cylindrical`, `field_solver_EZ_spherical`,
    `field_solver_gr`, and `field_solver_gr_EZ`.
- `src/cuda/`: CUDA support code.
  - constant memory interfaces, CUDA utility macros, CUDA-specific `sim_environment` parts.
- `src/utils/`: output, timing, and memory helpers.
  - `data_exporter` and `hdf_wrapper` are central for diagnostics/output.
- `src/main*.cpp`: executable entrypoints for specific problems.
- `src/user_*.hpp`: problem initialization and boundary presets included by entrypoints.

## 3. Build and Target Layout

### 3.1 Top-Level Build (`CMakeLists.txt`)

Important options:
- `use_cuda` (default `ON`): enables CUDA language and CUDA source paths when a CUDA compiler is detected.
- `build_tests` (default `ON`): includes `tests/` targets.
- `use_double`: toggles scalar precision (`float` default, `double` when set).

Required packages:
- MPI
- HDF5 (parallel preferred)

CUDA mode behavior:
- CUDA is enabled only if `use_cuda` is true and `CMAKE_CUDA_COMPILER` is found.
- Defines `CUDA_ENABLED` and configures CUDA flags/architectures.

### 3.2 Library + Executables (`src/CMakeLists.txt`)

`Coffee` library is assembled from a core set plus either:
- CUDA path (`*.cu`, CUDA `sim_env`, CUDA-aware array operations), or
- CPU path (`multi_array.cpp`, `sim_env.cpp`, CPU solver/boundary implementations).

Executables:
- Always built: `coffee-3d`, `coffee-alfven3d`.
- Built under `use_cuda`: `coffee`, `coffee-2d-cylindrical`, `coffee-2d-spherical`, `coffee-gr`, `coffee-gr-Yee`, `coffee-wave`, `coffee-tearing`.

## 4. Core Data and Runtime Model

### 4.1 Scalar Type Convention

In `src/data/typedefs.h`:
- `Scalar` is `float` by default.
- If `USE_DOUBLE` is defined, `Scalar` becomes `double`.

Most numerics are templated or written against `Scalar`.

### 4.2 `multi_array<T>`

`multi_array` is the base storage abstraction:
- Logical 1D/2D/3D extent.
- Host pointer + device pointer.
- Indexing maps 3D indices to contiguous linear memory.
- Explicit host/device sync methods:
  - `sync_to_device()`
  - `sync_to_host()`

This explicit sync model is important: compute and communication code assumes the caller keeps host/device views coherent.

### 4.3 `vector_field<T>`

`vector_field` wraps three `multi_array<T>` components and stores per-component staggering.

Typical usage:
- Field components `0,1,2` correspond to x/y/z components.
- Default stagger is edge-centered; `sim_data` sets E and B stagers explicitly for solver conventions.

### 4.4 `sim_data`

`sim_data` aggregates runtime state:
- Primary fields: `E`, `B`, `B0`.
- Auxiliary scalars: `P`, divergence diagnostics, dissipation diagnostics.
- Bulk sync helpers for all fields (`sync_to_host/device`).

### 4.5 `sim_environment`

`sim_environment` owns global runtime context:
- MPI initialization/finalization.
- Config parsing (`config.toml`).
- Grid setup and decomposition.
- Cartesian communicator setup and neighbor discovery.
- Guard-cell exchange API (`send_guard_cells`).

There are CPU and CUDA implementations for communication internals:
- CPU path exchanges host buffers and MPI datatypes.
- CUDA path stages halos with `cudaMemcpy3D` and then performs MPI send/recv.

## 5. Solver and Workflow Architecture

Solver interfaces in `src/algorithms/`:
- `field_solver`
- `field_solver_EZ`
- `field_solver_EZ_cylindrical`
- `field_solver_EZ_spherical`
- `field_solver_gr`
- `field_solver_gr_EZ`

Common pattern:
1. Construct with references to `sim_data` and `sim_environment`.
2. Call `evolve_fields(time)` each timestep.
3. Optional diagnostics like `total_energy`.

Boundary and cleaning operations are solver responsibilities, usually including:
- physical boundary conditions (`boundary_*`)
- force-free cleaning (`clean_epar`, `check_eGTb`)
- numerical dissipation (`Kreiss_Oliger` in EZ/GR-EZ paths)

## 6. Entry Point Workflow (`main*.cpp`)

Most executables follow this sequence:
1. `sim_environment env(&argc, &argv);`
2. Validate runtime config (`problem` checks in some binaries).
3. Build `sim_data data(env);`
4. Instantiate a solver variant.
5. Include a `user_*.hpp` setup file for initial conditions/boundaries.
6. Construct `data_exporter`.
7. Run timestep loop:
   - periodic output (`data_interval`)
   - `solver.evolve_fields(time)`
   - update time by `dt`

Notable convention:
- Problem initialization often happens through `#include "user_*.hpp"` directly inside `main`, so profile switching is done by changing include lines and/or executable choice.

## 7. Configuration and I/O Workflow

### 7.1 Runtime Config

`sim_params` is filled by `parse_config("config.toml")`:
- timestep/output settings (`dt`, `max_steps`, `data_interval`, ...)
- geometry/grid/decomposition (`N`, `guard`, `size`, `lower`, `nodes`)
- boundary and cleaning options
- problem-specific parameters (pulsar/alfven/GR)

Start from `config.toml.example`, then create `config.toml` in run directory.

### 7.2 Output and Restart

`data_exporter` handles:
- periodic field output
- optional sliced outputs
- snapshots (`save_snapshot` / `load_snapshot`)

Data format:
- HDF5 via `H5File` wrapper in `src/utils/hdf_wrapper.*`.
- Parallel HDF5 writes for distributed arrays.

## 8. Testing Workflow

`tests/CMakeLists.txt` defines:
- `tests` (Catch2 CPU tests)
- `tests_cuda` (CUDA tests, only when CUDA compiler available)
- utility binaries (`test_mpi`, `test_mpi2`, `test_output`)
- aggregate target: `check`

Typical flow:
1. configure/build project with tests enabled.
2. run `check` to execute registered CPU/CUDA tests.
3. run specific test binaries for MPI/output scenarios as needed.

## 9. Coding Conventions

Inferred from repository config and existing code:
- Language/style:
  - C++17 for C++ files.
  - CUDA files use C++14 mode in tooling config.
- Formatting:
  - `.editorconfig`: spaces, size 2, LF, UTF-8, trim trailing whitespace.
  - `.clang-format`: 2-space indentation, mostly attached braces, 72-column limit.
- Namespaces and naming:
  - Project code is under `namespace Coffee`.
  - Types/classes: snake_case for many classes (`sim_environment`, `sim_data`).
  - Methods/variables: mixed snake_case, physics abbreviations preserved.
- Header guards:
  - Traditional `_FILE_H_` style guards are consistently used.
- CPU/CUDA portability:
  - `cuda/cuda_control.h` defines `HOST_DEVICE`/`HD_INLINE` macros for dual-compilation helpers.
- Data movement:
  - Host/device transfer is explicit and deliberate.

## 10. Developer Workflow (Recommended)

For safe and consistent changes:
1. Choose the executable/problem path first (`main*.cpp` + `user_*.hpp`).
2. Update/extend config keys in `sim_params.h/.cpp` if needed.
3. Implement algorithm changes in both CPU and CUDA paths where applicable.
4. Keep guard-cell exchange assumptions consistent with grid/extent conventions.
5. Validate with:
   - unit tests (`check`)
   - a short smoke run with reduced `max_steps`
   - output/restart sanity checks (`Data/` HDF5 content)
6. For new physics setups:
   - add/modify a `user_*.hpp` initialization block,
   - optionally add a new `main-*.cpp` target,
   - wire it in `src/CMakeLists.txt`.

## 11. Practical Notes and Caveats

- The code relies heavily on compile-time selection (`use_cuda`, include-switching of user setup files).
- Entry points are similar but not fully unified; executable-specific checks (`problem` IDs) are part of runtime correctness.
- MPI + HDF5 environment consistency is critical; build/toolchain mismatches are a common failure mode.
- `config.toml` is expected in current working directory at runtime.

---

If needed, this document can be split into:
- `docs/architecture.md`
- `docs/developer-workflow.md`
- `docs/problem-setups.md`
for easier long-term maintenance.
