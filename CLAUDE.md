# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoffeeGPU (COmputational Force-FreE Electrodynamics) is a C++17/CUDA framework for force-free electrodynamics simulations. It uses MPI domain decomposition, GPU acceleration (with CPU fallback), and HDF5 parallel I/O.

## Build Commands

```bash
mkdir build && cd build
cmake .. -Duse_cuda=1 -Duse_double=0    # GPU + single precision
cmake .. -Duse_cuda=0                     # CPU-only
cmake .. -DCMAKE_BUILD_TYPE=Release       # production (default is Debug)
make
```

**Run tests:**
```bash
make check    # from build directory
```

**Dependencies:** CMake (≥2.8.10), MPI, HDF5 (parallel preferred), CUDA (optional).
Vendored deps in `deps/`: Catch2, cpptoml, cxxopt, vectorclass.

Executables land in `build/bin/`, library in `lib/libCoffee.a`.

## Architecture

### Data Layer (`src/data/`)
- **`multi_array<T>`** — Contiguous 1D/2D/3D array with Fortran-style indexing (`array(i,j,k)` → `i + j*width + k*width*height`). Explicit host/device memory with `sync_to_device()` / `sync_to_host()`.
- **`vector_field<T>`** — Bundle of 3 `multi_array`s with per-component stagger metadata. E fields are edge-centered, B fields are face-centered by convention.
- **`sim_data`** — All simulation state: `E`, `B`, `B0`, scalar field `P`, divergence trackers, update accumulators.
- **`Grid`** — Coordinate metadata (dims, guard cells, spacing, bounds, MPI offsets). Copied to CUDA constant memory at startup.
- **`Scalar`** type alias — `float` by default, `double` with `-Duse_double=1`.

### Simulation Management (`src/`)
- **`sim_environment`** — MPI init, 3D Cartesian domain decomposition, guard cell exchanges, device selection.
- **`sim_params`** — Runtime parameters read from `config.toml` (see `config.toml.example`). Also copied to device constant memory.

### Field Solvers (`src/algorithms/`)
Multiple solver variants using 3-stage SSP Runge-Kutta time integration:
- **`field_solver`** — Base 3D Cartesian Yee solver
- **`field_solver_EZ`** — Enhanced with Dedner divergence cleaning + Kreiss-Oliger dissipation
- **`field_solver_EZ_cylindrical`** / **`field_solver_EZ_spherical`** — 2D coordinate variants
- **`field_solver_gr`** / **`field_solver_gr_EZ`** — General relativistic (Kerr-Schild) solvers

All solvers enforce the force-free constraint E²≤B².

### CUDA Integration (`src/cuda/`)
- **`constant_mem.h`** — `dev_grid` and `dev_params` in CUDA constant memory
- **`cuda_control.h`** — `HOST_DEVICE` macros for host/device portability
- Kernels use grid-stride loops; see README.md for kernel writing patterns

### Executables (`src/main*.cpp`)
Each `main-*.cpp` targets a different physics problem: `coffee-3d` (pulsar 3D), `coffee-2d-cylindrical`, `coffee-2d-spherical`, `coffee-alfven3d`, `coffee-gr`, `coffee-tearing`, `coffee-wave`, etc. The default `main.cpp` uses the GR EZ solver.

### Problem Setup (`src/user_*.hpp`)
Include files defining initial/boundary conditions per problem type (e.g., `user_pulsar.hpp`, `user_alfven.hpp`, `user_wald.hpp`, `user_tearing.hpp`).

### I/O (`src/utils/`)
- **`data_exporter`** — HDF5 output with downsampling, 2D slicing, and restart snapshots
- **`hdf_wrapper`** — HDF5 interface layer

## Testing

Uses Catch2 (vendored). Tests are in `tests/`. Run with `make check`. Key test files cover stagger operations, multi_array, interpolation, HDF5 I/O, MPI communication, and parity checks.

## Code Style

Configured via `.clang-format` (Google-based, 72-column, 2-space indent). Language server config in `.ccls`.
