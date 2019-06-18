COmputational Force-FreE Electrodynamics (COFFEE)
======

# How to compile

First clone the directory:

    git clone https://github.com/fizban007/CoffeeGPU
    
Now to build this, you need to have `cmake`, `cuda`, `mpi`, and `hdf5` installed. 

## Compilation on tigressdata or tigergpu

To compile this on `tigergpu` or `tigressdata`, use the following module load:

    module load rh/devtoolset
    module load cudatoolkit
    module load openmpi/gcc/1.10.2
    module load hdf5/gcc
    
Then go into the cloned repo and run this:

    cd CoffeeGPU
    mkdir build
    cd build
    cmake3 ..
    make

## Compilation on Ascent
    
When on `Ascent`, use the following module load:

    module load cmake/3.14.2
    module load gcc/8.1.1
    module load cuda/10.1.105
    module load spectrum-mpi/10.3.0.0-20190419
    module load hdf5/1.10.3
    
Now go into the cloned repo:

    cd CoffeeGPU
    mkdir build
    cd build
    CC="gcc" CXX="g++" cmake ..
    make
    
The default build type is `Debug`, which is bad for performance. For production
one should use `cmake .. -DCMAKE_BUILD_TYPE=Release`. The `CC="gcc" CXX="g++"`
part seems to be needed on `Ascent` for mysterious reasons. It is not needed on
`tigressdata` or `tigergpu` or on my laptop.

To run unit tests in the `tests` directory, run `make check` in the `build`
directory. Every time you make, all the tests are compiled but not run. You'll
need to manually run them using `make check`. The first unit test
`test_stagger.cpp` should be a good indication of how to write unit tests. For
further information please look at the official documentation at
<https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md>.
   
# How the code is structured

At the lowest level we have a `multi_array` class that abstracts away the memory
management from the user (but provides a way to access the underlying raw
pointers). The `multi_array` always allocates a 1D chunk of memory and maps any
3D indices onto the linear memory using a Fortran-like indexing, i.e. `array(i,
j, k)` returns the element `idx = i + j * width + k * width * height`.
Communication between the device and host is done manually using
`sync_to_device()` and `sync_to_host()` functions.

A `Grid` object keeps track of the underlying grid structuring, including the
dimensions of the grid, sizes in numerical units, number of guard cells, etc.
This object is copied to the device constant memory at the start of the program.

`sim_params` holds all of the runtime parameters for the simulation. It is read
from `config.toml` at initialization, and copied to device constant memory as
well.

The `vector_field` class is basically a bundle of 3 `multi_array`s, with a
stagger structure, and has a pointer to the `Grid` object it lives on. Every
component of the `vector_field` has a `stagger`. By default all fields are
edge-centered, meaning that `E[i]` will be staggered in all directions except
`i`. The default convention is that `E` field will be edge-centered and `B`
field will be face-centered. A vector field can be initialized with a lambda
function as follows:

    vector_field<double> f(grid);
    f.initialize(0, [](double x, double y, double z){
        return x / sqrt(x * x + y * y + z * z);
    });
    // Now f[0] will be initialized to x/r over the entire domain

The `sim_data` class is a bundle of all the `vector_field`s involved in the
simulation. The data output routine should take the `sim_data` and serialize it
to HDF5.

The `sim_environment` class handles most of the initialization, including domain
decomposition and managing the parameters. All the other modules in the code
should be initialized using only `sim_environment` only, as it contains all
contextual information of any core algorithms.

The modules to be worked on will be the core field update module, the MPI
communication module, and the HDF5 output module. The `main.cpp` file will
initialize these modules sequentially, while including appropriate user files
that define initial and boundary conditions.
