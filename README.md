COFFEE
======

# How to compile

First clone the directory:

    git clone https://github.com/fizban007/CoffeeGPU
    
Now to build this, you need to have `cmake`, `cuda`, and `hdf5` installed. To
compile this on `tigergpu`, use the following module load:

    module load rh/devtoolset
    module load cudatoolkit
    module load openmpi/cuda-9.0/gcc
    module load hdf5/gcc
    
Now go into the cloned repo:

    cd CoffeeGPU
    mkdir build
    cd build
    cmake ..
    make
    
The default build type is `Debug`, which is bad for performance. For production
one should use `cmake .. -DCMAKE_BUILD_TYPE=Release`.
    
This is currently a barebone repo. The only thing included is a `multi_array`
class that is intended to be the basic 3D array data structure. We'll expand
this along the way.
