COFFEE
======

# How to compile

First clone the directory:

    git clone https://github.com/fizban007/CoffeeGPU
    
Now to build this, you need to have `cmake`, `cuda`, and `hdf5` installed. To
compile this on `tigergpu`, use the following module load:

    module load rh/devtoolset
    module load cudatoolkit
    module load openmpi/gcc/1.10.2
    module load hdf5/gcc
    
Now go into the cloned repo:

    cd CoffeeGPU
    mkdir build
    cd build
    cmake ..
    make
    
The default build type is `Debug`, which is bad for performance. For production
one should use `cmake .. -DCMAKE_BUILD_TYPE=Release`.

To run unit tests in the `tests` directory, run `make check` in the `build`
directory. Every time you make, all the tests are compiled but not run. You'll
need to manually run them using `make check`. The first unit test
`test_stagger.cpp` should be a good indication of how to write unit tests. For
further information please look at the official documentation at
[[https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md]].
    
