#define M_PI    3.141592653589793
#define MPREC   4
#define one_    REAL(1, MPREC)
#define zero_   REAL(0, MPREC)


!     prototype code for a parallel program. Array mx x my x mz is
!     decomposed in slabs along x direction, so that slabs are mx x
!     my / sizey x mz / (size0 / sizey) where size0 is the total number of
!     processors involved, sizey -- number of blocks in y direction.

!     Array has 2 cells in the beginning of every dimension and 3 at the end
!     allocated for guard cells. Correspondingly, effective size of the array is!    (mx - 5) * (my - 5) * (mz - 5). Each array in the subblock has similar 5 guard
!      cells. This can be changed for different guard cell layout.
!    The quantity (my - 5) should be divisible by sizey.
!    In order to be able to dynamically change the number of processors, one
!     has to be able to dynamically allocate arrays, hence I use fortran 90.
!     It is possible to do something like this with f77, but it's more
!     convoluted.
