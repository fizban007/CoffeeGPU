#ifndef _STAGGER_H_
#define _STAGGER_H_

#include "cuda/cuda_control.h"

namespace Coffee {

class Stagger {
 private:
  unsigned char stagger;

 public:
  /// Default constructor, initialize the stagger in each direction to be 0.
  HOST_DEVICE Stagger() : stagger(0) {}

  /// Constructor using an unsigned char. The recommended way to use this is to do
  ///
  ///     Stagger st(0b001);
  ///
  /// This will initialize the lowest bit to 1, and upper bits to 0. This is
  /// means staggered in x, but not staggered in y and z directions.
  HOST_DEVICE Stagger(unsigned char s) : stagger(s){};

  /// Copy constructor, simply copy the stagger of the given input.
  HOST_DEVICE Stagger(const Stagger& s) : stagger(s.stagger) {}

  /// Assignment, copy the stagger of the input.
  HD_INLINE Stagger& operator=(const Stagger& s) {
    stagger = s.stagger;
    return *this;
  }

  /// Assignment, using an unsigned char. The recommended way to use this is to do
  ///
  ///     st = 0b001;
  ///
  /// This will initialize the lowest bit to 1, and upper bits to 0. This is
  /// means staggered in x, but not staggered in y and z directions.
  HD_INLINE Stagger& operator=(const unsigned char s) {
    stagger = s;
    return *this;
  }

  /// Subscript operator. Use this to take the stagger of a given direction. For example,
  ///
  ///     Stagger st(0b110);
  ///     assert(st[0] == 1);
  ///     assert(st[1] == 1);
  ///     assert(st[2] == 1);
  ///
  /// Since this is inlined and bit shifts are cheap, feel free to use this inside a kernel.
  HD_INLINE int operator[](int i) const { return (stagger >> i) & 1UL; }

  /// Set the given bit to true or false.
  HD_INLINE void set_bit(int bit, bool i) {
    unsigned long x = !!i;
    stagger ^= (-x ^ stagger) & (1UL << bit);
  }

  /// Flip the stagger of a given direcion.
  HD_INLINE void flip(int n) { stagger ^= (1UL << n); }

  /// Return the complement of this stagger configuration.
  HD_INLINE Stagger complement() { return Stagger(~stagger); }
};

}  // namespace Coffee

#endif  // _STAGGER_H_
