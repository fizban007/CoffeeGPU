#ifndef _FIELDS_H_
#define _FIELDS_H_

#include "data/typedefs.h"
#include "data/multi_array.h"
#include "data/stagger.h"
#include "data/grid.h"

namespace Coffee {

template <typename T>
class vector_field {
 public:
  typedef vector_field<T> self_type;
  typedef multi_array<T> array_type;

  vector_field();
  ~vector_field();
  vector_field(const Grid &grid);
  vector_field(const self_type &field);
  vector_field(self_type &&field);

  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  /// Core functions
  void initialize();
  template <typename Func>
  void initialize(int component, const Func &f);
  template <typename Func>
  void initialize(const Func &f);

  void assign(T value, int n);
  void assign(T value);
  void assign(const vector_field<T> &field, const T &q);
  void copy_from(const self_type &field);

  void resize(const Grid &grid);

  /// Arithmetic operations
  self_type &multiplyBy(T value);
  self_type &addBy(T value, int n);
  self_type &addBy(const vector_field<T> &field);
  self_type &addBy(const vector_field<T> &field, T q);
  self_type &subtractBy(T value, int n);
  self_type &subtractBy(const vector_field<T> &field);

  // Interpolate the field to cell center and store the result to
  // @result
  void interpolate_to_center(self_type &result);

  // Interpolate the field from cell center to the stagger position
  // according to m_stagger, and store the result to @result
  void interpolate_from_center(self_type &result, Scalar q = 1.0);

  /// Index operator
  T &operator()(int n, int x, int y = 0, int z = 0) {
    return m_data[n](x, y, z);
  }
  const T &operator()(int n, int x, int y = 0,
                              int z = 0) const {
    return m_data[n](x, y, z);
  }
  T &operator()(int n, const Index &idx) {
    return m_data[n](idx);
  }
  const T &operator()(int n, const Index &idx) const {
    return m_data[n](idx);
  }

  /// Accessor methods
  array_type &data(int n) { return m_data[n]; }
  const array_type &data(int n) const { return m_data[n]; }
  Stagger stagger(int n) const { return m_stagger[n]; }
  const Grid &grid() const { return *m_grid; }
  const Extent& extent() const { return m_grid->extent(); }

  void set_stagger(int n, Stagger stagger) { m_stagger[n] = stagger; }
  void set_stagger(Stagger stagger[]) {
    m_stagger = stagger;
  }

 private:
  const Grid *m_grid;

  multi_array<T> m_data[3];
  Stagger m_stagger[3];
};


}

#endif // _FIELDS_H_
