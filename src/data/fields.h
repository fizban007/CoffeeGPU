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

  // Constructors
  vector_field();
  ~vector_field();
  vector_field(const Grid &grid);
  vector_field(const self_type &field);
  vector_field(self_type &&field);

  // Copy assignment operator
  self_type &operator=(const self_type &field);

  // Move assignment operator
  self_type &operator=(self_type &&field);

  /// Initialize all values to zero
  void initialize();

  /// Initialize a given component according to function f. f should accept 3
  /// parameters x, y, z, and return a single scalar.
  template <typename Func>
  void initialize(int component, const Func &f);

  /// Initialize all components according to function f. f should accept 4
  /// parameters, n, x, y, z, where n is the component. f should again return a
  /// single value.
  template <typename Func>
  void initialize(const Func &f);

  /// Assign a single value to component n at all points
  void assign(T value, int n);

  /// Assign a single value to all components at all points
  void assign(T value);

  /// Assign the value of another field to this one, multiplied by factor q
  void assign(const self_type &field, const T &q);

  /// Copy from the other field
  void copy_from(const self_type &field);

  /// Resize this field according to the given grid (and point the grid pointer
  /// to this new grid)
  void resize(const Grid &grid);

  /// Arithmetic operations
  self_type &multiplyBy(T value);
  self_type &addBy(T value, int n);
  self_type &addBy(const self_type &field);
  self_type &addBy(const self_type &field, T q);
  self_type &subtractBy(T value, int n);
  self_type &subtractBy(const self_type &field);

  // Interpolate the field to cell center and store the result to
  // @result
  void interpolate_to_center(self_type &result);

  // Interpolate the field from cell center to the stagger position
  // according to m_stagger, and store the result to @result
  void interpolate_from_center(self_type &result, Scalar q = 1.0);

  /// Index using three integers, and a component
  T &operator()(int n, int i, int j = 0, int k = 0) {
    return m_data[n](i, j, k);
  }
  /// Index using three integers, and a component, const version
  const T &operator()(int n, int i, int j = 0,
                              int k = 0) const {
    return m_data[n](i, j, k);
  }

  /// Index using an Index object, and a component
  T &operator()(int n, const Index &idx) {
    return m_data[n](idx);
  }

  /// Index using an Index object, and a component, const version
  const T &operator()(int n, const Index &idx) const {
    return m_data[n](idx);
  }

  void sync_to_host();
  void sync_to_host(int n);
  void sync_to_device();
  void sync_to_device(int n);

  /// Accessor methods
  array_type &data(int n) { return m_data[n]; }
  const array_type &data(int n) const { return m_data[n]; }
  Stagger stagger(int n) const { return m_stagger[n]; }
  const Grid &grid() const { return *m_grid; }
  Extent extent() const { return m_grid->extent(); }

  void set_stagger(int n, Stagger stagger) { m_stagger[n] = stagger; }
  void set_stagger(Stagger stagger[]) {
    for (int i = 0; i < 3; i++)
      m_stagger[i] = stagger[i];
  }

 private:
  void set_default_stagger();
  void check_component_range(int n) const;

  const Grid *m_grid;

  multi_array<T> m_data[3];
  Stagger m_stagger[3];
};


}

#endif // _FIELDS_H_
