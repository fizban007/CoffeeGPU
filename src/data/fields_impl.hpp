#ifndef _FIELDS_IMPL_H_
#define _FIELDS_IMPL_H_

#include "fields.h"

namespace Coffee {

template <typename T>
template <typename Func>
void
vector_field<T>::initialize(int component, const Func& f) {
  check_component_range(component);
  // This way vector field is always defined in the center of the cell
  // face, staggered in the direction of the component
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = m_grid->pos(2, k, m_stagger[component][2]);
    size_t k_offset =
        k * m_data[component].width() * m_data[component].height();
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = m_grid->pos(1, j, m_stagger[component][1]);
      size_t j_offset = j * m_data[component].width();
#pragma omp simd
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = m_grid->pos(0, i, m_stagger[component][0]);
        m_data[component][i + j_offset + k_offset] = f(x1, x2, x3);
      }
    }
  }
  sync_to_device(component);
}

template <typename T>
template <typename Func>
void
vector_field<T>::initialize(const Func& f) {
  initialize(0, [&f](T x1, T x2, T x3) { return f(0, x1, x2, x3); });
  initialize(1, [&f](T x1, T x2, T x3) { return f(1, x1, x2, x3); });
  initialize(2, [&f](T x1, T x2, T x3) { return f(2, x1, x2, x3); });
}

}  // namespace Coffee

#endif  // _FIELDS_IMPL_H_
