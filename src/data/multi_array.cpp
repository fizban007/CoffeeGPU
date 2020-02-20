#include "algorithms/interpolation.h"
#include "data/multi_array.h"
#include "data/multi_array_impl.hpp"
#include "utils/memory.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdlib>

namespace Coffee {

template <typename T>
void
multi_array<T>::alloc_mem(size_t size) {
  // m_data_h = new T[size];
  m_data_h = (T*)(aligned_malloc(size * sizeof(T), 64));
  // m_data_h = (T*)(std::aligned_alloc(64, size * sizeof(T)));
  // std::cout << "size of allocation is " << size * sizeof(T) << std::endl;
  // std::cout << "address is " << m_data_h << std::endl;
}

template <typename T>
void
multi_array<T>::free_mem() {
  if (m_data_h != nullptr) {
    // delete[] m_data_h;
    aligned_free(m_data_h);
    // free(m_data_h);
    m_data_h = nullptr;
  }
}

template <typename T>
void
multi_array<T>::copy_from(const self_type& other) {
  if (m_size != other.m_size) {
    throw std::range_error(
        "Trying to copy from a multi_array of different size!");
  }
  std::memcpy(m_data_h, other.m_data_h, m_size * sizeof(T));
}

template <typename T>
void
multi_array<T>::assign_dev(const T& value) {}

template <typename T>
void
multi_array<T>::sync_to_host() {}

template <typename T>
void
multi_array<T>::sync_to_device() {}

template <typename T>
void
multi_array<T>::downsample(int d, multi_array<float>& array,
                           Index offset, Stagger stagger) {
  auto& ext = array.extent();
  for (int k = 0; k < ext.z; k++) {
    for (int j = 0; j < ext.y; j++) {
      for (int i = 0; i < ext.x; i++) {
        size_t orig_idx = i * d + offset.x +
            (j * d + offset.y) * m_extent.x +
            (k * d + offset.z) * m_extent.x * m_extent.y;
        size_t dst_idx = i + j * ext.x + k * ext.x * ext.y;
        array[dst_idx] = interpolate(m_data_h, orig_idx, stagger,
                                     Stagger(0b111), m_extent.x,
                                     m_extent.y);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////
// Explicitly instantiate the classes we will use
/////////////////////////////////////////////////////////////////
template class multi_array<long long>;
template class multi_array<long>;
template class multi_array<int>;
template class multi_array<short>;
template class multi_array<char>;
template class multi_array<unsigned int>;
template class multi_array<unsigned long>;
template class multi_array<unsigned long long>;
template class multi_array<float>;
template class multi_array<double>;
template class multi_array<long double>;

}  // namespace Coffee
