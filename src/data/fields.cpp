#include "fields.h"

namespace Coffee {

template <typename T>
vector_field<T>::vector_field() : m_grid(nullptr) {
  for (int i = 0; i < 3; i++) {
    // Default stagger is edge-centered
    m_stagger[i] = Stagger(0b000);
    m_stagger[i].set_bit((i + 1) % 3, true);
    m_stagger[i].set_bit((i + 2) % 3, true);
  }
}

template <typename T>
vector_field<T>::~vector_field() {}


}
