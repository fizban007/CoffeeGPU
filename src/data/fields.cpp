#include "fields.h"

namespace Coffee {

template <typename T>
vector_field<T>::vector_field() : m_grid(nullptr) {
  set_default_stagger();
}

template <typename T>
vector_field<T>::vector_field(const Grid& grid) : m_grid(&grid) {
  set_default_stagger();
}

template <typename T>
vector_field<T>::vector_field(const self_type& field)
    : m_grid(field.m_grid) {
  for (int i = 0; i < 3; i++) {
    m_stagger[i] = field.m_stagger[i];
    m_data[i] = field.m_data[i];
  }
}

template <typename T>
vector_field<T>::vector_field(self_type&& field)
    : m_grid(field.m_grid) {
  for (int i = 0; i < 3; i++) {
    m_stagger[i] = field.m_stagger[i];
    m_data[i] = std::move(field.m_data[i]);
  }
}

template <typename T>
vector_field<T>::~vector_field() {}

template <typename T>
void
vector_field<T>::initialize() {
  assign(0.0);
}

template <typename T>
void
vector_field<T>::assign(T value) {
  for (int i = 0; i < 3; i++) {
    m_data[i].assign(value);
  }
}

template <typename T>
void
vector_field<T>::assign(T value, int n) {
  check_component_range(n);
  m_data[n].assign(value);
}

template <typename T>
void
vector_field<T>::copy_from(const self_type &field) {
  for (int i = 0; i < 3; i++) {
    m_data[i].copy_from(field.m_data[i]);
  }
}

template <typename T>
void
vector_field<T>::resize(const Grid &grid) {
  for (int i = 0; i < 3; i++) {
    m_data[i].resize(grid.extent());
  }
  m_grid = &grid;
}

template <typename T>
void
vector_field<T>::sync_to_host(int n) {
  check_component_range(n);
  m_data[n].sync_to_host();
}

template <typename T>
void
vector_field<T>::sync_to_host() {
  for (int i = 0; i < 3; i++) {
    m_data[i].sync_to_host();
  }
}

template <typename T>
void
vector_field<T>::sync_to_device(int n) {
  check_component_range(n);
  m_data[n].sync_to_device();
}

template <typename T>
void
vector_field<T>::sync_to_device() {
  for (int i = 0; i < 3; i++) {
    m_data[i].sync_to_device();
  }
}

template <typename T>
void
vector_field<T>::set_default_stagger() {
  for (int i = 0; i < 3; i++) {
    // Default stagger is edge-centered
    m_stagger[i] = Stagger(0b000);
    m_stagger[i].set_bit((i + 1) % 3, true);
    m_stagger[i].set_bit((i + 2) % 3, true);
  }
}

template <typename T>
void
vector_field<T>::check_component_range(int n) const {
  if (n < 0 || n >= 3) {
    throw std::out_of_range(
        "Trying to assign to a non-existent field component!");
  }
}

/////////////////////////////////////////////////////////////////
// Explicitly instantiate the classes we will use
/////////////////////////////////////////////////////////////////
template class vector_field<float>;
template class vector_field<double>;

}  // namespace Coffee
