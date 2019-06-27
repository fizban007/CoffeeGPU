#include "cuda/cuda_utility.h"
#include "multi_array.h"
#include <algorithm>
#include <stdexcept>

namespace Coffee {

namespace Kernels {

template <typename T>
__global__ void
assign_single_value(T* data, size_t size, T value) {
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size;
       i += blockDim.x * gridDim.x) {
    data[i] = value;
  }
}

}  // namespace Kernels

template <typename T>
multi_array<T>::multi_array()
    : m_data_h(nullptr),
      m_data_d(nullptr),
      m_extent(0, 1, 1),
      m_size(0) {}

template <typename T>
multi_array<T>::multi_array(int width, int height, int depth)
    : m_extent(width, height, depth) {
  m_size = width * height * depth;

  alloc_mem(m_size);
  assign_dev(0);
}

template <typename T>
multi_array<T>::multi_array(const Extent& extent)
    : multi_array(extent.width(), extent.height(), extent.depth()) {}

template <typename T>
multi_array<T>::multi_array(const self_type& other)
    : multi_array(other.m_extent) {
  copy_from(other);
  sync_to_host();
}

template <typename T>
multi_array<T>::multi_array(self_type&& other) {
  m_extent = other.m_extent;
  m_size = other.m_size;
  m_data_h = other.m_data_h;
  m_data_d = other.m_data_d;

  other.m_data_h = nullptr;
  other.m_data_d = nullptr;
}

template <typename T>
multi_array<T>::~multi_array() {
  free_mem();
}

template <typename T>
multi_array<T>&
multi_array<T>::operator=(const self_type& other) {
  free_mem();
  m_size = other.m_size;
  m_extent = other.m_extent;

  alloc_mem(other.m_size);
  copy_from(other);
  sync_to_host();
  return *this;
}

template <typename T>
multi_array<T>&
multi_array<T>::operator=(self_type&& other) {
  m_data_h = other.m_data_h;
  m_data_d = other.m_data_d;

  other.m_data_h = nullptr;
  other.m_data_d = nullptr;

  m_extent = other.m_extent;
  m_size = other.m_size;

  return *this;
}

template <typename T>
void
multi_array<T>::alloc_mem(size_t size) {
  m_data_h = new T[size];

  CudaSafeCall(cudaMalloc(&m_data_d, sizeof(T) * size));
}

template <typename T>
void
multi_array<T>::free_mem() {
  if (m_data_h != nullptr) {
    delete[] m_data_h;
    m_data_h = nullptr;
  }
  if (m_data_d != nullptr) {
    CudaSafeCall(cudaFree(m_data_d));
    m_data_d = nullptr;
  }
}

template <typename T>
const T&
multi_array<T>::operator()(int x, int y, int z) const {
  size_t idx = x + (y + z * m_extent.height()) * m_extent.width();
  return m_data_h[idx];
}

template <typename T>
T&
multi_array<T>::operator()(int x, int y, int z) {
  size_t idx = x + (y + z * m_extent.height()) * m_extent.width();
  return m_data_h[idx];
}

template <typename T>
const T&
multi_array<T>::operator()(const Index& index) const {
  return operator()(index.x, index.y, index.z);
}

template <typename T>
T&
multi_array<T>::operator()(const Index& index) {
  return operator()(index.x, index.y, index.z);
}

template <typename T>
const T& multi_array<T>::operator[](size_t n) const {
  return m_data_h[n];
}

template <typename T>
T& multi_array<T>::operator[](size_t n) {
  return m_data_h[n];
}

template <typename T>
void
multi_array<T>::copy_from(const self_type& other) {
  if (m_size != other.m_size) {
    throw std::range_error(
        "Trying to copy from a multi_array of different size!");
  }
  // memcpy(m_data_h, other.m_data_h, m_size * sizeof(T));
  CudaSafeCall(cudaMemcpy(m_data_d, other.m_data_d, m_size * sizeof(T),
                          cudaMemcpyDeviceToDevice));
}

template <typename T>
void
multi_array<T>::assign(const T& value) {
  std::fill_n(m_data_h, m_size, value);
}

template <typename T>
void
multi_array<T>::assign_dev(const T& value) {
  Kernels::assign_single_value<<<256, 512>>>(m_data_d, m_size, value);
  CudaCheckError();
}

template <typename T>
void
multi_array<T>::resize(int width, int height, int depth) {
  size_t size = width * height * depth;
  m_extent = Extent(width, height, depth);
  m_size = size;

  // Do nothing if the sizes already match
  if (m_size == size) {
    return;
  }
  free_mem();
  alloc_mem(size);
}

template <typename T>
void
multi_array<T>::resize(Extent extent) {
  resize(extent.width(), extent.height(), extent.depth());
}

template <typename T>
void
multi_array<T>::sync_to_host() {
  CudaSafeCall(cudaMemcpy(m_data_h, m_data_d, m_size * sizeof(T),
                          cudaMemcpyDeviceToHost));
}

template <typename T>
void
multi_array<T>::sync_to_device() {
  CudaSafeCall(cudaMemcpy(m_data_d, m_data_h, m_size * sizeof(T),
                          cudaMemcpyHostToDevice));
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
