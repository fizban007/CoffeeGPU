#include "algorithms/interpolation.h"
#include "cuda/cuda_utility.h"
#include "data/multi_array.h"
#include "data/multi_array_impl.hpp"
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

template <typename T>
__global__ void
downsample_average(T* orig_data, float* dst_data, Extent orig_ext,
                   Extent dst_ext, Index offset, Stagger st, int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < dst_ext.x && j < dst_ext.y && k < dst_ext.z) {
    size_t orig_idx = i * d + offset.x +
                      (j * d + offset.y) * orig_ext.x +
                      (k * d + offset.z) * orig_ext.x * orig_ext.y;
    size_t dst_idx = i + j * dst_ext.x + k * dst_ext.x * dst_ext.y;

    for (int kk = 0; kk < min(d, orig_ext.z); kk++) {
      for (int jj = 0; jj < min(d, orig_ext.y); jj++) {
        for (int ii = 0; ii < d; ii++) {
          dst_data[dst_idx] += interpolate(
              orig_data,
              orig_idx + ii + (jj + kk * orig_ext.y) * orig_ext.x, st,
              Stagger(0b111), orig_ext.x, orig_ext.y);
        }
      }
    }
    if (orig_ext.z > d) dst_data[dst_idx] /= d;
    if (orig_ext.y > d) dst_data[dst_idx] /= d;
    if (orig_ext.x > d) dst_data[dst_idx] /= d;
    // dst_data[dst_idx] = orig_data[orig_idx];
  }
}

template <typename T>
__global__ void
downsample(T* orig_data, float* dst_data, Extent orig_ext,
           Extent dst_ext, Index offset, Stagger st, int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < dst_ext.x && j < dst_ext.y && k < dst_ext.z) {
    size_t orig_idx = i * d + offset.x +
                      (j * d + offset.y) * orig_ext.x +
                      (k * d + offset.z) * orig_ext.x * orig_ext.y;
    size_t dst_idx = i + j * dst_ext.x + k * dst_ext.x * dst_ext.y;

    dst_data[dst_idx] =
        interpolate(orig_data, orig_idx, st, Stagger(0b111), orig_ext.x,
                    orig_ext.y);

    // dst_data[dst_idx] = orig_data[orig_idx];
  }
}

template <typename T>
__global__ void
downsample2d(T* orig_data, float* dst_data, Extent orig_ext,
             Extent dst_ext, Index offset, Stagger st, int d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < dst_ext.x && j < dst_ext.y) {
    size_t orig_idx =
        i * d + offset.x + (j * d + offset.y) * orig_ext.x;
    size_t dst_idx = i + j * dst_ext.x;

    dst_data[dst_idx] =
        interpolate2d(orig_data, orig_idx, st, Stagger(0b111), orig_ext.x);

    // dst_data[dst_idx] = orig_data[orig_idx];
  }
}

}  // namespace Kernels

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
multi_array<T>::assign_dev(const T& value) {
  Kernels::assign_single_value<<<256, 512>>>(m_data_d, m_size, value);
  CudaCheckError();
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

template <typename T>
void
multi_array<T>::downsample(int d, multi_array<float>& array,
                           Index offset, Stagger stagger,
                           float* h_ptr) {
  auto& ext = array.extent();
  if (ext.y == 1 && ext.z == 1) {
    // Use 1D version which we did not implement
  } else if (ext.z == 1) {  // Use 2D version
    dim3 blockSize(32, 32);
    dim3 gridSize((ext.x + blockSize.x - 1) / blockSize.x,
                  (ext.y + blockSize.y - 1) / blockSize.y);
    Kernels::downsample2d<<<gridSize, blockSize>>>(
        m_data_d, array.dev_ptr(), m_extent, array.extent(), offset,
        stagger, d);
    CudaCheckError();
  } else {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize((ext.x + blockSize.x - 1) / blockSize.x,
                  (ext.y + blockSize.y - 1) / blockSize.y,
                  (ext.z + blockSize.z - 1) / blockSize.z);
    Kernels::downsample<<<gridSize, blockSize>>>(
        m_data_d, array.dev_ptr(), m_extent, array.extent(), offset,
        stagger, d);
    CudaCheckError();
  }
  // CudaSafeCall(cudaMemcpy(h_ptr, array.m_data_d,
  //                         array.size() * sizeof(T),
  //                         cudaMemcpyDeviceToHost));
  array.sync_to_host();
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
