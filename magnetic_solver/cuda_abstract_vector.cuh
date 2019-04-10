#pragma once
#ifndef _CUDA_ABSTRACT_VECTOR_
#define _CUDA_ABSTRACT_VECTOR_
#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "vector_types.h"
#include <vector>
namespace  Libo {

template<typename T>
__global__ void cuda_add_to_element_kernel(T* devx, T* devy, T* devz, const T x,const  T y,const T z, size_t id, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i != id || i>= size) {
    return;
  }
  devx[i] += x;
  devy[i] += y;
  devz[i] += z;
}

template<typename T, typename other_T>
__global__ void cuda_conversion_kernel(other_T * out, const  T * in, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    auto blub = static_cast<other_T>(in[i]);
    out[i] = blub;
  }
}

//template<>
//__global__ void cuda_conversion_kernel(float * out, double * in, size_t size) {
//  int i = blockDim.x*blockIdx.x + threadIdx.x;
//  if (i < size) {
//    out[i] = double2float_rd(in[i]);
//  }
//}

template<typename T>
__global__ void cuda_add(T* a, const T * b, const T * c, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = b[i] + c[i];
  }
}

template<typename T>
__global__ void cuda_subtract(T* a, const T * b, const T * c, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = b[i] - c[i];
  }
}

template<typename T>
__global__ void cuda_negate(T * a, const T * b, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = -b[i];
  }
}

template<typename T>
__global__ void cuda_inplace_multiply(T* a, const T c, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = a[i] * c;
  }
}

template<typename T>
__global__ void cuda_multiply(T* a, const T * b, const T c, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = b[i] * c;
  }
}

template<typename T>
__global__ void cuda_pointwise_multiply(T* a, const T * b, const T * c, size_t size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < size) {
    a[i] = b[i] * c[i];
  }
}


template<typename T>
__global__ void square_reduction_kernel(T* g_idata, T* g_odata, size_t size) {
  const size_t blockSize = 512;
  __shared__ T sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * 2 * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;
  while (i < size) {
    sdata[tid] += g_idata[i] * g_idata[i];
    if (i + blockSize < size) {
      sdata[tid] += g_idata[i + blockSize] * g_idata[i + blockSize];
    }
    i += gridSize;
  }
  __syncthreads();

  if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
  if (tid < 32)
  {
    sdata[tid] += sdata[tid + 32];__syncthreads();
    sdata[tid] += sdata[tid + 16];__syncthreads();
    sdata[tid] += sdata[tid + 8];__syncthreads();
    sdata[tid] += sdata[tid + 4];__syncthreads();
    sdata[tid] += sdata[tid + 2];__syncthreads();
    sdata[tid] += sdata[tid + 1];__syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] += sdata[0];
  }
}

template<typename T>
__global__ void normal_reduction_kernel(T* g_idata, T* g_odata, size_t size) {
  const size_t blockSize = 512;
  __shared__ T sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * 2 * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;
  while (i < size) {
    sdata[tid] += g_idata[i];
    if (i + blockSize < size) {
      sdata[tid] += g_idata[i + blockSize];
    }
    i += gridSize;
  }
  __syncthreads();

  if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
  if (tid < 32)
  {
    sdata[tid] += sdata[tid + 32];__syncthreads();
    sdata[tid] += sdata[tid + 16];__syncthreads();
    sdata[tid] += sdata[tid + 8];__syncthreads();
    sdata[tid] += sdata[tid + 4];__syncthreads();
    sdata[tid] += sdata[tid + 2];__syncthreads();
    sdata[tid] += sdata[tid + 1];__syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] += sdata[0];
  }
}




template<typename T>
class cuda_abstract_vector {
public:
  cuda_abstract_vector() { np = 0; cudaStatus = cudaGetLastError();}
  cuda_abstract_vector(size_t size) {
    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_z, size * sizeof(T));
	cudaStatus = cudaGetLastError();
    np = size;
  }
  cuda_abstract_vector(const cuda_abstract_vector<T>& old) {
    cudaStatus = cudaMalloc((void**)&dev_x, old.np * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_y, old.np * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_z, old.np * sizeof(T));
    cudaMemcpy(this->dev_x, old.dev_x, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->dev_y, old.dev_y, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->dev_z, old.dev_z, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
    np = old.np;
  }

  ~cuda_abstract_vector() {
    if (np != 0) {
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_z);
    }
  }

  cuda_abstract_vector<T> & operator=(const cuda_abstract_vector<T> & old);
  cuda_abstract_vector<T> operator+(const cuda_abstract_vector<T> &other) const;
  cuda_abstract_vector<T> operator-(const cuda_abstract_vector<T>& other) const;
  cuda_abstract_vector<T> operator-() const;
  cuda_abstract_vector<T> & operator+=(const cuda_abstract_vector<T> &other);
  cuda_abstract_vector<T> & operator-=(const cuda_abstract_vector<T> &other);
  cuda_abstract_vector<T> & operator*=(const T C);
  template<typename other_T> operator cuda_abstract_vector<other_T>() const;
  const cuda_abstract_vector<T> operator*(const T C) const;
  void import_data(const T* host_x, const T* host_y, const T* host_z);
  void import_data(const std::vector<T> & host_x, const std::vector<T> & host_y, const std::vector<T> & host_z);
  void import_data(const std::vector<float3> & host_xyz);
  void export_data(T* host_x, T* host_y, T* host_z) const;
  void export_data(std::vector<T> &host_x, std::vector<T> &host_y, std::vector<T> &host_z) const;
  void add_to_element(size_t id, T x, T y, T z);
  void clear() {
    if (np!=0) {
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_z);
      np = 0;
    }
  }
  void reserve(size_t _np) {
    if (_np == np) {
      return;
    }
    else {
      clear();
      np = _np;
      cudaStatus = cudaMalloc((void**)&dev_x, _np * sizeof(T));
      cudaStatus = cudaMalloc((void**)&dev_y, _np * sizeof(T));
      cudaStatus = cudaMalloc((void**)&dev_z, _np * sizeof(T));
    }
  }


  T norm2();
  T norm() { return sqrt(norm2()); };
  T dot(cuda_abstract_vector<T>);

  typedef T value_type;

  cudaError_t cudaStatus;
  size_t np;
  T * dev_x;
  T * dev_y;
  T * dev_z;
};


template<typename T>
cuda_abstract_vector<T> & cuda_abstract_vector<T>::operator=(const cuda_abstract_vector<T> & old)
{
  if (this->np != old.np) {
    if (this->np!=0){
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_z);
    }
    
    this->np = old.np;
    cudaMalloc(&dev_x, np * sizeof(T));
    cudaMalloc(&dev_y, np * sizeof(T));
    cudaMalloc(&dev_z, np * sizeof(T));
  }

  cudaMemcpy(this->dev_x, old.dev_x, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(this->dev_y, old.dev_y, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(this->dev_z, old.dev_z, old.np * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaStatus = cudaGetLastError();
  return *this;
}

template<typename T>
cuda_abstract_vector<T> cuda_abstract_vector<T>::operator+(const cuda_abstract_vector<T> &other) const
{
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }

  cuda_add << <blockdim, threaddim >> > (result.dev_x, this->dev_x, other.dev_x, np);
  cuda_add << <blockdim, threaddim >> > (result.dev_y, this->dev_y, other.dev_y, np);
  cuda_add << <blockdim, threaddim >> > (result.dev_z, this->dev_z, other.dev_z, np);
//  getLastCudaError("cuda_abstract_vector<T>::operator+");
  return result;
}

template<typename T>
cuda_abstract_vector<T> cuda_abstract_vector<T>::operator-(const cuda_abstract_vector<T> &other) const
{
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_subtract << <blockdim, threaddim >> > (result.dev_x, this->dev_x, other.dev_x, np);
  cuda_subtract << <blockdim, threaddim >> > (result.dev_y, this->dev_y, other.dev_y, np);
  cuda_subtract << <blockdim, threaddim >> > (result.dev_z, this->dev_z, other.dev_z, np);
  return result;
}

template<typename T>
inline cuda_abstract_vector<T> cuda_abstract_vector<T>::operator-(void) const
{
  cuda_abstract_vector<T> result{ np };

  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }

  cuda_negate << <blockdim, threaddim >> > (result.dev_x, this->dev_x, np);
  cuda_negate << <blockdim, threaddim >> > (result.dev_y, this->dev_y, np);
  cuda_negate << <blockdim, threaddim >> > (result.dev_z, this->dev_z, np);
  return result;
}

template<typename T>
cuda_abstract_vector<T>& cuda_abstract_vector<T>::operator+=(const cuda_abstract_vector<T>& other)
{
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_add << <blockdim, threaddim >> > (result.dev_x, this->dev_x, other.dev_x, np);
  cuda_add << <blockdim, threaddim >> > (result.dev_y, this->dev_y, other.dev_y, np);
  cuda_add << <blockdim, threaddim >> > (result.dev_z, this->dev_z, other.dev_z, np);
  *this = result;
  return *this;
}

template<typename T>
cuda_abstract_vector<T>& cuda_abstract_vector<T>::operator-=(const cuda_abstract_vector<T>& other)
{
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_subtract << <blockdim, threaddim >> > (result.dev_x, this->dev_x, other.dev_x, np);
  cuda_subtract << <blockdim, threaddim >> > (result.dev_y, this->dev_y, other.dev_y, np);
  cuda_subtract << <blockdim, threaddim >> > (result.dev_z, this->dev_z, other.dev_z, np);
  *this = result;
  return *this;
}

template<typename T>
cuda_abstract_vector<T>& cuda_abstract_vector<T>::operator*=(const T C)
{
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_inplace_multiply << <blockdim, threaddim >> > (this->dev_x, C, np);
  cuda_inplace_multiply << <blockdim, threaddim >> > (this->dev_y, C, np);
  cuda_inplace_multiply << <blockdim, threaddim >> > (this->dev_z, C, np);
  return *this;
}

//conversion from int to double, double to int etc.
template<typename T>
template<typename other_T>
cuda_abstract_vector<T>::operator cuda_abstract_vector<other_T>() const
{
  cuda_abstract_vector<other_T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_conversion_kernel << <blockdim, threaddim >> > (result.dev_x, dev_x, np);
  cuda_conversion_kernel << <blockdim, threaddim >> > (result.dev_y, dev_y, np);
  cuda_conversion_kernel << <blockdim, threaddim >> > (result.dev_z, dev_z, np);
  return result;
}

template<typename T>
const cuda_abstract_vector<T> cuda_abstract_vector<T>::operator*(const T C) const
{
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_multiply << <blockdim, threaddim >> > (result.dev_x, this->dev_x, C, np);
  cuda_multiply << <blockdim, threaddim >> > (result.dev_y, this->dev_y, C, np);
  cuda_multiply << <blockdim, threaddim >> > (result.dev_z, this->dev_z, C, np);
  return result;
}

template<typename T>
const cuda_abstract_vector<T> operator*(const T & C, const cuda_abstract_vector<T> & in)
{
  size_t np = in.np;
  cuda_abstract_vector<T> result{ np };
  size_t threaddim = 512;
  int blockdim = in.np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_multiply << <blockdim, threaddim >> > (result.dev_x, in.dev_x, C, np);
  cuda_multiply << <blockdim, threaddim >> > (result.dev_y, in.dev_y, C, np);
  cuda_multiply << <blockdim, threaddim >> > (result.dev_z, in.dev_z, C, np);
  return result;
}

template<typename T>
void cuda_abstract_vector<T>::import_data(const T * host_x, const T * host_y, const T * host_z)
{
  cudaMemcpy(this->dev_x, host_x, np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(this->dev_y, host_y, np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(this->dev_z, host_z, np * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void cuda_abstract_vector<T>::import_data(const std::vector<T>& host_x, const std::vector<T>& host_y, const std::vector<T>& host_z)
{
  np = host_x.size();
  reserve(np);
  cudaMemcpy(this->dev_x, host_x.data(), np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(this->dev_y, host_y.data(), np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(this->dev_z, host_z.data(), np * sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T>
void cuda_abstract_vector<T>::import_data(const std::vector<float3>& host_xyz)
{
  if (np != host_xyz.size()) {
    if (np != 0) {
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_z);
    }
    dev_x = 0;
    dev_y = 0;
    dev_z = 0;
    np = host_xyz.size();
    cudaMalloc(&dev_x, sizeof(T)*np);
    cudaMalloc(&dev_y, sizeof(T)*np);
    cudaMalloc(&dev_z, sizeof(T)*np);
  }
  T * hostx, *hosty, *hostz;
  hostx = new T[np];
  hosty = new T[np];
  hostz = new T[np];
  for (int i = 0; i < host_xyz.size(); i++) {
    hostx[i] = host_xyz[i].x;
    hosty[i] = host_xyz[i].y;
    hostz[i] = host_xyz[i].z;
  }
  cudaMemcpy(dev_x, hostx, sizeof(T)*np, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, hosty, sizeof(T)*np, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_z, hostz, sizeof(T)*np, cudaMemcpyHostToDevice);
  delete[] hostx;
  delete[] hosty;
  delete[] hostz;
}


template<typename T>
void cuda_abstract_vector<T>::export_data(T * host_x, T * host_y, T * host_z) const
{
  cudaMemcpy(host_x, this->dev_x, np * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_y, this->dev_y, np * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_z, this->dev_z, np * sizeof(T), cudaMemcpyDeviceToHost);
}
template<typename T>
void cuda_abstract_vector<T>::export_data(std::vector<T> &host_x, std::vector<T> &host_y, std::vector<T> &host_z) const {
  T * host_hfx = new T[np];
  T * host_hfy = new T[np];
  T * host_hfz = new T[np];
  export_data(host_hfx, host_hfy, host_hfz);
  for (int i = 0;i < np;i++) {
    host_x[i] = host_hfx[i];
    host_y[i] = host_hfy[i];
    host_z[i] = host_hfz[i];
  }
  delete[] host_hfx;
  delete[] host_hfy;
  delete[] host_hfz;
}

template<typename T>
void cuda_abstract_vector<T>::add_to_element(size_t id, T x, T y, T z)
{
  if (id >= np) {
    return;
  }

  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_add_to_element_kernel<<<blockdim, threaddim>>>(dev_x, dev_y, dev_z, x, y, z, id, np);
}

template<typename T>
T cuda_abstract_vector<T>::norm2()
{
  int blocksize = 512;
  const int num_blocks = 64;
  T * dev_odata = 0;
  T host_odata[num_blocks] = { 0 };
  cudaMalloc((void **)&dev_odata, num_blocks * sizeof(T));
  cudaMemcpy(dev_odata, host_odata, sizeof(T)*num_blocks, cudaMemcpyHostToDevice);
  square_reduction_kernel << <num_blocks, blocksize >> > (dev_x, dev_odata, np);
  square_reduction_kernel << <num_blocks, blocksize >> > (dev_y, dev_odata, np);
  square_reduction_kernel << <num_blocks, blocksize >> > (dev_z, dev_odata, np);
  cudaMemcpy(host_odata, dev_odata, sizeof(T)*num_blocks, cudaMemcpyDeviceToHost);
  T total_sum = 0;
  for (int i = 0;i < num_blocks;i++) {
    total_sum += host_odata[i];
  }
  cudaFree(dev_odata);
  return total_sum;
}

template<typename T>
T cuda_abstract_vector<T>::dot(cuda_abstract_vector<T> other)
{
  int blocksize = 512;
  const int num_blocks = 64;
  T * dev_odata = 0;
  T * dev_tempx = 0;
  T * dev_tempy = 0;
  T * dev_tempz = 0;
  T host_odata[num_blocks] = { 0 };
  cudaMalloc((void **)&dev_odata, num_blocks * sizeof(T));
  cudaMemcpy(dev_odata, host_odata, sizeof(T)*num_blocks, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&dev_tempx, other.np * sizeof(T));
  cudaMalloc((void **)&dev_tempy, other.np * sizeof(T));
  cudaMalloc((void **)&dev_tempz, other.np * sizeof(T));

  size_t threaddim = 512;
  size_t blockdim = np / threaddim;
  if (np % threaddim != 0) {
    blockdim++;
  }
  cuda_pointwise_multiply << <blockdim, threaddim >> > (dev_tempx, this->dev_x, other.dev_x, np);
  cuda_pointwise_multiply << <blockdim, threaddim >> > (dev_tempy, this->dev_y, other.dev_y, np);
  cuda_pointwise_multiply << <blockdim, threaddim >> > (dev_tempz, this->dev_z, other.dev_z, np);

  normal_reduction_kernel << <num_blocks, blocksize >> > (dev_tempx, dev_odata, np);
  normal_reduction_kernel << <num_blocks, blocksize >> > (dev_tempy, dev_odata, np);
  normal_reduction_kernel << <num_blocks, blocksize >> > (dev_tempz, dev_odata, np);

  cudaMemcpy(host_odata, dev_odata, sizeof(T)*num_blocks, cudaMemcpyDeviceToHost);
  T total_sum = 0;
  for (int i = 0;i < num_blocks;i++) {
    total_sum += host_odata[i];
  }
  cudaFree(dev_odata);
  cudaFree(dev_tempx);
  cudaFree(dev_tempy);
  cudaFree(dev_tempz);
  return total_sum;
}

}

#endif