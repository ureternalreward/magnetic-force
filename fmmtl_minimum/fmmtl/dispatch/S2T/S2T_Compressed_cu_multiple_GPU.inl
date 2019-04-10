#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "fmmtl/config.hpp"

#include "fmmtl/dispatch/S2T/S2T_Blocked_CSR_cu_Multi_GPU.inl"
typedef unsigned int uint;

struct Data {
  unsigned num_sources;
  unsigned num_targets;
  unsigned num_blocks;
  unsigned max_body_per_box;
  unsigned num_sbox;
  Data(unsigned s, unsigned t, unsigned b, unsigned max_bpb,unsigned _sbox)
    : num_sources(s),
    num_targets(t),
    num_blocks(b),
    max_body_per_box(max_bpb),
    num_sbox(_sbox){
  }
};


template <typename T>
inline T* gpu_new(unsigned n) {
  return thrust::raw_pointer_cast(thrust::device_malloc<T>(n));
}

template <typename Container>
inline typename Container::value_type* gpu_copy(const Container& c) {
  typedef typename Container::value_type c_value;
  // Allocate
  thrust::device_ptr<c_value> dptr = thrust::device_malloc<c_value>(c.size());
  // Copy
  //thrust::uninitialized_copy(c.begin(), c.end(), dptr);
  thrust::copy(c.begin(), c.end(), dptr);
  // Return
  return thrust::raw_pointer_cast(dptr);
}
template <typename Container>
inline typename Container::value_type* gpu_copy(Container& c) {
  typedef typename Container::value_type c_value;
  // Allocate
  thrust::device_ptr<c_value> dptr = thrust::device_malloc<c_value>(c.size());
  // Copy
  //thrust::uninitialized_copy(c.begin(), c.end(), dptr);
  thrust::copy(c.begin(), c.end(), dptr);
  // Return
  return thrust::raw_pointer_cast(dptr);
}

template <typename T>
inline void gpu_free(T* p) {
  thrust::device_free(thrust::device_pointer_cast<void>(p));
}

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::S2T_Compressed()
  : data_(0) {
}

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::S2T_Compressed(
  std::vector<thrust::pair<unsigned, unsigned> >& target_ranges,
  std::vector<unsigned>& source_range_ptrs,
  std::vector<thrust::pair<unsigned, unsigned> >& source_ranges,
  const std::vector<source_type>& sources,
  const std::vector<target_type>& targets)
  :
  target_ranges_(gpu_copy(target_ranges)),
  source_range_ptrs_(gpu_copy(source_range_ptrs)),
  source_ranges_(gpu_copy(source_ranges)),
  sources_(gpu_copy(sources)),
  targets_(gpu_copy(targets)) {
  unsigned max_body_per_box = 0;
  unsigned min_body_per_box = 20000;
  for (int i = 0; i < target_ranges.size(); i++) {
    int nbody_in_this_box = target_ranges[i].second - target_ranges[i].first;
    if (nbody_in_this_box > max_body_per_box) {
      max_body_per_box = nbody_in_this_box;
    }
    if (nbody_in_this_box < min_body_per_box) {
      min_body_per_box = nbody_in_this_box;
    }
  }
  for (int i = 0; i < source_ranges.size(); i++) {
    int nbody_in_this_box = source_ranges[i].second - source_ranges[i].first;
    if (nbody_in_this_box > max_body_per_box) {
      max_body_per_box = nbody_in_this_box;
    }
    if (nbody_in_this_box < min_body_per_box) {
      min_body_per_box = nbody_in_this_box;
    }
  }
  data_ = new Data(sources.size(), targets.size(), target_ranges.size(), max_body_per_box, source_ranges.size());
}

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::~S2T_Compressed() {
  delete reinterpret_cast<Data*>(data_);
  gpu_free(target_ranges_);
  gpu_free(source_range_ptrs_);
  gpu_free(source_ranges_);
  gpu_free(sources_);
  gpu_free(targets_);
}

/** A functor that indexes an array as one type but returns another type */
template <typename T1, typename T2>
class tricky_cast {
  T1* a_;
public:
  __host__ __device__
    tricky_cast(T1* a) : a_(a) {}
  __host__ __device__
    T2 operator[](unsigned blockidx) const {
    return *((T2*)(a_ + blockidx));
  }
};

template <typename _Kernel_Ty_>
void S2T_Compressed<_Kernel_Ty_>::execute(
  const _Kernel_Ty_& K,
  const std::vector<charge_type>& charges,
  std::vector<result_type>& results) {
  typedef _Kernel_Ty_ kernel_type;
  typedef typename kernel_type::source_type source_type;
  typedef typename kernel_type::target_type target_type;
  typedef typename kernel_type::charge_type charge_type;
  typedef typename kernel_type::result_type result_type;
  Data* data = reinterpret_cast<Data*>(data_);

  //determine the number of available devices
  int n_devices = 0;
  cudaError_t cerror = cudaGetDeviceCount(&n_devices);
  if (omp_get_max_threads() < n_devices) {
    n_devices = omp_get_max_threads();
  }
  if (cerror != cudaSuccess) {
    std::cout << "failed to get device number" << std::endl;
  }
  int hosting_device = 0;
  cudaGetDevice(&hosting_device);

  //host result for each of the device
  //result_from_device[i] stores the result from ith device.
  std::vector<std::vector<result_type>> result_from_device(n_devices);

  for (int i = 0; i < n_devices; i++) {
    //initialize the result for each device.
    result_from_device[i].assign(results.size(), result_type{}-result_type{});
  }

  uint average_block_per_device = data->num_blocks / n_devices;
  uint block_size_last_device = data->num_blocks - (n_devices - 1)*average_block_per_device;
  
#pragma omp parallel num_threads(4)
  {
    uint at_device = omp_get_thread_num();

    if (at_device < n_devices) {
      cudaSetDevice(at_device);
      thrust::pair<unsigned, unsigned>* this_device_target_ranges_;
      cudaMalloc(&this_device_target_ranges_, sizeof(thrust::pair<unsigned, unsigned>)*(data->num_blocks));
      cudaMemcpyPeer(this_device_target_ranges_, at_device, 
                     target_ranges_, hosting_device, 
                     sizeof(thrust::pair<unsigned, unsigned>)*(data->num_blocks));

      

      unsigned* this_device_source_range_ptrs_;
      cudaMalloc(&this_device_source_range_ptrs_, sizeof(unsigned)*(data->num_blocks+1));
      cudaMemcpyPeer(this_device_source_range_ptrs_, at_device,
        source_range_ptrs_, hosting_device,
        sizeof(unsigned)*(data->num_blocks + 1));
      
      thrust::pair<unsigned, unsigned>* this_device_source_ranges_;
      cudaMalloc(&this_device_source_ranges_, sizeof(thrust::pair<unsigned, unsigned>)*(data->num_sbox));
      cudaMemcpyPeer(this_device_source_ranges_, at_device,
                     source_ranges_, hosting_device,
                     sizeof(thrust::pair<unsigned, unsigned>)*(data->num_sbox));
      
      source_type* this_device_sources_;
      cudaMalloc(&this_device_sources_, sizeof(source_type)*(data->num_sources));
      cudaMemcpyPeer(this_device_sources_, at_device,
                     sources_, hosting_device,
                     sizeof(source_type)*(data->num_sources));
      
      charge_type* this_device_d_charges = gpu_copy(charges);
      

      target_type* this_device_targets_;
      cudaMalloc(&this_device_targets_, sizeof(target_type)*(data->num_targets));
      
      cudaMemcpyPeer(this_device_targets_, at_device,
                     targets_, hosting_device,
                     sizeof(target_type)*(data->num_targets));

      
      result_type* this_device_d_results = gpu_copy(result_from_device[at_device]);
     
      //each device is incharge of a portion of target box
      uint target_box_portion_size = average_block_per_device;
      uint target_box_offset = at_device*average_block_per_device;
      if (at_device == n_devices - 1) {
        target_box_portion_size = block_size_last_device;
      }

      unsigned num_blocks = target_box_portion_size;
      //printf("Calculating %d targets on GPU%d\n",target_box_portion_size,at_device);
      typedef thrust::pair<unsigned, unsigned> upair;
      //source_ranges_ stores the body begin and end in a sourcebox.
      // Launch kernel <<<grid_size, block_size>>>
      if (num_blocks > 0) {
        blocked_p2p<384> << <num_blocks, data->max_body_per_box >> >(
          K,
          this_device_target_ranges_,
          tricky_cast<unsigned, upair>(this_device_source_range_ptrs_),
          this_device_source_ranges_,
          this_device_sources_,
          //thrust::raw_pointer_cast(d_charges.data()),
          this_device_d_charges,
          this_device_targets_,
          this_device_d_results,
          target_box_offset);
      }

      //thrust::raw_pointer_cast(d_results.data()));
      FMMTL_CUDA_CHECK;

      // Copy results back
      thrust::device_ptr<result_type> d_results_ptr = thrust::device_pointer_cast(this_device_d_results);
      thrust::copy(d_results_ptr, d_results_ptr + results.size(), (result_from_device[at_device]).begin());

      cudaFree(this_device_target_ranges_);
      cudaFree(this_device_source_range_ptrs_);
      cudaFree(this_device_source_ranges_);
      cudaFree(this_device_sources_);
      gpu_free(this_device_d_charges);
      cudaFree(this_device_targets_);
      gpu_free(this_device_d_results);
    }
  }//end omp

  cudaSetDevice(hosting_device);

  //sum the result
   for (int i = 1; i < n_devices; i++) {
    for (int j = 0; j < results.size(); j++) {
      result_from_device[0][j] += result_from_device[i][j];
    }
  }
  results = result_from_device[0];
}


/** A functor that maps blockidx -> (target_begin,target_end) */
template <unsigned BLOCKDIM>
class block_range {
  unsigned N_;
public:
  __host__ __device__
    block_range(unsigned N) : N_(N) {}
  __host__ __device__
    thrust::pair<unsigned, unsigned> operator[](unsigned blockidx) const {
    return thrust::make_pair(blockidx * BLOCKDIM,
                             min(blockidx * BLOCKDIM + BLOCKDIM, N_));
  }
};

/** A functor that returns a constant */
template <typename T>
class constant {
  T value_;
public:
  __host__ __device__
    constant(T value) : value_(value) {}
  __host__ __device__
    T operator[](unsigned) const {
    return value_;
  }
};

template <typename _Kernel_Ty_>
void
S2T_Compressed<_Kernel_Ty_>::execute(const _Kernel_Ty_& K,
                                     const std::vector<source_type>& s,
                                     const std::vector<charge_type>& c,
                                     const std::vector<target_type>& t,
                                     std::vector<result_type>& r) {
  typedef _Kernel_Ty_ kernel_type;
  typedef typename kernel_type::source_type source_type;
  typedef typename kernel_type::target_type target_type;
  typedef typename kernel_type::charge_type charge_type;
  typedef typename kernel_type::result_type result_type;

  source_type* d_sources = gpu_copy(s);
  charge_type* d_charges = gpu_copy(c);
  target_type* d_targets = gpu_copy(t);
  result_type* d_results = gpu_copy(r);

  // XXX: device_vector doesn't like our vector?
  //thrust::device_vector<source_type> d_sources(s);
  //thrust::device_vector<charge_type> d_charges(c);
  //thrust::device_vector<target_type> d_targets(t);
  //thrust::device_vector<result_type> d_results(r);

  const unsigned num_tpb = 256;
  const unsigned num_blocks = (t.size() + num_tpb - 1) / num_tpb;

#if defined(FMMTL_DEBUG)
  std::cout << "Launching GPU Kernel: (blocks, threads/block) = ("
    << num_blocks << ", " << num_tpb << ")" << std::endl;
#endif

  typedef thrust::pair<unsigned, unsigned> upair;

  // Launch kernel <<<grid_size, block_size>>>
  blocked_p2p<num_tpb> << <num_blocks, num_tpb >> >(
    K,
    block_range<num_tpb>(t.size()),
    constant<upair>(upair(0, 1)),
    constant<upair>(upair(0, s.size())),
    d_sources,
    d_charges,
    d_targets,
    d_results);
  //thrust::raw_pointer_cast(d_sources.data()),
  //thrust::raw_pointer_cast(d_charges.data()),
  //thrust::raw_pointer_cast(d_targets.data()),
  //thrust::raw_pointer_cast(d_results.data()));
  FMMTL_CUDA_CHECK;

  // Copy results back and assign
  thrust::device_ptr<result_type> d_results_ptr = thrust::device_pointer_cast(d_results);
  thrust::copy(d_results_ptr, d_results_ptr + r.size(), r.begin());

  gpu_free(d_sources);
  gpu_free(d_charges);
  gpu_free(d_targets);
  gpu_free(d_results);
}
