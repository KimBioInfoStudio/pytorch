#pragma once

#include <c10/xpu/XPUMacros.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

struct C10_XPU_API DeviceProp {
  // Returns the device name of this SYCL device.
  sycl::info::device::name::return_type device_name;

  // Returns the device type associated with the device.
  sycl::info::device::device_type::return_type device_type;

  // Returns the platform name.
  sycl::info::platform::name::return_type platform_name;

  // Returns the vendor of this SYCL device.
  sycl::info::device::vendor::return_type vendor;

  // Returns a backend-defined driver version as a std::string.
  sycl::info::device::driver_version::return_type driver_version;

  // Returns the SYCL version as a std::string in the form:
  // <major_version>.<minor_version>
  sycl::info::device::version::return_type version;

  // Returns true if the SYCL device is available. Otherwise, return false.
  sycl::info::device::is_available::return_type is_available;

  // Returns the maximum size in bytes of the arguments that can be passed to a
  // kernel.
  sycl::info::device::max_parameter_size::return_type max_param_size;

  // Returns the number of parallel compute units available to the device.
  sycl::info::device::max_compute_units::return_type max_compute_units;

  // Returns the maximum dimensions that specify the global and local work-item
  // IDs used by the data parallel execution model.
  sycl::info::device::max_work_item_dimensions::return_type max_work_item_dims;

  // Returns the maximum number of workitems that are permitted in a work-group
  // executing a kernel on a single compute unit.
  sycl::info::device::max_work_group_size::return_type max_work_group_size;

  // Returns the maximum number of subgroups in a work-group for any kernel
  // executed on the device.
  sycl::info::device::max_num_sub_groups::return_type max_num_sub_groups;

  // Returns a std::vector of size_t containing the set of sub-group sizes
  // supported by the device.
  sycl::info::device::sub_group_sizes::return_type sub_group_sizes;

  // Returns the maximum configured clock frequency of this SYCL device in MHz.
  sycl::info::device::max_clock_frequency::return_type max_clock_freq;

  // Returns the default compute device address space size specified as an
  // unsigned integer value in bits. Must return either 32 or 64.
  sycl::info::device::address_bits::return_type address_bits;

  // Returns the maximum size of memory object allocation in bytes.
  sycl::info::device::max_mem_alloc_size::return_type max_mem_alloc_size;

  // Returns the minimum value in bits of the largest supported SYCL built-in
  // data type if this SYCL device is not of device type
  // sycl::info::device_type::custom.
  sycl::info::device::mem_base_addr_align::return_type mem_base_addr_align;

  // Returns a std::vector of info::fp_config describing the half precision
  // floating-point capability of this SYCL device.
  sycl::info::device::half_fp_config::return_type half_fp_config;

  // Returns a std::vector of info::fp_config describing the single precision
  // floating-point capability of this SYCL device.
  sycl::info::device::single_fp_config::return_type single_fp_config;

  // Returns a std::vector of info::fp_config describing the double precision
  // floating-point capability of this SYCL device.
  sycl::info::device::double_fp_config::return_type double_fp_config;

  // Returns the size of global device memory in bytes.
  sycl::info::device::global_mem_size::return_type global_mem_size;

  // Returns the type of global memory cache supported.
  sycl::info::device::global_mem_cache_type::return_type global_mem_cache_type;

  // Returns the size of global memory cache in bytes.
  sycl::info::device::global_mem_cache_size::return_type global_mem_cache_size;

  // Returns the size of global memory cache line in bytes.
  sycl::info::device::global_mem_cache_line_size::return_type
      global_mem_cache_line_size;

  // Returns the type of local memory supported.
  sycl::info::device::local_mem_type::return_type local_mem_type;

  // Returns the size of local memory arena in bytes.
  sycl::info::device::local_mem_size::return_type local_mem_size;

  // Returns the maximum number of sub-devices that can be created when this
  // device is partitioned.
  sycl::info::device::partition_max_sub_devices::return_type max_sub_devices;

  // Returns the resolution of device timer in nanoseconds.
  sycl::info::device::profiling_timer_resolution::return_type
      profiling_resolution;

  // Returns the preferred native vector width size for built-in scalar types
  // that can be put into vectors.
  sycl::info::device::preferred_vector_width_char::return_type
      pref_vec_width_char;
  sycl::info::device::preferred_vector_width_short::return_type
      pref_vec_width_short;
  sycl::info::device::preferred_vector_width_int::return_type
      pref_vec_width_int;
  sycl::info::device::preferred_vector_width_long::return_type
      pref_vec_width_long;
  sycl::info::device::preferred_vector_width_float::return_type
      pref_vec_width_float;
  sycl::info::device::preferred_vector_width_double::return_type
      pref_vec_width_double;
  sycl::info::device::preferred_vector_width_half::return_type
      pref_vec_width_half;

  // Returns the native ISA vector width. The vector width is defined as the
  // number of scalar elements that can be stored in the vector.
  sycl::info::device::native_vector_width_char::return_type
      native_vec_width_char;
  sycl::info::device::native_vector_width_short::return_type
      native_vec_width_short;
  sycl::info::device::native_vector_width_int::return_type native_vec_width_int;
  sycl::info::device::native_vector_width_long::return_type
      native_vec_width_long;
  sycl::info::device::native_vector_width_float::return_type
      native_vec_width_float;
  sycl::info::device::native_vector_width_double::return_type
      native_vec_width_double;
  sycl::info::device::native_vector_width_half::return_type
      native_vec_width_half;

  // Returns the number of EUs associated with the Intel GPU.
  sycl::ext::intel::info::device::gpu_eu_count::return_type gpu_eu_count;

  // Returns the number of EUs in a subslice.
  sycl::ext::intel::info::device::gpu_eu_count_per_subslice::return_type
      gpu_eu_count_per_subslice;

  // Returns the simd width of EU of GPU.
  sycl::ext::intel::info::device::gpu_eu_simd_width::return_type
      gpu_eu_simd_width;

  // Returns the number of hardware threads per EU of GPU.
  sycl::ext::intel::info::device::gpu_hw_threads_per_eu::return_type
      gpu_hw_threads_per_eu;
};

} // namespace c10::xpu
