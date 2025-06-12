// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/hip_utils.hpp

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "check_errors.hip.hpp"
#include "type_traits.hip.hpp"

constexpr size_t NULL_FLAGS = 0;
constexpr static long MAX_BLOCK_SIZE = 1024;
constexpr static int FULL_MASK = -1;

#define WARP_SIZE warpSize
#define MAX_N_WARPS (MAX_BLOCK_SIZE / WARP_SIZE)
#define LAST_LANE (WARP_SIZE - 1)

__host__
int get_warp_size() {
    hipDeviceProp_t props;
    hip_check_error(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties");
    return props.warpSize;
}
__host__
long compute_n_threads_per_block(const dim3 block_dim) {
    return long(block_dim.x) * long(block_dim.y) * long(block_dim.z);
}
__device__
long compute_n_threads_per_block() {
    return long(blockDim.x) * long(blockDim.y) * long(blockDim.z);
}
__host__ __device__
long compute_n_warps_per_block(const long n_threads_per_block, const int warp_size) {
    return (n_threads_per_block + warp_size - 1) / warp_size;
}
__host__
long compute_n_warps_per_block(const dim3 block_dim, const int warp_size = get_warp_size()) {
    return compute_n_warps_per_block(compute_n_threads_per_block(block_dim), warp_size);
}
__device__
long compute_n_warps_per_block(const long n_threads_per_block) {
    return (n_threads_per_block + warpSize - 1) / warpSize;
}
__device__
long compute_n_warps_per_block() {
    return (compute_n_threads_per_block() + warpSize - 1) / warpSize;
}

template <HIP_scalar Number>
__host__ __device__ Number hip_max(Number a, Number b) {
    return max(a, b);
}

// Template specialization declarations (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ __half hip_max<__half>(__half a, __half b);

template <HIP_scalar Number>
__host__ __device__ Number hip_min(Number a, Number b) {
    return min(a, b);
}

// Template specialization declarations (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ __half hip_min<__half>(__half a, __half b);

template <HIP_scalar Number>
__host__ __device__ Number hip_sum(Number a, Number b) {
    return a + b;
}

template <HIP_scalar Number>
__host__ __device__ Number hip_prod(Number a, Number b) {
    return a * b;
}

template <HIP_scalar Number>
__host__ __device__ Number device_nan() {
    static_assert(std::is_floating_point_v<Number>, "device_nan only supports floating point types");
    if constexpr (std::is_same_v<Number, float>) {
        return nanf(nullptr);
    } else if constexpr (std::is_same_v<Number, double>) {
        return nan(nullptr);
    }
    // __half specialization is defined in hip_utils.hip.cpp
}

// Template specialization declaration (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ __half device_nan<__half>();

void report_completion_time_callback(hipStream_t stream, hipError_t status, void* userData);

// Function objects for HIP operations - these work reliably in device code
template <typename Number_>
struct hip_max_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return max(a, b);
    }

    __host__ __device__ static Number identity() {
        if constexpr (std::is_floating_point_v<Number>) {
            return -INFINITY;
        } else {
            return Number(1) << (sizeof(Number) * 8 - 1);
        }
    }
};

template <>
struct hip_max_op<__half> {
    using Number = __half;
    __host__ __device__ static __half apply(__half a, __half b) {
        return __hmax(a, b);
    }

    __host__ __device__ static __half identity() {
        return __hmax(__hneg(0x7c00), __hneg(0x7c00));
    }
};

template <typename Number_>
struct hip_min_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return min(a, b);
    }

    __host__ __device__ static Number identity() {
        if constexpr (std::is_floating_point_v<Number>) {
            return INFINITY;
        } else {
            return ~(Number(1) << (sizeof(Number) * 8 - 1));
        }
    }
};

template <>
struct hip_min_op<__half> {
    using Number = __half;
    __host__ __device__ static __half apply(__half a, __half b) {
        return __hmin(a, b);
    }

    __host__ __device__ static __half identity() {
        return __hmin(__hneg(0x7c00), __hneg(0x7c00));
    }
};

template <typename Number_>
struct hip_sum_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return a + b;
    }

    __host__ __device__ static Number identity() {
        return Number(0);
    }
};

template <typename Number_>
struct hip_prod_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return a * b;
    }

    __host__ __device__ static Number identity() {
        return Number(1);
    }
};

hipDeviceProp_t get_device_prop(const int device_id);
hipDeviceProp_t get_default_device_prop();
