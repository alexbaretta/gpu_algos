// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/hip_utils.hpp

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "type_traits.hpp"

constexpr size_t NULL_FLAGS = 0;


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

hipDeviceProp get_device_prop(const int device_id);
hipDeviceProp get_default_device_prop();
