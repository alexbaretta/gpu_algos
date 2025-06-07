// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/hip_utils.hip.h

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>
#include "hip/check_errors.hip.h"
#include "hip/type_traits.hip.h"

constexpr size_t NULL_FLAGS = 0;


template <HIP_scalar Number>
__host__ __device__ Number hip_max(Number a, Number b) {
    return max(a, b);
}

// Template specialization declarations (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ _Float16 hip_max<_Float16>(_Float16 a, _Float16 b);

template <HIP_scalar Number>
__host__ __device__ Number hip_min(Number a, Number b) {
    return min(a, b);
}

// Template specialization declarations (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ _Float16 hip_min<_Float16>(_Float16 a, _Float16 b);

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
    // _Float16 specialization is defined in hip_utils.hip.cpp
}

// Template specialization declaration (defined in hip_utils.hip.cpp)
template <>
__host__ __device__ _Float16 device_nan<_Float16>();

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
struct hip_max_op<_Float16> {
    using Number = _Float16;
    __host__ __device__ static _Float16 apply(_Float16 a, _Float16 b) {
        return _Float16(fmaxf(float(a), float(b)));  // Use explicit cast
    }

    __host__ __device__ static _Float16 identity() {
        return _Float16(-INFINITY);
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
struct hip_min_op<_Float16> {
    using Number = _Float16;
    __host__ __device__ static _Float16 apply(_Float16 a, _Float16 b) {
        return _Float16(fminf(float(a), float(b)));  // Use explicit cast
    }

    __host__ __device__ static _Float16 identity() {
        return _Float16(INFINITY);
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
