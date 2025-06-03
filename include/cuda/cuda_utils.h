// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/cuda_utils.h

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>
#include "check_errors.h"
#include "type_traits.h"

constexpr size_t NULL_FLAGS = 0;


template <CUDA_scalar Number>
__host__ __device__ Number cuda_max(Number a, Number b) {
    return max(a, b);
}

// Template specialization declarations (defined in cuda_utils.cu)
template <>
__host__ __device__ __half cuda_max<__half>(__half a, __half b);

template <CUDA_scalar Number>
__host__ __device__ Number cuda_min(Number a, Number b) {
    return min(a, b);
}

// Template specialization declarations (defined in cuda_utils.cu)
template <>
__host__ __device__ __half cuda_min<__half>(__half a, __half b);

template <CUDA_scalar Number>
__host__ __device__ Number cuda_sum(Number a, Number b) {
    return a + b;
}

template <CUDA_scalar Number>
__host__ __device__ Number cuda_prod(Number a, Number b) {
    return a * b;
}


void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData);

cudaDeviceProp get_device_prop(const int device_id);
cudaDeviceProp get_default_device_prop();
