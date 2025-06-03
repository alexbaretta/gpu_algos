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

constexpr size_t NULL_FLAGS = 0;


template <typename Number>
__device__ inline Number cuda_max(Number a, Number b) {
    return max(a, b);
}

template <>
__device__ inline __half cuda_max<__half>(__half a, __half b) {
    return __hmax(a, b);
}


void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData);

cudaDeviceProp get_device_prop(const int device_id);
cudaDeviceProp get_default_device_prop();
