// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/cuda_utils.h

#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>
#include "check_errors.h"

constexpr size_t NULL_FLAGS = 0;

void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData);

cudaDeviceProp get_device_prop(const int device_id);
cudaDeviceProp get_default_device_prop();
