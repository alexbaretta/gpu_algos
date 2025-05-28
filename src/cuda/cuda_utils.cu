// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/cuda/cuda_utils.cu

#include "cuda/cuda_utils.h"

void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData) {
    // This is a CUDA callback: it may not call cuda functions
    // It runs in a separate thread, so it may not write to iostreams
    auto& time = *static_cast<std::chrono::high_resolution_clock::time_point*>(userData);
    time = std::chrono::high_resolution_clock::now();
}

cudaDeviceProp get_device_prop(const int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop;
}
cudaDeviceProp get_default_device_prop() {
    int device_id;
    cudaGetDevice(&device_id);
    return get_device_prop(device_id);
}
