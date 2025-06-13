// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/hip/hip_utils.hip.cpp

#include <chrono>

#include "hip/hip_utils.hip.hpp"

void report_completion_time_callback(hipStream_t stream, hipError_t status, void* userData) {
    // This is a HIP callback: it may not call hip functions
    // It runs in a separate thread, so it may not write to iostreams
    auto& time = *static_cast<std::chrono::high_resolution_clock::time_point*>(userData);
    time = std::chrono::high_resolution_clock::now();
}

hipDeviceProp get_device_prop(const int device_id) {
    hipDeviceProp prop;
    hipGetDeviceProperties(&prop, device_id);
    return prop;
}
hipDeviceProp get_default_device_prop() {
    int device_id;
    hipGetDevice(&device_id);
    return get_device_prop(device_id);
}

// Template specializations for __half type
template <>
__host__ __device__ __half hip_max<__half>(__half a, __half b) {
    return __hmax(a, b);
}

template <>
__host__ __device__ __half hip_min<__half>(__half a, __half b) {
    return __hmin(a, b);
}

template <>
__host__ __device__ __half device_nan<__half>() {
    return __ushort_as_half(0x7e00);
}
