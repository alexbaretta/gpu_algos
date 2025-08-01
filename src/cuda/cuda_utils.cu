/*
    Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// source path: src/cuda/cuda_utils.cu

#include <chrono>

#include "cuda/cuda_utils.cuh"

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

// Template specializations for __half type
template <>
__host__ __device__ __half cuda_max<__half>(__half a, __half b) {
    return __hmax(a, b);
}

template <>
__host__ __device__ __half cuda_min<__half>(__half a, __half b) {
    return __hmin(a, b);
}

template <>
__host__ __device__ __half device_nan<__half>() {
    return __ushort_as_half(0x7e00);
}
