// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: src/hip/hip_utils.hip

#include <hip/hip_runtime.h>
#include <chrono>
#include <random>
#include "hip/hip_utils.hiph"

void report_completion_time_callback(hipStream_t stream, hipError_t status, void* userData) {
    auto* time_point = static_cast<std::chrono::high_resolution_clock::time_point*>(userData);
    *time_point = std::chrono::high_resolution_clock::now();
}

hipDeviceProp_t get_device_prop(const int device_id) {
    hipDeviceProp_t prop;
    hip_check_error(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
    return prop;
}

hipDeviceProp_t get_default_device_prop() {
    return get_device_prop(0);
}

// Random vector generation for HIP types
template <typename T>
void randomize_container(std::vector<T>& data, int seed) {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>, "T must be floating point or integral");

    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dis(T(-1.0), T(1.0));
        for (auto& element : data) {
            element = dis(gen);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dis(T(0), T(100));
        for (auto& element : data) {
            element = dis(gen);
        }
    }
}

// Explicit instantiations for common types
template void randomize_container<float>(std::vector<float>& data, int seed);
template void randomize_container<double>(std::vector<double>& data, int seed);
template void randomize_container<int8_t>(std::vector<int8_t>& data, int seed);
template void randomize_container<int16_t>(std::vector<int16_t>& data, int seed);
template void randomize_container<int32_t>(std::vector<int32_t>& data, int seed);
template void randomize_container<int64_t>(std::vector<int64_t>& data, int seed);
template void randomize_container<uint8_t>(std::vector<uint8_t>& data, int seed);
template void randomize_container<uint16_t>(std::vector<uint16_t>& data, int seed);
template void randomize_container<uint32_t>(std::vector<uint32_t>& data, int seed);
template void randomize_container<uint64_t>(std::vector<uint64_t>& data, int seed);

// Specialization for __half
void randomize_container(std::vector<__half>& data, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& element : data) {
        element = _Float16(dis(gen));
    }
}
