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


// source path: src/hip/hip_utils.cu

#include <chrono>
#include "hip/check_errors.hiph"
#include "hip/hip_utils.hiph"

void report_completion_time_callback(hipStream_t stream, hipError_t status, void* userData) {
    // This is a CUDA callback: it may not call hip functions
    // It runs in a separate thread, so it may not write to iostreams
    auto& time = *static_cast<std::chrono::high_resolution_clock::time_point*>(userData);
    time = std::chrono::high_resolution_clock::now();
}

hipDeviceProp_t get_device_prop(const int device_id) {
    hipDeviceProp_t prop;
    hip_check_error(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
    return prop;
}
hipDeviceProp_t get_default_device_prop() {
    int device_id;
    hip_check_error(hipGetDevice(&device_id), "hipGetDevice");
    return get_device_prop(device_id);
}
