#include "common/cuda/cuda_utils.h"

void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData) {
    // This is a CUDA callback: it may not call cuda functions
    // It runs in a separate thread, so it may not write to iostreams
    auto& time = *static_cast<std::chrono::high_resolution_clock::time_point*>(userData);
    time = std::chrono::high_resolution_clock::now();
}
