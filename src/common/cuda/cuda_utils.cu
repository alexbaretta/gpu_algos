#include "common/cuda/cuda_utils.h"

void report_completion_callback(cudaStream_t stream, cudaError_t status, void* userData) {
    auto& [event, prev_event, start_event, step_name] = *static_cast<std::tuple<cudaEvent_t, cudaEvent_t, cudaEvent_t, std::string>*>(userData);

    cuda_check_error(status, step_name);

    float step_time = 0.0f, total_time = 0.0f;
    cuda_check_error(cudaEventElapsedTime(&step_time, prev_event, event), step_name);
    cuda_check_error(cudaEventElapsedTime(&total_time, start_event, event), step_name);
    std::cout << " - " << step_name << ": " << step_time << " ms (" << total_time << " ms total)" << std::endl;
}
