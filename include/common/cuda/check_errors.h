// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#define cuda_check_error(cuda_err, step_name) \
    do { \
        if (cuda_err != cudaSuccess) { \
            auto msg = cudaGetErrorString(cuda_err); \
            std::cerr << "[CUDA error " << step_name << "] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            exit(1); \
        } \
    } while (0)
