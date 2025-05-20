// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>
#include "check_errors.h"

void report_completion_callback(cudaStream_t stream, cudaError_t status, void* userData);
