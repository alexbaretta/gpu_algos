// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>
#include "check_errors.h"

constexpr unsigned int NULL_FLAGS = 0;

void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData);
