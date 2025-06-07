// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/random.hip.h

#pragma once

#include <vector>
#include <random>
#include <concepts>
#include <hip/hip_fp16.h>

// Random vector generation for HIP types
template <typename T>
void randomize_vector(std::vector<T>& data, int seed);

// Specialization for _Float16
void randomize_vector(std::vector<_Float16>& data, int seed);
