// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/random.hip.hpp

#pragma once

#include <hip/hip_fp16.h>
#include "common/random.hpp"

inline void randomize_vector(
    std::vector<__half>& data,
    int seed
) {
    constexpr float min = 0.0;
    constexpr float max = 1.0;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(min, max);

    for (auto& value : data) value = distribution(generator);
}
