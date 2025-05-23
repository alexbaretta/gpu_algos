// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/random.h

#pragma once

#include <random>
#include <vector>
#include <cuda_fp16.h>


// Function to initialize vector with random values
template <std::floating_point T>
void randomize_vector(
    std::vector<T>& data,
    int seed
) {
    constexpr T min = 0.0;
    constexpr T max = 1.0;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> distribution(min, max);

    for (auto& value : data) value = distribution(generator);
}

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
