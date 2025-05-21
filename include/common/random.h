// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <random>
#include <vector>


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
