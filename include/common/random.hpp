// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/random.hpp

#pragma once


#include <random>
#include <concepts>
#include <utility>

#include <type_traits>
#include <vector>


template <typename Container, typename T>
concept Forward_iterable = requires (Container vector) {
    { vector.begin() } -> std::forward_iterator;
    { *(vector.begin()) } -> std::same_as<T&>;
};

template <typename Container>
using contained_type = std::remove_reference_t<decltype(*std::declval<Container>().begin())>;

// Function to initialize container with random values of type T, converted from a random sequence of type U
template <typename Container, typename T = contained_type<Container>, std::floating_point U = std::conditional_t<std::is_floating_point_v<T>, T, float>>
requires Forward_iterable<Container, T>
void randomize_container(
    Container& data,
    int seed
) {
    // static_assert(std::is_floating_point_v<U>, "U must be a C++ floating point type");
    constexpr U min = 0.0;
    constexpr U max = 1.0;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<U> distribution(min, max);

    for (auto& value : data) value = distribution(generator);
}

// Function to initialize container with random values of type T, converted from a random sequence of type U
template <typename Container, typename T = contained_type<Container>, std::integral U = std::conditional_t<std::is_integral_v<T>, T, int>, U default_min = 0, U default_max = 100>
requires Forward_iterable<Container, T>
void randomize_container(
    std::vector<T>& data,
    int seed,
    const U min = default_min,
    const U max = default_max
) {
    // static_assert(std::is_floating_point_v<U>, "U must be a C++ integeral type");
    std::mt19937 generator(seed);
    std::uniform_int_distribution<U> distribution(min, max);

    for (auto& value : data) value = distribution(generator);
}
