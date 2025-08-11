/*
    Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// source path: include/common/random.hpp

#pragma once


#include <random>
#include <concepts>
#include <utility>

#include <type_traits>

template <typename Container, typename T>
concept Forward_iterable = requires (Container vector) {
    { vector.begin() } -> std::forward_iterator;
    { *(vector.begin()) } -> std::same_as<T&>;
};

template <typename Container>
using contained_type = std::remove_reference_t<decltype(*std::declval<Container>().begin())>;

// Function to initialize container with random values of type T, converted from a random sequence of type U
template <typename Container, typename T = contained_type<Container>, std::floating_point U = std::conditional_t<std::is_floating_point_v<T>, T, float>>
requires Forward_iterable<Container, T> && (!std::is_integral_v<T>)
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
template <typename Container, typename T = contained_type<Container>, std::integral U = std::conditional_t<std::is_integral_v<T>, T, int>>
requires Forward_iterable<Container, T> && std::is_integral_v<T>
void randomize_container(
    Container& data,
    int seed,
    const U min = U(0),
    const U max = U(100)
) {
    // static_assert(std::is_integral_v<U>, "U must be a C++ integral type");
    std::mt19937 generator(seed);
    std::uniform_int_distribution<U> distribution(min, max);

    for (auto& value : data) value = distribution(generator);
}
