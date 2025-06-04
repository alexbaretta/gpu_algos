// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/type_traits.h

#pragma once

#include <hip/hip_fp16.h>
#include <concepts>
#include <type_traits>

template <typename T>
using is_HIP_integer = std::bool_constant<
    std::is_integral<T>::value &&
        (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
>;

template <typename T>
constexpr bool is_HIP_integer_v = is_HIP_integer<T>::value;

template <typename T>
concept HIP_integer = is_HIP_integer_v<T>;

template <typename T>
using is_HIP_floating_point = std::bool_constant<
    std::is_floating_point_v<T> || std::is_same_v<T, _Float16>
>;

template <typename T>
constexpr bool is_HIP_floating_point_v = is_HIP_floating_point<T>::value;

template <typename T>
concept HIP_floating_point = is_HIP_floating_point_v<T>;

template <typename T>
inline constexpr bool dependent_false_v = false;

template <HIP_floating_point HIP_FLOAT>
struct HIP_vector2 {
    static_assert(dependent_false_v<HIP_FLOAT>, "HIP_vector2 must be specialized");
    using vector_t = void; // This will never be reached due to static_assert
};

template <typename T>
using is_HIP_scalar = std::bool_constant<
    is_HIP_integer_v<T> || is_HIP_floating_point_v<T>
>;

template <typename T>
constexpr bool is_HIP_scalar_v = is_HIP_scalar<T>::value;

template <typename T>
concept HIP_scalar = is_HIP_scalar_v<T>;



template <HIP_floating_point HIP_FLOAT>
using HIP_vector2_t = HIP_vector2<HIP_FLOAT>::vector_t;


template <> struct HIP_vector2<_Float16> { using vector_t = _Float16_2; };
template <> struct HIP_vector2<float> { using vector_t = float2; };
template <> struct HIP_vector2<double> { using vector_t = double2; };
