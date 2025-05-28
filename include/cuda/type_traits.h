// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/type_traits.h

#pragma once

#include <cuda_fp16.h>
#include <concepts>
#include <type_traits>

template <typename T>
using is_CUDA_floating_point = std::bool_constant<
    std::is_floating_point_v<T> || std::is_same_v<T, __half>
>;

template <typename T>
constexpr bool is_CUDA_floating_point_v = is_CUDA_floating_point<T>::value;

template <typename T>
concept CUDA_floating_point = is_CUDA_floating_point_v<T>;

template <typename T>
inline constexpr bool dependent_false_v = false;

template <CUDA_floating_point CUDA_FLOAT>
struct CUDA_vector2 {
    static_assert(dependent_false_v<CUDA_FLOAT>, "CUDA_vector2 must be specialized");
    using vector_t = void; // This will never be reached due to static_assert
};

template <CUDA_floating_point CUDA_FLOAT>
using CUDA_vector2_t = CUDA_vector2<CUDA_FLOAT>::vector_t;


template <> struct CUDA_vector2<__half> { using vector_t = __half2; };
template <> struct CUDA_vector2<float> { using vector_t = float2; };
template <> struct CUDA_vector2<double> { using vector_t = double2; };
