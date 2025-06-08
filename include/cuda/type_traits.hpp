// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/type_traits.hpp

#pragma once

#include <cuda_fp16.h>
#include <type_traits>

template <typename T>
using is_CUDA_integer = std::bool_constant<
    std::is_integral<T>::value &&
        (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
>;

template <typename T>
constexpr bool is_CUDA_integer_v = is_CUDA_integer<T>::value;

template <typename T>
concept CUDA_integer = is_CUDA_integer_v<T>;

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

template <typename T>
using is_CUDA_scalar = std::bool_constant<
    is_CUDA_integer_v<T> || is_CUDA_floating_point_v<T>
>;

template <typename T>
constexpr bool is_CUDA_scalar_v = is_CUDA_scalar<T>::value;

template <typename T>
concept CUDA_scalar = is_CUDA_scalar_v<T>;



template <CUDA_floating_point CUDA_FLOAT>
using CUDA_vector2_t = CUDA_vector2<CUDA_FLOAT>::vector_t;


template <> struct CUDA_vector2<__half> { using vector_t = __half2; };
template <> struct CUDA_vector2<float> { using vector_t = float2; };
template <> struct CUDA_vector2<double> { using vector_t = double2; };

#include <Eigen/Dense>
template<typename MATRIX_LIKE>
concept is_matrix_like = (
    std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    || std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Map<Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>
) && is_CUDA_scalar_v<typename MATRIX_LIKE::Scalar>;
