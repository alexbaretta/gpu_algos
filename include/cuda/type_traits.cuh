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


// source path: include/cuda/type_traits.cuh

#pragma once

#include <stdfloat>

#include <cuda_runtime.h>
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


#include <Eigen/Dense>
template<typename MATRIX_LIKE>
concept is_matrix_like = (
    std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    || std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Map<Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>
) && is_CUDA_scalar_v<typename MATRIX_LIKE::Scalar>;

template <CUDA_scalar T>
using Printable_Number = std::conditional_t<std::is_same_v<T, __half>, float, T>;
