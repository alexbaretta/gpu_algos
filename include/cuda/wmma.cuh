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


// source path: include/cuda/wmma.cuh

#pragma once

#include <cstdint>

#include <mma.h>
#include <cublasLt.h>

// Looks like nvcc does not actually enforce this.
constexpr static long wmma_alignment_requirement_bits = 256;
constexpr static long wmma_alignment_requirement_bytes = wmma_alignment_requirement_bits/8;

template <typename T>
struct wmma_config;

template <>
struct wmma_config<std::int8_t> {
    using argument_type = std::int8_t;
    using operand_type = std::int8_t;
    using accumulator_type = int;
    using result_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");

    // cuBLASlt config
    constexpr static cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32I;
    constexpr static cudaDataType_t cublas_scale_type = CUDA_R_32I;
    constexpr static cudaDataType_t cublas_operand_type = CUDA_R_8I;
};

template <>
struct wmma_config<std::uint8_t> {
    using argument_type = uint8_t;
    using operand_type = std::uint8_t;
    using accumulator_type = int;
    using result_type = int;
    using temp_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");

    // cuBLASlt config
    constexpr static cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32I;
    constexpr static cudaDataType_t cublas_scale_type = CUDA_R_32I;
    constexpr static cudaDataType_t cublas_operand_type = CUDA_R_8U;
};

template <>
struct wmma_config<__half> {
    using argument_type = __half;
    using operand_type = __half;
    using accumulator_type = __half;
    using result_type = __half;
    using temp_type = __half;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");

    // cuBLASlt config
    constexpr static cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_16F;
    constexpr static cudaDataType_t cublas_scale_type = CUDA_R_16F;
    constexpr static cudaDataType_t cublas_operand_type = CUDA_R_16F;
};

template <>
struct wmma_config<float> {
    // tf32 is just a placeholder type declared as a struct without a definition (an incomplete type)
    using argument_type = float;
    using operand_type = nvcuda::wmma::precision::tf32;
    using accumulator_type = float;
    using result_type = float;
    using temp_type = float;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 8;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");

    // cuBLASlt config
    constexpr static cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    constexpr static cudaDataType_t cublas_scale_type = CUDA_R_32F;
    constexpr static cudaDataType_t cublas_operand_type = CUDA_R_32F;
};

template <>
struct wmma_config<double> {
    using argument_type = double;
    using operand_type = double;
    using accumulator_type = double;
    using result_type = double;
    using temp_type = double;
    constexpr static unsigned M = 8;
    constexpr static unsigned N = 8;
    constexpr static unsigned K = 4;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");

    // cuBLASlt config
    constexpr static cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_64F;
    constexpr static cudaDataType_t cublas_scale_type = CUDA_R_64F;
    constexpr static cudaDataType_t cublas_operand_type = CUDA_R_64F;
};

template <typename T>
concept wmma_type = requires {
    typename wmma_config<T>::argument_type;
};

static_assert(wmma_type<std::int8_t>, "float is not recognized as a wmma_type");
static_assert(wmma_type<std::uint8_t>, "float is not recognized as a wmma_type");

static_assert(wmma_type<__half>, "float is not recognized as a wmma_type");
static_assert(wmma_type<float>, "float is not recognized as a wmma_type");
static_assert(wmma_type<double>, "float is not recognized as a wmma_type");
