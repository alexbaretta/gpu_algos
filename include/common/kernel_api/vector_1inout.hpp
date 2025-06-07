// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "cuda/type_traits.hpp"

template <typename Vector_kernel_spec_1InOut>
concept VECTOR_KERNEL_SPEC_1INOUT = requires (Vector_kernel_spec_1InOut spec) {
    { spec.n_C_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Vector_kernel_spec_1InOut>
struct Check_vector_kernel_spec_1InOut {
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_1InOut>().n_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_1InOut>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_1InOut>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_1InOut>().dynamic_shared_mem_words_), const size_t>);

    static_assert(VECTOR_KERNEL_SPEC_1INOUT<Vector_kernel_spec_1InOut>, "not a valid VECTOR_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Vector_kernel_1InOut>
concept VECTOR_KERNEL_1INOUT = requires (Vector_kernel_1InOut kernel) {
    typename Vector_kernel_1InOut::Number;
    typename Vector_kernel_1InOut::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Vector_kernel_1InOut::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Vector_kernel_1InOut::Number*>(),
        std::declval<typename Vector_kernel_1InOut::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Vector_kernel_1InOut::Number, Eigen::Dynamic, 1>>&>()
    ) } -> std::same_as<void>;
};

template <typename Vector_kernel_1InOut>
struct Check_vector_kernel_1InOut {
    using Number = typename Vector_kernel_1InOut::Number;
    using Kernel_spec = typename Vector_kernel_1InOut::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Vector_kernel_1InOut>().spec_), const typename Vector_kernel_1InOut::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_1InOut>().run_device_kernel(
        std::declval<typename Vector_kernel_1InOut::Number*>(),
        std::declval<typename Vector_kernel_1InOut::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_1InOut>().run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Vector_kernel_1InOut::Number, Eigen::Dynamic, 1>>&>()
    )), void>);

    static_assert(VECTOR_KERNEL_1INOUT<Vector_kernel_1InOut>, "not a valid VECTOR_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Vector_kernel_1InOut>
struct Check_vector_kernel_1InOut_template {
    static_assert(Check_vector_kernel_1InOut<Vector_kernel_1InOut<__half>>::check_passed);
    static_assert(Check_vector_kernel_1InOut<Vector_kernel_1InOut<float>>::check_passed);
    static_assert(Check_vector_kernel_1InOut<Vector_kernel_1InOut<double>>::check_passed);

    constexpr static bool check_passed = true;
};
