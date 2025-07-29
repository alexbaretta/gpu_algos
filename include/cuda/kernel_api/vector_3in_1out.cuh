// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/vector_3in_1out.hpp

#pragma once

#include <concepts>
#include <utility>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#include "cuda/kernel_detect_types.cuh"
#include "cuda/type_traits.cuh"

template <typename Vector_kernel_spec_3In_1Out>
concept VECTOR_KERNEL_SPEC_3IN_1OUT = requires (Vector_kernel_spec_3In_1Out spec) {
    { spec.n_A_ } -> std::same_as<const long&>;
    { spec.n_B_ } -> std::same_as<const long&>;
    { spec.n_C_ } -> std::same_as<const long&>;
    { spec.n_D_ } -> std::same_as<const long&>;
    { spec.n_temp_ } -> std::same_as<const long&>;

    { dim3(spec.block_dim_) } -> std::same_as<dim3>;
    { dim3(spec.grid_dim_) } -> std::same_as<dim3>;
};

template <typename Vector_kernel_spec_3In_1Out>
struct Check_vector_kernel_spec_3In_1Out {
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_3In_1Out>().n_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_3In_1Out>().n_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_3In_1Out>().n_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_3In_1Out>().n_D_), const long>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_spec_3In_1Out>().n_temp_), const long>);

    static_assert(std::convertible_to<decltype(std::declval<Vector_kernel_spec_3In_1Out>().block_dim_), const dim3>);
    static_assert(std::convertible_to<decltype(std::declval<Vector_kernel_spec_3In_1Out>().grid_dim_), const dim3>);

    static_assert(VECTOR_KERNEL_SPEC_3IN_1OUT<Vector_kernel_spec_3In_1Out>, "not a valid VECTOR_KERNEL_SPEC_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Vector_kernel_3In_1Out>
concept VECTOR_KERNEL_3IN_1OUT = requires (Vector_kernel_3In_1Out kernel) {
    typename Vector_kernel_3In_1Out::Number;
    typename Vector_kernel_3In_1Out::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Vector_kernel_3In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>;
};

template <typename Vector_kernel_3In_1Out>
struct Check_vector_kernel_3In_1Out {
    using Number = typename Vector_kernel_3In_1Out::Number;
    using Kernel_spec = typename Vector_kernel_3In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Vector_kernel_3In_1Out>().spec_), const typename Vector_kernel_3In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_3In_1Out>().run_device_kernel(
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<const typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<typename Vector_kernel_3In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Vector_kernel_3In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>&>()
    )), Eigen::Matrix<typename Vector_kernel_3In_1Out::Number, Eigen::Dynamic, 1>>);

    static_assert(VECTOR_KERNEL_3IN_1OUT<Vector_kernel_3In_1Out>, "not a valid VECTOR_KERNEL_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Vector_kernel_3In_1Out>
struct Check_vector_kernel_3In_1Out_template {
    static_assert(Check_vector_kernel_3In_1Out<Vector_kernel_3In_1Out<__half>>::check_passed);
    static_assert(Check_vector_kernel_3In_1Out<Vector_kernel_3In_1Out<float>>::check_passed);
    static_assert(Check_vector_kernel_3In_1Out<Vector_kernel_3In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};
