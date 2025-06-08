// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernel_api/matrix_1inout.hip.hpp

#pragma once

#include <hip/hip_runtime.h>
#include <Eigen/Dense>

template <typename Matrix_kernel_spec_1InOut>
concept MATRIX_KERNEL_SPEC_1INOUT = requires (Matrix_kernel_spec_1InOut spec) {
    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Matrix_kernel_spec_1InOut>
struct Check_matrix_kernel_spec_1InOut {
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1InOut>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1InOut>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1InOut>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1InOut>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1InOut>().dynamic_shared_mem_words_), const size_t>);

    static_assert(MATRIX_KERNEL_SPEC_1INOUT<Matrix_kernel_spec_1InOut>, "not a valid MATRIX_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Matrix_kernel_1InOut>
concept MATRIX_KERNEL_1INOUT = requires (Matrix_kernel_1InOut kernel) {
    typename Matrix_kernel_1InOut::Number;
    typename Matrix_kernel_1InOut::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Matrix_kernel_1InOut::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Matrix_kernel_1InOut::Number*>(),
        std::declval<typename Matrix_kernel_1InOut::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Matrix_kernel_1InOut::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<void>;
};

template <typename Matrix_kernel_1InOut>
struct Check_matrix_kernel_1InOut {
    using Number = typename Matrix_kernel_1InOut::Number;
    using Kernel_spec = typename Matrix_kernel_1InOut::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1InOut>().spec_), const typename Matrix_kernel_1InOut::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1InOut>().run_device_kernel(
        std::declval<typename Matrix_kernel_1InOut::Number*>(),
        std::declval<typename Matrix_kernel_1InOut::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1InOut>().run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Matrix_kernel_1InOut::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), void>);

    static_assert(MATRIX_KERNEL_1INOUT<Matrix_kernel_1InOut>, "not a valid MATRIX_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Matrix_kernel_1InOut>
struct Check_matrix_kernel_1InOut_template {
    static_assert(Check_matrix_kernel_1InOut<Matrix_kernel_1InOut<__half>>::check_passed);
    static_assert(Check_matrix_kernel_1InOut<Matrix_kernel_1InOut<float>>::check_passed);
    static_assert(Check_matrix_kernel_1InOut<Matrix_kernel_1InOut<double>>::check_passed);

    constexpr static bool check_passed = true;
};
