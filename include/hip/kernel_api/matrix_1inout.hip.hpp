// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernel_api/matrix_1inout.hip.hpp

#pragma once

#include <hip/hip_runtime.h>
#include <Eigen/Dense>

template <typename Matrix_kernel_spec_1Inout>
concept MATRIX_KERNEL_SPEC_1INOUT = requires (Matrix_kernel_spec_1Inout spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;
};

template <typename Matrix_kernel_spec_1Inout>
struct Check_matrix_kernel_spec_1Inout {
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().n_cols_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().n_cols_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_1Inout>().grid_dim_), const dim3>);

    static_assert(MATRIX_KERNEL_SPEC_1INOUT<Matrix_kernel_spec_1Inout>, "not a valid MATRIX_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Matrix_kernel_1Inout>
concept MATRIX_KERNEL_1INOUT = requires (Matrix_kernel_1Inout kernel) {
    typename Matrix_kernel_1Inout::Number;
    typename Matrix_kernel_1Inout::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Matrix_kernel_1Inout::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Matrix_kernel_1Inout::Number*>(),
        std::declval<typename Matrix_kernel_1Inout::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Matrix_kernel_1Inout::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<void>;
};

template <typename Matrix_kernel_1Inout>
struct Check_matrix_kernel_1Inout {
    using Number = typename Matrix_kernel_1Inout::Number;
    using Kernel_spec = typename Matrix_kernel_1Inout::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1Inout>().spec_), const typename Matrix_kernel_1Inout::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1Inout>().run_device_kernel(
        std::declval<typename Matrix_kernel_1Inout::Number*>(),
        std::declval<typename Matrix_kernel_1Inout::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_1Inout>().run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Matrix_kernel_1Inout::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), void>);

    static_assert(MATRIX_KERNEL_1INOUT<Matrix_kernel_1Inout>, "not a valid MATRIX_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <HIP_scalar HIP_Number> class Matrix_kernel_1Inout>
struct Check_matrix_kernel_1Inout_template {
    static_assert(Check_matrix_kernel_1Inout<Matrix_kernel_1Inout<__half>>::check_passed);
    static_assert(Check_matrix_kernel_1Inout<Matrix_kernel_1Inout<float>>::check_passed);
    static_assert(Check_matrix_kernel_1Inout<Matrix_kernel_1Inout<double>>::check_passed);

    constexpr static bool check_passed = true;
};
