// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernel_api/matrix_3in_1out.hip.hpp

#pragma once

#include <hip/hip_runtime.h>
#include <Eigen/Dense>

template <typename Matrix_kernel_spec_3In_1Out>
concept MATRIX_KERNEL_SPEC_3IN_1OUT = requires (Matrix_kernel_spec_3In_1Out spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;

    { spec.n_rows_B_ } -> std::same_as<const long&>;
    { spec.n_cols_B_ } -> std::same_as<const long&>;

    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;

    { spec.n_rows_D_ } -> std::same_as<const long&>;
    { spec.n_cols_D_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Matrix_kernel_spec_3In_1Out>
struct Check_matrix_kernel_spec_3In_1Out {
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_cols_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_rows_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_cols_B_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_rows_D_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_cols_D_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().n_cols_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_spec_3In_1Out>().dynamic_shared_mem_words_), const size_t>);

    static_assert(MATRIX_KERNEL_SPEC_3IN_1OUT<Matrix_kernel_spec_3In_1Out>, "not a valid MATRIX_KERNEL_SPEC_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Matrix_kernel_3In_1Out>
concept MATRIX_KERNEL_3IN_1OUT = requires (Matrix_kernel_3In_1Out kernel) {
    typename Matrix_kernel_3In_1Out::Number;
    typename Matrix_kernel_3In_1Out::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Matrix_kernel_3In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
};

template <typename Matrix_kernel_3In_1Out>
struct Check_matrix_kernel_3In_1Out {
    using Number = typename Matrix_kernel_3In_1Out::Number;
    using Kernel_spec = typename Matrix_kernel_3In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_3In_1Out>().spec_), const typename Matrix_kernel_3In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_3In_1Out>().run_device_kernel(
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<const typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<typename Matrix_kernel_3In_1Out::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Matrix_kernel_3In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<typename Matrix_kernel_3In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(MATRIX_KERNEL_3IN_1OUT<Matrix_kernel_3In_1Out>, "not a valid MATRIX_KERNEL_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Matrix_kernel_3In_1Out>
struct Check_matrix_kernel_3In_1Out_template {
    static_assert(Check_matrix_kernel_3In_1Out<Matrix_kernel_3In_1Out<__half>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Matrix_kernel_3In_1Out<float>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Matrix_kernel_3In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};
