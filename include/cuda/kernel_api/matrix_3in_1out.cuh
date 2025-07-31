// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/matrix_3in_1out.hpp

#pragma once

#include <concepts>
#include <utility>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#include "cuda/kernel_detect_types.cuh"
#include "cuda/type_traits.cuh"

template <typename Kernel_spec>
concept MATRIX_KERNEL_SPEC_3IN_1OUT = requires (Kernel_spec spec) {
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

    { dim3(spec.block_dim_) } -> std::same_as<dim3>;
    { dim3(spec.grid_dim_) } -> std::same_as<dim3>;
};

template <typename Kernel_spec>
struct Check_matrix_kernel_spec_3In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_B_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_D_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_D_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_temp_), const long>);

    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().block_dim_), const dim3>);
    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().grid_dim_), const dim3>);

    static_assert(MATRIX_KERNEL_SPEC_3IN_1OUT<Kernel_spec>, "not a valid MATRIX_KERNEL_SPEC_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel>
concept MATRIX_KERNEL_3IN_1OUT = requires (Kernel kernel) {
    typename Kernel::Number;
    typename Kernel::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Kernel::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename detect::Numbers<Kernel>::A*>(),
        std::declval<const typename detect::Numbers<Kernel>::B*>(),
        std::declval<const typename detect::Numbers<Kernel>::C*>(),
        std::declval<typename detect::Numbers<Kernel>::D*>(),
        std::declval<typename detect::Numbers<Kernel>::Temp*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename detect::Numbers<Kernel>::A, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename detect::Numbers<Kernel>::B, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename detect::Numbers<Kernel>::C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename detect::Numbers<Kernel>::A, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
};

template <typename Kernel>
struct Check_matrix_kernel_3In_1Out {
    using Kernel_spec = typename Kernel::Kernel_spec;
    using Numbers = detect::Numbers<Kernel>;
    using NumberA = typename Numbers::A;
    using NumberB = typename Numbers::B;
    using NumberC = typename Numbers::C;
    using NumberD = typename Numbers::D;
    using NumberE = typename Numbers::E;
    using NumberTemp = typename Numbers::Temp;

    static_assert(std::same_as<decltype(std::declval<Kernel>().spec_), const typename Kernel::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel>().run_device_kernel(
        std::declval<const NumberA*>(),
        std::declval<const NumberB*>(),
        std::declval<const NumberC*>(),
        std::declval<NumberD*>(),
        std::declval<NumberTemp*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<NumberB, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<NumberC, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(MATRIX_KERNEL_3IN_1OUT<Kernel>, "not a valid MATRIX_KERNEL_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Kernel>
struct Check_matrix_kernel_3In_1Out_template {
    static_assert(Check_matrix_kernel_3In_1Out<Kernel<std::int8_t>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Kernel<std::uint8_t>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Kernel<__half>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Kernel<float>>::check_passed);
    static_assert(Check_matrix_kernel_3In_1Out<Kernel<double>>::check_passed);

    constexpr static bool check_passed = true;
};
