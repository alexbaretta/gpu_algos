// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernel_api.hip.h

#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include <type_traits>

#include <Eigen/Dense>

#include "hip/type_traits.hip.h"

template <typename Kernel_spec_2In_1Out>
concept KERNEL_SPEC_2IN_1OUT = requires (Kernel_spec_2In_1Out spec) {
    { spec.m_ } -> std::same_as<const long&>;
    { spec.n_ } -> std::same_as<const long&>;
    { spec.k_ } -> std::same_as<const long&>;

    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;

    { spec.n_rows_B_ } -> std::same_as<const long&>;
    { spec.n_cols_B_ } -> std::same_as<const long&>;

    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    // Will be allocated at runtime via <<<...>>> kernel launch
    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Kernel_spec_2In_1Out>
struct Check_matrix_kernel_spec_2In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().m_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().k_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_B_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().grid_dim_), const dim3>);

    // Will be allocated at runtime via <<<...>>> kernel launch
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().dynamic_shared_mem_words_), const size_t>);

    static_assert(KERNEL_SPEC_2IN_1OUT<Kernel_spec_2In_1Out>, "not a valid KERNEL_SPEC_2IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel_2In_1Out>
concept KERNEL_2IN_1OUT = requires (Kernel_2In_1Out kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename Kernel_2In_1Out::Number;
    typename Kernel_2In_1Out::Kernel_spec;
    // requires HIP_scalar<typename Kernel_2In_1Out::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename Kernel_2In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

};

template <typename Kernel_2In_1Out>
struct Check_matrix_kernel_2In_1Out {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using Number = typename Kernel_2In_1Out::Number;
    using Kernel_spec = typename Kernel_2In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().spec_), const typename Kernel_2In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().run_device_kernel(
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(KERNEL_2IN_1OUT<Kernel_2In_1Out>, "not a valid KERNEL_2IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <HIP_scalar HIP_Number> class Kernel_2In_1Out>
struct Check_matrix_kernel_2In_1Out_template {
    static_assert(Check_matrix_kernel_2In_1Out<Kernel_2In_1Out<_Float16>>::check_passed);
    static_assert(Check_matrix_kernel_2In_1Out<Kernel_2In_1Out<float>>::check_passed);
    static_assert(Check_matrix_kernel_2In_1Out<Kernel_2In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};






template <typename Kernel_spec_1In_1Out>
concept KERNEL_SPEC_1IN_1OUT = requires (Kernel_spec_1In_1Out spec) {
    { spec.m_ } -> std::same_as<const long&>;
    { spec.n_ } -> std::same_as<const long&>;
    { spec.k_ } -> std::same_as<const long&>;

    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;

    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    // Will be allocated at runtime via <<<...>>> kernel launch
    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Kernel_spec_1In_1Out>
struct Check_matrix_kernel_spec_1In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().m_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().k_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_cols_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_cols_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().grid_dim_), const dim3>);

    // Will be allocated at runtime via <<<...>>> kernel launch
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().dynamic_shared_mem_words_), const size_t>);

    static_assert(KERNEL_SPEC_1IN_1OUT<Kernel_spec_1In_1Out>, "not a valid KERNEL_SPEC_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel_1In_1Out>
concept KERNEL_1IN_1OUT = requires (Kernel_1In_1Out kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename Kernel_1In_1Out::Number;
    typename Kernel_1In_1Out::Kernel_spec;
    // requires HIP_scalar<typename Kernel_1In_1Out::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename Kernel_1In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

};

template <typename Kernel_1In_1Out>
struct Check_matrix_kernel_1In_1Out {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using Number = typename Kernel_1In_1Out::Number;
    using Kernel_spec = typename Kernel_1In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().spec_), const typename Kernel_1In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().run_device_kernel(
        std::declval<const typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(KERNEL_1IN_1OUT<Kernel_1In_1Out>, "not a valid KERNEL_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <HIP_scalar HIP_Number> class Kernel_1In_1Out>
struct Check_matrix_kernel_1In_1Out_template {
    static_assert(Check_matrix_kernel_1In_1Out<Kernel_1In_1Out<_Float16>>::check_passed);
    static_assert(Check_matrix_kernel_1In_1Out<Kernel_1In_1Out<float>>::check_passed);
    static_assert(Check_matrix_kernel_1In_1Out<Kernel_1In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};









template <typename Kernel_spec_1InOut>
concept KERNEL_SPEC_1INOUT = requires (Kernel_spec_1InOut spec) {
    { spec.m_ } -> std::same_as<const long&>;
    { spec.n_ } -> std::same_as<const long&>;

    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    // Will be allocated at runtime via <<<...>>> kernel launch
    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Kernel_spec_1InOut>
struct Check_matrix_kernel_spec_1InOut {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().m_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().n_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().n_cols_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().grid_dim_), const dim3>);

    // Will be allocated at runtime via <<<...>>> kernel launch
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1InOut>().dynamic_shared_mem_words_), const size_t>);

    static_assert(KERNEL_SPEC_1INOUT<Kernel_spec_1InOut>, "not a valid KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel_1InOut>
concept KERNEL_1INOUT = requires (Kernel_1InOut kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename Kernel_1InOut::Number;
    typename Kernel_1InOut::Kernel_spec;
    // requires HIP_scalar<typename Kernel_1InOut::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename Kernel_1InOut::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Kernel_1InOut::Number*>(),
        std::declval<hipStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Kernel_1InOut::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<void>;

};

template <typename Kernel_1InOut>
struct Check_matrix_kernel_1InOut {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using Number = typename Kernel_1InOut::Number;
    using Kernel_spec = typename Kernel_1InOut::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Kernel_1InOut>().spec_), const typename Kernel_1InOut::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1InOut>().run_device_kernel(
        std::declval<typename Kernel_1InOut::Number*>(),
        std::declval<hipStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1InOut>().run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename Kernel_1InOut::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), void>);

    static_assert(KERNEL_1INOUT<Kernel_1InOut>, "not a valid KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <HIP_scalar HIP_Number> class Kernel_1InOut>
struct Check_matrix_kernel_1InOut_template {
    static_assert(Check_matrix_kernel_1InOut<Kernel_1InOut<_Float16>>::check_passed);
    static_assert(Check_matrix_kernel_1InOut<Kernel_1InOut<float>>::check_passed);
    static_assert(Check_matrix_kernel_1InOut<Kernel_1InOut<double>>::check_passed);

    constexpr static bool check_passed = true;
};
