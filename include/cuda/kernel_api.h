// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api.h

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <type_traits>

#include "cuda/type_traits.h"

template <typename Kernel_spec_2In_1Out>
concept KERNEL_SPEC_2IN_1OUT = requires (Kernel_spec_2In_1Out spec) {
    { spec.m_ } -> std::same_as<const unsigned int&>;
    { spec.n_ } -> std::same_as<const unsigned int&>;
    { spec.k_ } -> std::same_as<const unsigned int&>;

    { spec.n_rows_A_ } -> std::same_as<const unsigned int&>;
    { spec.n_cols_A_ } -> std::same_as<const unsigned int&>;

    { spec.n_rows_B_ } -> std::same_as<const unsigned int&>;
    { spec.n_cols_B_ } -> std::same_as<const unsigned int&>;

    { spec.n_rows_C_ } -> std::same_as<const unsigned int&>;
    { spec.n_cols_C_ } -> std::same_as<const unsigned int&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;
    { spec.shared_mem_size_ } -> std::same_as<const size_t&>;
};

template <typename Kernel_spec_2In_1Out>
struct Check_kernel_spec_2In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().m_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().k_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_A_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_A_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_B_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_B_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_rows_C_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().n_cols_C_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().grid_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_2In_1Out>().shared_mem_size_), const size_t>);

    static_assert(KERNEL_SPEC_2IN_1OUT<Kernel_spec_2In_1Out>, "not a valid KERNEL_SPEC_2IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel_2In_1Out>
concept KERNEL_2IN_1OUT = requires (Kernel_2In_1Out kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename Kernel_2In_1Out::Number;
    typename Kernel_2In_1Out::Kernel_spec;
    // requires CUDA_floating_point<typename Kernel_2In_1Out::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename Kernel_2In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

};

template <typename Kernel_2In_1Out>
struct Check_kernel_2In_1Out {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using Number = typename Kernel_2In_1Out::Number;
    using Kernel_spec = typename Kernel_2In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().spec_), const typename Kernel_2In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().run_device_kernel(
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<const typename Kernel_2In_1Out::Number*>(),
        std::declval<typename Kernel_2In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel_2In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>(),
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<typename Kernel_2In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(KERNEL_2IN_1OUT<Kernel_2In_1Out>, "not a valid KERNEL_2IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_floating_point CUDA_FLOAT> class Kernel_2In_1Out>
struct Check_kernel_2In_1Out_template {
    // static_assert(Check_kernel_2In_1Out<Kernel_2In_1Out<__half>>::check_passed);
    static_assert(Check_kernel_2In_1Out<Kernel_2In_1Out<float>>::check_passed);
    static_assert(Check_kernel_2In_1Out<Kernel_2In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};






template <typename Kernel_spec_1In_1Out>
concept KERNEL_SPEC_1IN_1OUT = requires (Kernel_spec_1In_1Out spec) {
    { spec.m_ } -> std::same_as<const unsigned int&>;
    { spec.n_ } -> std::same_as<const unsigned int&>;
    { spec.k_ } -> std::same_as<const unsigned int&>;

    { spec.n_rows_A_ } -> std::same_as<const unsigned int&>;
    { spec.n_cols_A_ } -> std::same_as<const unsigned int&>;

    { spec.n_rows_C_ } -> std::same_as<const unsigned int&>;
    { spec.n_cols_C_ } -> std::same_as<const unsigned int&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;
    { spec.shared_mem_size_ } -> std::same_as<const size_t&>;
};

template <typename Kernel_spec_1In_1Out>
struct Check_kernel_spec_1In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().m_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().k_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_rows_A_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_cols_A_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_rows_C_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().n_cols_C_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().grid_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec_1In_1Out>().shared_mem_size_), const size_t>);

    static_assert(KERNEL_SPEC_1IN_1OUT<Kernel_spec_1In_1Out>, "not a valid KERNEL_SPEC_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel_1In_1Out>
concept KERNEL_1IN_1OUT = requires (Kernel_1In_1Out kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename Kernel_1In_1Out::Number;
    typename Kernel_1In_1Out::Kernel_spec_1In_1Out;
    // requires CUDA_floating_point<typename Kernel_1In_1Out::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename Kernel_1In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

};

template <typename Kernel_1In_1Out>
struct Check_kernel_1In_1Out {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using Number = typename Kernel_1In_1Out::Number;
    using Kernel_spec = typename Kernel_1In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().spec_), const typename Kernel_1In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().run_device_kernel(
        std::declval<const typename Kernel_1In_1Out::Number*>(),
        std::declval<typename Kernel_1In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel_1In_1Out>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>()
    )), Eigen::Matrix<typename Kernel_1In_1Out::Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);

    static_assert(KERNEL_1IN_1OUT<Kernel_1In_1Out>, "not a valid KERNEL_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_floating_point CUDA_FLOAT> class Kernel_1In_1Out>
struct Check_kernel_1In_1Out_template {
    // static_assert(Check_kernel_1In_1Out<Kernel_1In_1Out<__half>>::check_passed);
    static_assert(Check_kernel_1In_1Out<Kernel_1In_1Out<float>>::check_passed);
    static_assert(Check_kernel_1In_1Out<Kernel_1In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};
