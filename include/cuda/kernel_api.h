// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api.h

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <type_traits>

#include "cuda/type_traits.h"

template <typename KERNEL_SPEC>
concept Kernel_spec = requires (KERNEL_SPEC spec) {
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

template <typename KERNEL_SPEC>
struct Check_kernel_spec {
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().m_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().k_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_rows_A_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_cols_A_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_rows_B_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_cols_B_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_rows_C_), const unsigned int>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().n_cols_C_), const unsigned int>);

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().grid_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().shared_mem_size_), const size_t>);

    static_assert(Kernel_spec<KERNEL_SPEC>, "KERNEL_SPEC is not a valid kernel spec");

    constexpr static bool check_passed = true;
};

template <typename KERNEL>
concept Kernel = requires (KERNEL kernel) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename KERNEL::NUMBER;
    typename KERNEL::KERNEL_SPEC;
    // requires CUDA_floating_point<typename KERNEL::FLOAT>;

    { kernel.spec_ } -> std::same_as<const typename KERNEL::KERNEL_SPEC&>;
    { kernel.run_kernel(
        std::declval<const typename KERNEL::NUMBER*>(),
        std::declval<const typename KERNEL::NUMBER*>(),
        std::declval<typename KERNEL::NUMBER*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
};

template <typename KERNEL>
struct Check_kernel {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using NUMBER = typename KERNEL::NUMBER;
    using KERNEL_SPEC = typename KERNEL::KERNEL_SPEC;

    static_assert(std::same_as<decltype(std::declval<KERNEL>().spec_), const typename KERNEL::KERNEL_SPEC>);
    static_assert(std::same_as<decltype(std::declval<KERNEL>().run_kernel(
        std::declval<const typename KERNEL::NUMBER*>(),
        std::declval<const typename KERNEL::NUMBER*>(),
        std::declval<typename KERNEL::NUMBER*>(),
        std::declval<cudaStream_t>()
    )), void>);

    static_assert(Kernel<KERNEL>, "KERNEL is not a valid kernel");

    constexpr static bool check_passed = true;
};

template <template <CUDA_floating_point CUDA_FLOAT> class KERNEL>
struct Check_kernel_template {
    // static_assert(Check_kernel<KERNEL<__half>>::check_passed);
    static_assert(Check_kernel<KERNEL<float>>::check_passed);
    static_assert(Check_kernel<KERNEL<double>>::check_passed);

    constexpr static bool check_passed = true;
};
