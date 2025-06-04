// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "cuda/type_traits.h"

// Use a simple tensor representation as 3D array dimensions
template <typename Number>
struct Tensor3D {
    long rows;
    long cols;
    long sheets;
    Number* data;
};

template <typename Tensor3d_kernel_spec_1InOut>
concept TENSOR3D_KERNEL_SPEC_1INOUT = requires (Tensor3d_kernel_spec_1InOut spec) {
    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;
    { spec.n_sheets_C_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Tensor3d_kernel_spec_1InOut>
struct Check_tensor3d_kernel_spec_1InOut {
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().n_cols_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().n_sheets_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1InOut>().dynamic_shared_mem_words_), const size_t>);

    static_assert(TENSOR3D_KERNEL_SPEC_1INOUT<Tensor3d_kernel_spec_1InOut>, "not a valid TENSOR3D_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Tensor3d_kernel_1InOut>
concept TENSOR3D_KERNEL_1INOUT = requires (Tensor3d_kernel_1InOut kernel) {
    typename Tensor3d_kernel_1InOut::Number;
    typename Tensor3d_kernel_1InOut::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Tensor3d_kernel_1InOut::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Tensor3d_kernel_1InOut::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Tensor3D<typename Tensor3d_kernel_1InOut::Number>&>()
    ) } -> std::same_as<void>;
};

template <typename Tensor3d_kernel_1InOut>
struct Check_tensor3d_kernel_1InOut {
    using Number = typename Tensor3d_kernel_1InOut::Number;
    using Kernel_spec = typename Tensor3d_kernel_1InOut::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1InOut>().spec_), const typename Tensor3d_kernel_1InOut::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1InOut>().run_device_kernel(
        std::declval<typename Tensor3d_kernel_1InOut::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1InOut>().run_host_kernel(
        std::declval<Tensor3D<typename Tensor3d_kernel_1InOut::Number>&>()
    )), void>);

    static_assert(TENSOR3D_KERNEL_1INOUT<Tensor3d_kernel_1InOut>, "not a valid TENSOR3D_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Tensor3d_kernel_1InOut>
struct Check_tensor3d_kernel_1InOut_template {
    static_assert(Check_tensor3d_kernel_1InOut<Tensor3d_kernel_1InOut<__half>>::check_passed);
    static_assert(Check_tensor3d_kernel_1InOut<Tensor3d_kernel_1InOut<float>>::check_passed);
    static_assert(Check_tensor3d_kernel_1InOut<Tensor3d_kernel_1InOut<double>>::check_passed);

    constexpr static bool check_passed = true;
};
