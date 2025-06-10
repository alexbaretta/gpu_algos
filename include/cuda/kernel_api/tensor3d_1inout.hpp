// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/tensor3d_1inout.hpp

#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "common/types/tensor3d.hpp"

template <typename Tensor3D_kernel_spec_1Inout>
concept TENSOR3D_KERNEL_SPEC_1INOUT = requires (Tensor3D_kernel_spec_1Inout spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;
    { spec.n_sheets_A_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;
    { spec.n_sheets_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Tensor3D_kernel_spec_1Inout>
struct Check_tensor3d_kernel_spec_1Inout {
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_cols_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_sheets_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_cols_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().n_sheets_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_1Inout>().dynamic_shared_mem_words_), const size_t>);

    static_assert(TENSOR3D_KERNEL_SPEC_1INOUT<Tensor3D_kernel_spec_1Inout>, "not a valid TENSOR3D_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Tensor3D_kernel_1Inout>
concept TENSOR3D_KERNEL_1INOUT = requires (Tensor3D_kernel_1Inout kernel) {
    typename Tensor3D_kernel_1Inout::Number;
    typename Tensor3D_kernel_1Inout::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Tensor3D_kernel_1Inout::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename Tensor3D_kernel_1Inout::Number*>(),
        std::declval<typename Tensor3D_kernel_1Inout::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Tensor3D<typename Tensor3D_kernel_1Inout::Number>&>()
    ) } -> std::same_as<void>;
};

template <typename Tensor3D_kernel_1Inout>
struct Check_tensor3d_kernel_1Inout {
    using Number = typename Tensor3D_kernel_1Inout::Number;
    using Kernel_spec = typename Tensor3D_kernel_1Inout::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_1Inout>().spec_), const typename Tensor3D_kernel_1Inout::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_1Inout>().run_device_kernel(
        std::declval<typename Tensor3D_kernel_1Inout::Number*>(),
        std::declval<typename Tensor3D_kernel_1Inout::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_1Inout>().run_host_kernel(
        std::declval<Tensor3D<typename Tensor3D_kernel_1Inout::Number>&>()
    )), void>);

    static_assert(TENSOR3D_KERNEL_1INOUT<Tensor3D_kernel_1Inout>, "not a valid TENSOR3D_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Tensor3D_kernel_1Inout>
struct Check_tensor3d_kernel_1Inout_template {
    static_assert(Check_tensor3d_kernel_1Inout<Tensor3D_kernel_1Inout<__half>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Tensor3D_kernel_1Inout<float>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Tensor3D_kernel_1Inout<double>>::check_passed);

    constexpr static bool check_passed = true;
};
