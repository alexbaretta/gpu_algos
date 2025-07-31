// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/tensor3d_1inout.hpp

#pragma once

#include <concepts>
#include <utility>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/kernel_detect_types.cuh"
#include "cuda/type_traits.cuh"

template <typename Kernel_spec>
concept TENSOR3D_KERNEL_SPEC_1INOUT = requires (Kernel_spec spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;
    { spec.n_sheets_A_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;
    { spec.n_sheets_temp_ } -> std::same_as<const long&>;

    { dim3(spec.block_dim_) } -> std::same_as<dim3>;
    { dim3(spec.grid_dim_) } -> std::same_as<dim3>;
};

template <typename Kernel_spec>
struct Check_tensor3d_kernel_spec_1Inout {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_sheets_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_cols_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_sheets_temp_), const long>);

    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().block_dim_), const dim3>);
    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().grid_dim_), const dim3>);

    static_assert(TENSOR3D_KERNEL_SPEC_1INOUT<Kernel_spec>, "not a valid TENSOR3D_KERNEL_SPEC_1INOUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel>
concept TENSOR3D_KERNEL_1INOUT = requires (Kernel kernel) {
    typename Kernel::Number;
    typename Kernel::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Kernel::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<typename detect::Numbers<Kernel>::A*>(),
        std::declval<typename detect::Numbers<Kernel>::A*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Tensor3D<typename detect::Numbers<Kernel>::A>&>()
    ) } -> std::same_as<void>;
};

template <typename Kernel>
struct Check_tensor3d_kernel_1Inout {
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
        std::declval<NumberA*>(),
        std::declval<NumberA*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel>().run_host_kernel(
        std::declval<Tensor3D<NumberA>&>()
    )), void>);

    static_assert(TENSOR3D_KERNEL_1INOUT<Kernel>, "not a valid TENSOR3D_KERNEL_1INOUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Kernel>
struct Check_tensor3d_kernel_1Inout_template {
    static_assert(Check_tensor3d_kernel_1Inout<Kernel<std::int8_t>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Kernel<std::uint8_t>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Kernel<__half>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Kernel<float>>::check_passed);
    static_assert(Check_tensor3d_kernel_1Inout<Kernel<double>>::check_passed);

    constexpr static bool check_passed = true;
};
