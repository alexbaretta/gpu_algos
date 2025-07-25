// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/tensor3d_3in_1out.hpp

#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/type_traits.cuh"

template <typename Tensor3D_kernel_spec_3In_1Out>
concept TENSOR3D_KERNEL_SPEC_3IN_1OUT = requires (Tensor3D_kernel_spec_3In_1Out spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;
    { spec.n_sheets_A_ } -> std::same_as<const long&>;

    { spec.n_rows_B_ } -> std::same_as<const long&>;
    { spec.n_cols_B_ } -> std::same_as<const long&>;
    { spec.n_sheets_B_ } -> std::same_as<const long&>;

    { spec.n_rows_C_ } -> std::same_as<const long&>;
    { spec.n_cols_C_ } -> std::same_as<const long&>;
    { spec.n_sheets_C_ } -> std::same_as<const long&>;

    { spec.n_rows_D_ } -> std::same_as<const long&>;
    { spec.n_cols_D_ } -> std::same_as<const long&>;
    { spec.n_sheets_D_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;
    { spec.n_sheets_temp_ } -> std::same_as<const long&>;

    { dim3(spec.block_dim_) } -> std::same_as<dim3>;
    { dim3(spec.grid_dim_) } -> std::same_as<dim3>;
};

template <typename Tensor3D_kernel_spec_3In_1Out>
struct Check_tensor3d_kernel_spec_3In_1Out {
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_cols_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_sheets_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_rows_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_cols_B_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_sheets_B_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_rows_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_cols_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_sheets_C_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_rows_D_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_cols_D_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_sheets_D_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_cols_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().n_sheets_temp_), const long>);

    static_assert(std::convertible_to<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().block_dim_), const dim3>);
    static_assert(std::convertible_to<decltype(std::declval<Tensor3D_kernel_spec_3In_1Out>().grid_dim_), const dim3>);

    static_assert(TENSOR3D_KERNEL_SPEC_3IN_1OUT<Tensor3D_kernel_spec_3In_1Out>, "not a valid TENSOR3D_KERNEL_SPEC_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Tensor3D_kernel_3In_1Out>
concept TENSOR3D_KERNEL_3IN_1OUT = requires (Tensor3D_kernel_3In_1Out kernel) {
    typename Tensor3D_kernel_3In_1Out::Number;
    typename Tensor3D_kernel_3In_1Out::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Tensor3D_kernel_3In_1Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>(),
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>(),
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>()
    ) } -> std::same_as<Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>>;
};

template <typename Tensor3D_kernel_3In_1Out>
struct Check_tensor3d_kernel_3In_1Out {
    using Number = typename Tensor3D_kernel_3In_1Out::Number;
    using Kernel_spec = typename Tensor3D_kernel_3In_1Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_3In_1Out>().spec_), const typename Tensor3D_kernel_3In_1Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_3In_1Out>().run_device_kernel(
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<const typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<typename Tensor3D_kernel_3In_1Out::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Tensor3D_kernel_3In_1Out>().run_host_kernel(
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>(),
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>(),
        std::declval<const Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>&>()
    )), Tensor3D<typename Tensor3D_kernel_3In_1Out::Number>>);

    static_assert(TENSOR3D_KERNEL_3IN_1OUT<Tensor3D_kernel_3In_1Out>, "not a valid TENSOR3D_KERNEL_3IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Tensor3D_kernel_3In_1Out>
struct Check_tensor3d_kernel_3In_1Out_template {
    static_assert(Check_tensor3d_kernel_3In_1Out<Tensor3D_kernel_3In_1Out<__half>>::check_passed);
    static_assert(Check_tensor3d_kernel_3In_1Out<Tensor3D_kernel_3In_1Out<float>>::check_passed);
    static_assert(Check_tensor3d_kernel_3In_1Out<Tensor3D_kernel_3In_1Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};
