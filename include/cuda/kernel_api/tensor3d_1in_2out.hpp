// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "cuda/type_traits.hpp"
#include "common/types/tensor3d.hpp"

template <typename Tensor3d_kernel_spec_1In_2Out>
concept TENSOR3D_KERNEL_SPEC_1IN_2OUT = requires (Tensor3d_kernel_spec_1In_2Out spec) {
    { spec.n_rows_A_ } -> std::same_as<const long&>;
    { spec.n_cols_A_ } -> std::same_as<const long&>;
    { spec.n_sheets_A_ } -> std::same_as<const long&>;

    { spec.n_rows_C1_ } -> std::same_as<const long&>;
    { spec.n_cols_C1_ } -> std::same_as<const long&>;
    { spec.n_sheets_C1_ } -> std::same_as<const long&>;

    { spec.n_rows_C2_ } -> std::same_as<const long&>;
    { spec.n_cols_C2_ } -> std::same_as<const long&>;
    { spec.n_sheets_C2_ } -> std::same_as<const long&>;

    { spec.n_rows_temp_ } -> std::same_as<const long&>;
    { spec.n_cols_temp_ } -> std::same_as<const long&>;
    { spec.n_sheets_temp_ } -> std::same_as<const long&>;

    { spec.block_dim_ } -> std::same_as<const dim3&>;
    { spec.grid_dim_ } -> std::same_as<const dim3&>;

    { spec.dynamic_shared_mem_words_ } -> std::same_as<const size_t&>;
};

template <typename Tensor3d_kernel_spec_1In_2Out>
struct Check_tensor3d_kernel_spec_1In_2Out {
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_rows_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_cols_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_sheets_A_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_rows_C1_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_cols_C1_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_sheets_C1_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_rows_C2_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_cols_C2_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_sheets_C2_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_rows_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_cols_temp_), const long>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().n_sheets_temp_), const long>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().block_dim_), const dim3>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().grid_dim_), const dim3>);

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_spec_1In_2Out>().dynamic_shared_mem_words_), const size_t>);

    static_assert(TENSOR3D_KERNEL_SPEC_1IN_2OUT<Tensor3d_kernel_spec_1In_2Out>, "not a valid TENSOR3D_KERNEL_SPEC_1IN_2OUT");

    constexpr static bool check_passed = true;
};

template <typename Tensor3d_kernel_1In_2Out>
concept TENSOR3D_KERNEL_1IN_2OUT = requires (Tensor3d_kernel_1In_2Out kernel) {
    typename Tensor3d_kernel_1In_2Out::Number;
    typename Tensor3d_kernel_1In_2Out::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Tensor3d_kernel_1In_2Out::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<const Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>&>()
    ) } -> std::same_as<std::pair<Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>, Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>>>;
};

template <typename Tensor3d_kernel_1In_2Out>
struct Check_tensor3d_kernel_1In_2Out {
    using Number = typename Tensor3d_kernel_1In_2Out::Number;
    using Kernel_spec = typename Tensor3d_kernel_1In_2Out::Kernel_spec;

    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1In_2Out>().spec_), const typename Tensor3d_kernel_1In_2Out::Kernel_spec>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1In_2Out>().run_device_kernel(
        std::declval<const typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<typename Tensor3d_kernel_1In_2Out::Number*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Tensor3d_kernel_1In_2Out>().run_host_kernel(
        std::declval<const Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>&>()
    )), std::pair<Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>, Tensor3D<typename Tensor3d_kernel_1In_2Out::Number>>>);

    static_assert(TENSOR3D_KERNEL_1IN_2OUT<Tensor3d_kernel_1In_2Out>, "not a valid TENSOR3D_KERNEL_1IN_2OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Tensor3d_kernel_1In_2Out>
struct Check_tensor3d_kernel_1In_2Out_template {
    static_assert(Check_tensor3d_kernel_1In_2Out<Tensor3d_kernel_1In_2Out<__half>>::check_passed);
    static_assert(Check_tensor3d_kernel_1In_2Out<Tensor3d_kernel_1In_2Out<float>>::check_passed);
    static_assert(Check_tensor3d_kernel_1In_2Out<Tensor3d_kernel_1In_2Out<double>>::check_passed);

    constexpr static bool check_passed = true;
};
