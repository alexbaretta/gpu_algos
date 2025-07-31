// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_api/vector_1in_1out.hpp

#pragma once

#include <concepts>
#include <utility>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/kernel_detect_types.cuh"
#include "cuda/type_traits.cuh"

template <typename Kernel_spec>
concept VECTOR_KERNEL_SPEC_1IN_1OUT = requires (Kernel_spec spec) {
    { spec.n_A_ } -> std::same_as<const long&>;
    { spec.n_C_ } -> std::same_as<const long&>;
    { spec.n_temp_ } -> std::same_as<const long&>;

    { dim3(spec.block_dim_) } -> std::same_as<dim3>;
    { dim3(spec.grid_dim_) } -> std::same_as<dim3>;
};

template <typename Kernel_spec>
struct Check_vector_kernel_spec_1In_1Out {
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_A_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_C_), const long>);
    static_assert(std::same_as<decltype(std::declval<Kernel_spec>().n_temp_), const long>);

    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().block_dim_), const dim3>);
    static_assert(std::convertible_to<decltype(std::declval<Kernel_spec>().grid_dim_), const dim3>);

    static_assert(VECTOR_KERNEL_SPEC_1IN_1OUT<Kernel_spec>, "not a valid VECTOR_KERNEL_SPEC_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <typename Kernel>
concept VECTOR_KERNEL_1IN_1OUT = requires (Kernel kernel) {
    typename Kernel::Number;
    typename Kernel::Kernel_spec;

    { kernel.spec_ } -> std::same_as<const typename Kernel::Kernel_spec&>;
    { kernel.run_device_kernel(
        std::declval<const typename detect::Numbers<Kernel>::A*>(),
        std::declval<typename detect::Numbers<Kernel>::C*>(),
        std::declval<typename detect::Numbers<Kernel>::Temp*>(),
        std::declval<cudaStream_t>()
    ) } -> std::same_as<void>;
    { kernel.run_host_kernel(
        std::declval<Eigen::Map<Eigen::Matrix<typename detect::Numbers<Kernel>::A, Eigen::Dynamic, 1>>&>()
    ) } -> std::same_as<Eigen::Matrix<typename detect::Numbers<Kernel>::A, Eigen::Dynamic, 1>>;
};

template <typename Kernel>
struct Check_vector_kernel_1In_1Out {
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
        std::declval<NumberC*>(),
        std::declval<NumberTemp*>(),
        std::declval<cudaStream_t>()
    )), void>);
    static_assert(std::same_as<decltype(std::declval<Kernel>().run_host_kernel(
        std::declval<const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, 1>>&>()
    )), Eigen::Matrix<NumberC, Eigen::Dynamic, 1>>);

    static_assert(VECTOR_KERNEL_1IN_1OUT<Kernel>, "not a valid VECTOR_KERNEL_1IN_1OUT");

    constexpr static bool check_passed = true;
};

template <template <CUDA_scalar CUDA_Number> class Kernel>
struct Check_vector_kernel_1In_1Out_template {
    static_assert(Check_vector_kernel_1In_1Out<Kernel<std::int8_t>>::check_passed);
    static_assert(Check_vector_kernel_1In_1Out<Kernel<std::uint8_t>>::check_passed);
    static_assert(Check_vector_kernel_1In_1Out<Kernel<__half>>::check_passed);
    static_assert(Check_vector_kernel_1In_1Out<Kernel<float>>::check_passed);
    static_assert(Check_vector_kernel_1In_1Out<Kernel<double>>::check_passed);

    constexpr static bool check_passed = true;
};
