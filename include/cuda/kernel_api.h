// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <type_traits>

#include "cuda/type_traits.h"

template <typename KERNEL_SPEC>
concept Kernel_spec = requires (KERNEL_SPEC spec) {
    // Concepts evaluate what you actually get when expressions run (references to members)...
    typename KERNEL_SPEC::NUMBER;
    // requires CUDA_floating_point<typename KERNEL_SPEC::FLOAT>;
    { spec.grid_dim } -> std::same_as<const dim3&>;
    { spec.block_dim } -> std::same_as<const dim3&>;
    { spec.shared_mem_size } -> std::same_as<const size_t&>;
    { spec.stream } -> std::same_as<cudaStream_t&>;
    { spec.run_kernel() } -> std::same_as<void>;
};

template <typename KERNEL_SPEC>
struct Check_kernel_spec {
    // ...while decltype determines the static declared type of expressions (underlying types)
    using NUMBER = typename KERNEL_SPEC::NUMBER;

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().grid_dim), const dim3>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().block_dim), const dim3>);
    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().shared_mem_size), const size_t>);

    static_assert(std::same_as<decltype(std::declval<KERNEL_SPEC>().run_kernel()), void>);

    static_assert(Kernel_spec<KERNEL_SPEC>, "KERNEL_SPEC is not a valid kernel");

    constexpr static bool check_passed = true;
};

template <template <CUDA_floating_point CUDA_FLOAT> class KERNEL_SPEC>
struct Check_kernel_spec_template {
    static_assert(Check_kernel_spec<KERNEL_SPEC<__half>>::check_passed);
    static_assert(Check_kernel_spec<KERNEL_SPEC<float>>::check_passed);
    static_assert(Check_kernel_spec<KERNEL_SPEC<double>>::check_passed);

    constexpr static bool check_passed = true;
};
