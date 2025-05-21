// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#include <cuda_fp16.h>
#include <concepts>
#include <type_traits>

template <typename T>
concept CUDA_floating_point = std::is_floating_point_v<T> || std::is_same_v<T, __half>;
