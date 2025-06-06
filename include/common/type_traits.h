// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/type_traits.h

#pragma once

#include <Eigen/Dense>
#include "cuda/type_traits.h"


template<typename MATRIX_LIKE>
concept is_matrix_like = (
    std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    || std::is_same_v<std::decay_t<MATRIX_LIKE>, Eigen::Map<Eigen::Matrix<typename MATRIX_LIKE::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>
) && is_CUDA_scalar_v<typename MATRIX_LIKE::Scalar>;
