// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/eigen_utils.cuh

#pragma once

#include <cmath>

#include "type_traits.cuh"

struct Compute_error_absolute {
    template <std::floating_point Scalar>
    Scalar operator()(const Scalar& lhs, const Scalar& rhs) const {
        if (std::isnan(lhs) && std::isnan(rhs)) {
            // Both are NaN, so there is no error
            return 0;
        } else if (std::isinf(lhs) && std::isinf(rhs)) {
            const auto inf_delta = lhs - rhs;
            if (std::isnan(inf_delta)) {
                // lhs and rhs are both infinities, so inf_delta is NaN only if they are
                // infinities of the same sign, hence once could argue that there is no error
                return 0;
            }
            else { return std::abs(inf_delta); }
        } else {
            // lhs and rhs and normal numbers. We just compute the absolute difference.
            return std::abs(lhs - rhs);
        }
    }
    __half operator()(const __half& lhs, const __half& rhs) const {
        if (__hisnan(lhs) && __hisnan(rhs)) {
            // Both are NaN, so there is no error
            return 0;
        } else if (__hisinf(lhs) && __hisinf(rhs)) {
            const auto inf_delta = lhs - rhs;
            if (__hisnan(inf_delta)) {
                // lhs and rhs are both infinities, so inf_delta is NaN only if they are
                // infinities of the same sign, hence once could argue that there is no error
                return 0;
            }
            else { return __habs(inf_delta); }
        } else {
            // lhs and rhs and normal numbers. We just compute the difference.
            return __habs(lhs - rhs);
        }
    }
    template <CUDA_integer Scalar>
    std::int64_t operator() (const Scalar& lhs, const Scalar& rhs) const {
        // There is no std::abs for unsigned integers, for obvious reasons, and subtraction
        // of signed integer types is subject to overflow. To make it easy for ourselves,
        // we use the widest signed integer type: int64_t
        return std::abs(std::int64_t(lhs) - std::int64_t(rhs));
    }
};

constexpr static Compute_error_absolute compute_error_absolute{};

struct Compute_error_relative {
    template <CUDA_scalar Scalar>
    double operator()(const Scalar& lhs, const Scalar& rhs) const {
        const auto e = compute_error_absolute(lhs, rhs);
        if (e == decltype(e){0}) {
            return 0;
        } else {
            return double(e)/std::abs(double(rhs));
        }
    }
};

constexpr static Compute_error_relative compute_error_relative{};
