// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_detect_types.cuh

#pragma once

#include <concepts>
#include <condition_variable>
#include <string>
#include <tuple>
#include <type_traits>

#include "mma.h"
#include "cuda/type_traits.cuh"

namespace detect {
    template <typename KERNEL> concept has_NumberA = requires { typename KERNEL::NumberA; };
    template <typename KERNEL> concept has_NumberB = requires { typename KERNEL::NumberB;};
    template <typename KERNEL> concept has_NumberC = requires { typename KERNEL::NumberC;};
    template <typename KERNEL> concept has_NumberD = requires { typename KERNEL::NumberD;};
    template <typename KERNEL> concept has_NumberTemp = requires { typename KERNEL::NumberTemp;};
    template <typename KERNEL> concept has_NumberInternal = requires { typename KERNEL::NumberInternal;};

    template <typename KERNEL> struct _NumberA { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberB { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberC { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberD { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberTemp { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberInternal { using type = typename KERNEL::Number; };

    template <has_NumberA KERNEL> struct _NumberA<KERNEL> { using type = typename KERNEL::NumberA;};
    template <has_NumberB KERNEL> struct _NumberB<KERNEL> { using type = typename KERNEL::NumberB;};
    template <has_NumberC KERNEL> struct _NumberC<KERNEL> { using type = typename KERNEL::NumberC;};
    template <has_NumberD KERNEL> struct _NumberD<KERNEL> { using type = typename KERNEL::NumberD;};
    template <has_NumberTemp KERNEL> struct _NumberTemp<KERNEL> { using type = typename KERNEL::NumberTemp;};
    template <has_NumberInternal KERNEL> struct _NumberInternal<KERNEL> { using type = typename KERNEL::NumberInternal;};

    template <typename KERNEL> using NumberA = typename _NumberA<KERNEL>::type;
    template <typename KERNEL> using NumberB = typename _NumberB<KERNEL>::type;
    template <typename KERNEL> using NumberC = typename _NumberC<KERNEL>::type;
    template <typename KERNEL> using NumberD = typename _NumberD<KERNEL>::type;
    template <typename KERNEL> using NumberTemp = typename _NumberTemp<KERNEL>::type;
    template <typename KERNEL> using NumberInternal = typename _NumberInternal<KERNEL>::type;


    // The error type is the same as the output type: D, for 3IN 1OUT; C for 1/2IN 1OUT, A for 1INOUT
    template <typename KERNEL> struct _NumberE {
        using type = std::conditional_t<
            std::is_integral_v<typename KERNEL::Number>,
            double,
            typename KERNEL::Number
        >;
    };
    template <has_NumberD KERNEL> struct _NumberE<KERNEL> { using type = typename KERNEL::NumberD;};
    template <has_NumberA KERNEL> requires (!has_NumberD<KERNEL>) struct _NumberE<KERNEL> { using type = typename KERNEL::NumberA;};
    template <typename KERNEL> using NumberE = typename _NumberE<KERNEL>::type;

    template <typename...> struct _Tolerance;

    template <unsigned precision_bits>
    struct Precision {
        constexpr static float value = 0.5 * Precision<precision_bits-1>::value;
    };
    template <> struct Precision<0> {
        constexpr static float value = 1.0;
    };


    template <>
    struct _Tolerance<__half> {
        using type = __half;
        static std::string name() { return "CUDA __half"; };
        constexpr static int precision_bits = 11;
        constexpr static float precision = Precision<precision_bits>::value;
        constexpr static float tolerance(const int lost_precision_bits) {
            return std::pow(0.5, precision_bits - lost_precision_bits);
        }
        constexpr static float tolerance_pct(const int lost_precision_bits) {
            return tolerance(lost_precision_bits) * 100.0;
        }
    };
    template <>
    struct _Tolerance<nvcuda::wmma::precision::tf32> {
        using type = nvcuda::wmma::precision::tf32;
        static std::string name() { return "TensorFloat32"; };
        constexpr static int precision_bits = 11;
        constexpr static float precision = Precision<precision_bits>::value;
        constexpr static float tolerance(const int lost_precision_bits) {
            return std::pow(0.5, precision_bits - lost_precision_bits);
        }
        constexpr static float tolerance_pct(const int lost_precision_bits) {
            return tolerance(lost_precision_bits) * 100.0;
        }
    };
    template <>
    struct _Tolerance<float> {
        using type = float;
        static std::string name() { return "float"; };
        constexpr static int precision_bits = 24;
        constexpr static float precision = Precision<precision_bits>::value;
        constexpr static float tolerance(const int lost_precision_bits) {
            return std::pow(0.5, precision_bits - lost_precision_bits);
        }
        constexpr static float tolerance_pct(const int lost_precision_bits) {
            return tolerance(lost_precision_bits) * 100.0;
        }
    };
    template <>
    struct _Tolerance<double> {
        using type = double;
        static std::string name() { return "double"; };
        constexpr static int precision_bits = 53;
        constexpr static float precision = Precision<precision_bits>::value;
        constexpr static float tolerance(const int lost_precision_bits) {
            return std::pow(0.5, precision_bits - lost_precision_bits);
        }
        constexpr static float tolerance_pct(const int lost_precision_bits) {
            return tolerance(lost_precision_bits) * 100.0;
        }
    };
    template <std::unsigned_integral Integer>
    struct _Tolerance<Integer> {
        using type = Integer;
        static std::string name() { return "unsigned integer"; };
        constexpr static int precision_bits = 8 * sizeof(Integer);
        constexpr static float precision = 0;
        constexpr static float tolerance(const int) { return 0; }
        constexpr static float tolerance_pct(const int) { return 0; }
    };
    template <std::signed_integral Integer>
    struct _Tolerance<Integer> {
        using type = Integer;
        static std::string name() { return "signed integer"; };
        constexpr static int precision_bits = 8 * sizeof(Integer);
        constexpr static float precision = 0;
        constexpr static float tolerance(const int) { return 0; }
        constexpr static float tolerance_pct(const int) { return 0; }
    };

    template <typename T1, typename T2, typename... Ts>
    struct _Tolerance<T1, T2, Ts...>
    : _Tolerance<
        std::conditional_t<_Tolerance<T1>::precision >= _Tolerance<T2>::precision, T1, T2>,
        Ts...
    > {};

    template <typename KERNEL>
    struct Numbers {
        using A = NumberA<KERNEL>;
        using B = NumberB<KERNEL>;
        using C = NumberC<KERNEL>;
        using D = NumberD<KERNEL>;
        using Temp = NumberTemp<KERNEL>;
        using Internal = NumberInternal<KERNEL>;
        using E = NumberE<KERNEL>;
        using Tolerance = _Tolerance<A, B, C, D, Temp, Internal>;
    };
}
