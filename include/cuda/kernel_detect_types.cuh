// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernel_detect_types.cuh

#pragma once

#include <type_traits>

namespace detect {
    template <typename KERNEL> concept has_NumberA = requires { typename KERNEL::NumberA; };
    template <typename KERNEL> concept has_NumberB = requires { typename KERNEL::NumberB;};
    template <typename KERNEL> concept has_NumberC = requires { typename KERNEL::NumberC;};
    template <typename KERNEL> concept has_NumberD = requires { typename KERNEL::NumberD;};
    template <typename KERNEL> concept has_NumberTemp = requires { typename KERNEL::NumberTemp;};

    template <typename KERNEL> struct _NumberA { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberB { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberC { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberD { using type = typename KERNEL::Number; };
    template <typename KERNEL> struct _NumberTemp { using type = typename KERNEL::Number; };

    template <has_NumberA KERNEL> struct _NumberA<KERNEL> { using type = typename KERNEL::NumberA;};
    template <has_NumberB KERNEL> struct _NumberB<KERNEL> { using type = typename KERNEL::NumberB;};
    template <has_NumberC KERNEL> struct _NumberC<KERNEL> { using type = typename KERNEL::NumberC;};
    template <has_NumberD KERNEL> struct _NumberD<KERNEL> { using type = typename KERNEL::NumberD;};
    template <has_NumberTemp KERNEL> struct _NumberTemp<KERNEL> { using type = typename KERNEL::NumberTemp;};

    template <typename KERNEL> using NumberA = typename _NumberA<KERNEL>::type;
    template <typename KERNEL> using NumberB = typename _NumberB<KERNEL>::type;
    template <typename KERNEL> using NumberC = typename _NumberC<KERNEL>::type;
    template <typename KERNEL> using NumberD = typename _NumberD<KERNEL>::type;
    template <typename KERNEL> using NumberTemp = typename _NumberTemp<KERNEL>::type;


    // The error type is the same as the output type: D, for 3IN 1OUT; C for 1/2IN 1OUT, A for 1INOUT
    template <typename KERNEL> struct _NumberE { using type = typename KERNEL::Number; };
    template <has_NumberD KERNEL> struct _NumberE<KERNEL> { using type = typename KERNEL::NumberD;};
    template <has_NumberA KERNEL> requires (!has_NumberD<KERNEL>) struct _NumberE<KERNEL> { using type = typename KERNEL::NumberA;};
    template <typename KERNEL> using NumberE = typename _NumberE<KERNEL>::type;

    template <typename KERNEL>
    struct Numbers {
        using A = NumberA<KERNEL>;
        using B = NumberB<KERNEL>;
        using C = NumberC<KERNEL>;
        using D = NumberD<KERNEL>;
        using Temp = NumberTemp<KERNEL>;
        using E = NumberE<KERNEL>;
    };
}
