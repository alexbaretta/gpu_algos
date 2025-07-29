// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/common/metaprogramming.hpp

#include <type_traits>

template <typename T, typename V, typename... Vs>
struct is_any {
    constexpr static bool value = std::is_same_v<T, V> || is_any<T, Vs...>::value;
};

template <typename T, typename V>
struct is_any<T, V> {
    constexpr static bool value = std::is_same_v<T, V>;
};

template <typename T, typename... Vs>
constexpr static bool is_any_v = is_any<T, Vs...>::value;
