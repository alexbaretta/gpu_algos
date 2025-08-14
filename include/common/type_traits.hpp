
/*
    Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// source path: include/common/type_traits.hpp

#include <cstdint>
#include <string_view>

template <typename T>
struct type_t {};


consteval std::string_view type_name(type_t<float>) { return {"float"}; }
consteval std::string_view type_name(type_t<double>) { return {"double"}; }
consteval std::string_view type_name(type_t<int8_t>) { return {"int8"}; }
consteval std::string_view type_name(type_t<int16_t>) { return {"int16"}; }
consteval std::string_view type_name(type_t<int32_t>) { return {"int32"}; }
consteval std::string_view type_name(type_t<int64_t>) { return {"int64"}; }
consteval std::string_view type_name(type_t<uint8_t>) { return {"uint8"}; }
consteval std::string_view type_name(type_t<uint16_t>) { return {"uint16"}; }
consteval std::string_view type_name(type_t<uint32_t>) { return {"uint32"}; }
consteval std::string_view type_name(type_t<uint64_t>) { return {"uint64"}; }

template <typename Operation>
concept OPERATION = requires(typename Operation::Number a, typename Operation::Number b) {
    { Operation::apply(a, b) } -> std::same_as<typename Operation::Number>;
    requires std::is_empty_v<Operation>;
};

