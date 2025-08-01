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


// source path: src/common/benchmark_options.cpp

#include "common/benchmark_options.hpp"

void add_benchmark_options(cxxopts::Options& options) {
    options.add_options()
        ("i,init-method", "How to generate data: random, increasing, decreasing", cxxopts::value<std::string>()->default_value("random"))
        ("gpumem", "GPU memory size", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_GPU_MEM)))
        ("seed", "Random seed", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_SEED)))
        ("tol-bits", "Number of bits of precision lost due to arithmetic rounding", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_TOL_BITS)))
        ("errors", "Display errors", cxxopts::value<bool>()->default_value("false"))
        ("verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("force,f", "Allow verbose output even if the input matrix is large", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
}
