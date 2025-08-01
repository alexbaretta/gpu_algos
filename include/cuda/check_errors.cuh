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


// source path: include/cuda/check_errors.cuh

#pragma once

#include <iostream>

#define cuda_check_error(cuda_err, step_name) \
    do { \
        if (cuda_err != cudaSuccess) { \
            auto msg = cudaGetErrorString(cuda_err); \
            std::cerr << "[CUDA error " << step_name << "] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            exit(1); \
        } \
    } while (0)
