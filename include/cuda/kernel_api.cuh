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


// source path: include/cuda/kernel_api.cuh

#pragma once

#include "cuda/kernel_detect_types.cuh"

#include "cuda/kernel_api/matrix_1inout.cuh"
#include "cuda/kernel_api/matrix_1in_1out.cuh"
#include "cuda/kernel_api/matrix_2in_1out.cuh"
#include "cuda/kernel_api/matrix_3in_1out.cuh"

#include "cuda/kernel_api/vector_1inout.cuh"
#include "cuda/kernel_api/vector_1in_1out.cuh"
#include "cuda/kernel_api/vector_2in_1out.cuh"
#include "cuda/kernel_api/vector_3in_1out.cuh"

#include "cuda/kernel_api/tensor3d_1inout.cuh"
#include "cuda/kernel_api/tensor3d_1in_1out.cuh"
#include "cuda/kernel_api/tensor3d_2in_1out.cuh"
#include "cuda/kernel_api/tensor3d_3in_1out.cuh"
