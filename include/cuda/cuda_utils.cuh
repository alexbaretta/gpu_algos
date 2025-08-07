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


// source path: include/cuda/cuda_utils.cuh

#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "check_errors.cuh"
#include "type_traits.cuh"

#if __cplusplus <= 202002L
namespace std {
    using float16_t = _Float16;
}
#endif

constexpr size_t NULL_FLAGS = 0;
constexpr static long MAX_BLOCK_SIZE = 1024;
constexpr static unsigned FULL_MASK = std::numeric_limits<unsigned>::max();

#define STATIC_WARP_SIZE 32 // TODO: How can I check that warpSize is less than or equal to this?
#define WARP_SIZE warpSize
#define MAX_N_WARPS (MAX_BLOCK_SIZE / WARP_SIZE)
#define LAST_LANE (WARP_SIZE - 1)

// Utility macros for thread indexing
#define BLOCK_1D_NTHREADS   (blockDim.x)
#define BLOCK_2D_NTHREADS   (BLOCK_1D_NTHREADS * blockDim.y)
#define BLOCK_3D_NTHREADS   (BLOCK_3D_NTHREADS * blockDim.Z)
#define BLOCK_1D_NWARPS     (BLOCK_1D_NTHREADS / warpSize)
#define BLOCK_2D_NWARPS     (BLOCK_2D_NTHREADS / warpSize)
#define BLOCK_3D_NWARPS     (BLOCK_3D_NTHREADS / warpSize)
#define GRID_1D_BID         (blockIdx.x)
#define GRID_2D_BID         (GRID_1D_BID + blockIdx.y * gridDim.x)
#define GRID_3D_BID         (GRID_2D_BID + blockIdx.z * gridDim.y * gridDim.x)
#define BLOCK_1D_TID_BLOCK  (threadIdx.x)
#define BLOCK_2D_TID_BLOCK  (BLOCK_1D_TID_BLOCK + blockDim.x * threadIdx.y)
#define BLOCK_3D_TID_BLOCK  (BLOCK_2D_TID_BLOCK + blockDim.x * blockDim.y * threadIdx.z)
#define BLOCK_1D_WID_BLOCK  (BLOCK_1D_TID_BLOCK / warpSize)
#define BLOCK_2D_WID_BLOCK  (BLOCK_2D_TID_BLOCK / warpSize)
#define BLOCK_3D_WID_BLOCK  (BLOCK_3D_TID_BLOCK / warpSize)


__host__ __inline__
unsigned int get_warp_size() {
    cudaDeviceProp props;
    cuda_check_error(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
    return props.warpSize;
}
__host__ __inline__
unsigned int compute_n_threads_per_block(const dim3 block_dim) {
    return long(block_dim.x) * long(block_dim.y) * long(block_dim.z);
}
__device__ __inline__
unsigned int compute_n_threads_per_block() {
    return long(blockDim.x) * long(blockDim.y) * long(blockDim.z);
}

template <std::integral T, std::integral U>
__host__ __inline__
unsigned int compute_n_warps_per_block(const T n_threads_per_block, const U warp_size = get_warp_size()) {
    return (n_threads_per_block + warp_size - 1) / warp_size;
}

__host__ __inline__
unsigned int compute_n_warps_per_block(const dim3& block_dim, const int warp_size = get_warp_size()) {
    return compute_n_warps_per_block(compute_n_threads_per_block(dim3(block_dim)), warp_size);
}

__device__ __inline__
unsigned int get_n_warps_per_block(const long n_threads_per_block) {
    return (n_threads_per_block + warpSize - 1) / warpSize;
}
__device__ __inline__
unsigned int get_n_warps_per_block() {
    return (compute_n_threads_per_block() + warpSize - 1) / warpSize;
}

__host__ __device__ __inline__
void* align_pointer(void *ptr, const std::size_t alignment) {
    const std::size_t ptr_value = reinterpret_cast<size_t>(ptr);
    const std::size_t aligned_ptr_value = (ptr_value + (alignment - 1)) & ~(alignment - 1);
    void* const aligned_ptr = reinterpret_cast<void*>(aligned_ptr_value);
    return aligned_ptr;
}

// Device function to get dynamic shared memory pointer - non-template to avoid symbol issues
// We cannot declare dynamic_shm aligned with __aligned__(alignof(T)) because this would require
// templatizing this function, which causes linker errors.
__device__ __forceinline__ unsigned dynamic_shared_mem_size() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}
__device__ __forceinline__ unsigned total_shared_mem_size() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %total_smem_size;" : "=r"(ret));
    return ret;
}
template <typename T>
__device__ __inline__
T* get_dynamic_shared_memory() {
    assert(dynamic_shared_mem_size() > sizeof(T));
    extern __shared__ void* dynamic_shm[];
    const std::size_t alignment = alignof(T);
    void* aligned_shm = align_pointer(dynamic_shm, alignment);
    return reinterpret_cast<T*>(aligned_shm);
}
__device__ __inline__
void* get_dynamic_shared_memory(const std::size_t alignment) {
    assert(dynamic_shared_mem_size() > alignment);
    extern __shared__ void* dynamic_shm[];
    void* aligned_shm = align_pointer(dynamic_shm, alignment);
    return aligned_shm;
}


template <CUDA_scalar Number>
__host__ __device__ Number cuda_max(Number a, Number b) {
    return max(a, b);
}

// Template specialization declarations (defined in cuda_utils.cu)
template <>
__host__ __device__ __half cuda_max<__half>(__half a, __half b);

template <CUDA_scalar Number>
__host__ __device__ Number cuda_min(Number a, Number b) {
    return min(a, b);
}

// Template specialization declarations (defined in cuda_utils.cu)
template <>
__host__ __device__ __half cuda_min<__half>(__half a, __half b);

template <CUDA_scalar Number>
__host__ __device__ Number cuda_sum(Number a, Number b) {
    return a + b;
}

template <CUDA_scalar Number>
__host__ __device__ Number cuda_prod(Number a, Number b) {
    return a * b;
}

template <CUDA_scalar Number>
__host__ __device__ Number device_nan() {
    static_assert(std::is_floating_point_v<Number>, "device_nan only supports floating point types");
    if constexpr (std::is_same_v<Number, float>) {
        return nanf(nullptr);
    } else if constexpr (std::is_same_v<Number, double>) {
        return nan(nullptr);
    }
    // __half specialization is defined in cuda_utils.cu
}

// Template specialization declaration (defined in cuda_utils.cu)
template <>
__host__ __device__ __half device_nan<__half>();

void report_completion_time_callback(cudaStream_t stream, cudaError_t status, void* userData);

// Function objects for CUDA operations - these work reliably in device code
template <typename Number_>
struct cuda_max_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return max(a, b);
    }

    __host__ __device__ static Number identity() {
        if constexpr (std::is_same_v<Number, __half>) {
            return -CUDART_INF_FP16;
        } else if constexpr (std::is_floating_point_v<Number>) {
            return -std::numeric_limits<Number>::infinity();
        } else {
            return Number(1) << (sizeof(Number) * 8 - 1);
        }
    }
};

template <>
struct cuda_max_op<__half> {
    using Number = __half;
    __host__ __device__ static Number apply(Number a, Number b) {
        return __hmax(a, b);
    }

    __host__ __device__ static Number identity() {
        return -CUDART_INF_FP16;
    }
};

template <typename Number_>
struct cuda_min_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return min(a, b);
    }

    __host__ __device__ static Number identity() {
        if constexpr (std::is_same_v<Number, __half>) {
            return CUDART_INF_FP16;
        } else if constexpr (std::is_floating_point_v<Number>) {
            return std::numeric_limits<Number>::infinity();
        } else {
            return ~(Number(1) << (sizeof(Number) * 8 - 1));
        }
    }
};

template <>
struct cuda_min_op<__half> {
    using Number = __half;
    __host__ __device__ static Number apply(Number a, Number b) {
        return __hmin(a, b);
    }

    __host__ __device__ static Number identity() {
        return CUDART_INF_FP16;
    }
};

template <typename Number_>
struct cuda_sum_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return a + b;
    }

    __host__ __device__ static Number identity() {
        return Number(0);
    }
};

template <typename Number_>
struct cuda_prod_op {
    using Number = Number_;
    __host__ __device__ static Number apply(Number a, Number b) {
        return a * b;
    }

    __host__ __device__ static Number identity() {
        return Number(1);
    }
};

cudaDeviceProp get_device_prop(const int device_id);
cudaDeviceProp get_default_device_prop();

template <typename Number_, int size>
requires (size <= 4)
struct CUDA_vector {
    // You must use a specialization
    std::enable_if_t<false, Number_> must_use_a_specialization;
};
template <typename Number_, int size>
using CUDA_vector_t = typename CUDA_vector<Number_, size>::vector_type;

template <>
struct CUDA_vector<std::int8_t, 2> {
    using scalar_type = std::int8_t;
    using vector_type = char2;
};
template <>
struct CUDA_vector<std::int8_t, 3> {
    using scalar_type = std::int8_t;
    using vector_type = char3;
};
template <>
struct CUDA_vector<std::int8_t, 4> {
    using scalar_type = std::int8_t;
    using vector_type = char4;
};

template <>
struct CUDA_vector<std::uint8_t, 2> {
    using scalar_type = std::uint8_t;
    using vector_type = uchar2;
};
template <>
struct CUDA_vector<std::uint8_t, 3> {
    using scalar_type = std::uint8_t;
    using vector_type = uchar3;
};
template <>
struct CUDA_vector<std::uint8_t, 4> {
    using scalar_type = std::uint8_t;
    using vector_type = uchar4;
};
template <>
struct CUDA_vector<std::int16_t, 2> {
    using scalar_type = std::int16_t;
    using vector_type = short2;
};
template <>
struct CUDA_vector<std::int16_t, 3> {
    using scalar_type = std::int16_t;
    using vector_type = short3;
};
template <>
struct CUDA_vector<std::int16_t, 4> {
    using scalar_type = std::int16_t;
    using vector_type = short4;
};

template <>
struct CUDA_vector<std::uint16_t, 2> {
    using scalar_type = std::uint16_t;
    using vector_type = ushort2;
};
template <>
struct CUDA_vector<std::uint16_t, 3> {
    using scalar_type = std::uint16_t;
    using vector_type = ushort3;
};
template <>
struct CUDA_vector<std::uint16_t, 4> {
    using scalar_type = std::uint16_t;
    using vector_type = ushort4;
};

template <>
struct CUDA_vector<std::int32_t, 2> {
    using scalar_type = std::int32_t;
    using vector_type = int2;
};
template <>
struct CUDA_vector<std::int32_t, 3> {
    using scalar_type = std::int32_t;
    using vector_type = int3;
};
template <>
struct CUDA_vector<std::int32_t, 4> {
    using scalar_type = std::int32_t;
    using vector_type = int4;
};

template <>
struct CUDA_vector<std::uint32_t, 2> {
    using scalar_type = uint32_t;
    using vector_type = uint2;
};
template <>
struct CUDA_vector<std::uint32_t, 3> {
    using scalar_type = uint32_t;
    using vector_type = uint3;
};
template <>
struct CUDA_vector<std::uint32_t, 4> {
    using scalar_type = uint32_t;
    using vector_type = uint4;
};

template <>
struct CUDA_vector<std::int64_t, 2> {
    using scalar_type = std::int64_t;
    using vector_type = long2;
};
template <>
struct CUDA_vector<std::int64_t, 3> {
    using scalar_type = std::int64_t;
    using vector_type = long3;
};
template <>
struct CUDA_vector<std::int64_t, 4> {
    using scalar_type = std::int64_t;
    using vector_type = long4;
};

template <>
struct CUDA_vector<std::uint64_t, 2> {
    using scalar_type = std::uint64_t;
    using vector_type = ulong2;
};
template <>
struct CUDA_vector<std::uint64_t, 3> {
    using scalar_type = std::uint64_t;
    using vector_type = ulong3;
};
template <>
struct CUDA_vector<std::uint64_t, 4> {
    using scalar_type = std::uint64_t;
    using vector_type = ulong4;
};


template <>
struct CUDA_vector<float, 2> {
    using scalar_type = float;
    using vector_type = float2;
};
template <>
struct CUDA_vector<float, 3> {
    using scalar_type = float;
    using vector_type = float3;
};
template <>
struct CUDA_vector<float, 4> {
    using scalar_type = float;
    using vector_type = float4;
};

template <>
struct CUDA_vector<double, 2> {
    using scalar_type = double;
    using vector_type = double2;
};
template <>
struct CUDA_vector<double, 3> {
    using scalar_type = double;
    using vector_type = double3;
};
template <>
struct CUDA_vector<double, 4> {
    using scalar_type = double;
    using vector_type = double4;
};
