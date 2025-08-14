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


// source path: include/cuda/kernels/vector_reduction/vector_reduction_parallel.cuh

#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "cuda/kernel_api/vector_1in_1out.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/type_traits.cuh"
#include "cuda/check_errors.cuh"



template <CUDA_scalar Number, OPERATION Operation>
__global__ void vector_reduction_by_blocks_parallel(
    const Number* A,
    const long A_n,     // size of A
    Number* C          // size ((n + block_dim - 1) / block_dim)
) {
    // for writing, index this using `wid_block` (warp id)
    Number* shm = get_dynamic_shm<Number>();
    const auto shm_size = get_n_warps_per_block();

    // tid_xxx represent the index of the thread within grid, block, and warp
    // const long bid_grid = blockIdx.x;
    // const unsigned short tid_block = threadIdx.x;
    const long tid_grid = long(threadIdx.x) + long(blockIdx.x) * long(blockDim.x);
    // printf("tid_grid: %d, n: %d, blockDim.x: %d, gridDim.x: %d\n", tid_grid, n, blockDim.x, gridDim.x);

    Number value = Operation::identity();

    const auto tid_warp = threadIdx.x % WARP_SIZE;
    const auto wid_block = threadIdx.x / WARP_SIZE;
    // const auto n_warps_per_block = blockDim.x / WARP_SIZE;

    // Load data, only if index is within bounds
    // Notice that we need to execute this even if tid_grid >= n, because we need to
    // ensure that the shared memory is initialized correctly for the block-level reduction.
    if (tid_grid < A_n) {
        value = A[tid_grid];
    }

    // Warp-shuffle reduction
    for (unsigned short reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
        value = Operation::apply(value, __shfl_down_sync(__activemask(), value, reduced_lanes));
    }

    // Copy reduced value to shared memory
    // Notice that we need to execute this even if tid_grid >= n, because we need to
    // ensure that the shared memory is initialized correctly for the block-level reduction.
    if (tid_warp == 0) {
        shm[wid_block] = value;
    }
    __syncthreads();

    // Use wid_block == 0 to sum the warp-level values recorded in shared memory
    if (wid_block == 0) {
        value = (tid_warp < shm_size) ? shm[tid_warp] : Operation::identity();

        // Warp-shuffle reduction: compute total value for this block
        for (unsigned short reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
            value = Operation::apply(value, __shfl_down_sync(__activemask(), value, reduced_lanes));
        }

        // Write to global memory
        if (tid_warp == 0) {
            C[blockIdx.x] = value;
        }
    }
}

struct Vector_reduction_recursive_spec {
    const cudaDeviceProp device_prop_;

    const std::string type_;
    const std::string operation_;

    const long m_;    // unused for vector scan
    const long n_;    // size of vector
    const long k_;    // unused for vector scan

    const long n_A_;
    const long n_C_;
    const long n_temp_;

    const dim3 block_dim_ = 0;
    const dim3 grid_dim_ = 0;
    const size_t shared_mem_size_ = 0;

    constexpr static int DEFAULT_M = 0;    // unused
    constexpr static int DEFAULT_N = 3000; // size of vector
    constexpr static int DEFAULT_K = 0;    // unused
    constexpr static int DEFAULT_BLOCK_DIM_X = 1024;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("operation,op", "Operation to perform (max, min, sum, prod)", cxxopts::value<std::string>()->default_value("sum"))
            ("N", "Size of vector", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("block-dim", "Number of threads in the x dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Vector_reduction_recursive_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        const auto& operation = options_parsed["operation"].as<std::string>();
        if (operation != "max" && operation != "min" && operation != "sum" && operation != "prod") {
            std::cerr << "[ERROR] --operation must be one of: max, min, sum, prod" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --operation: " + operation);
        }
        const auto size = options_parsed["N"].as<long>();
        const auto block_dim_option = options_parsed["block-dim"].as<unsigned>();
        const auto warp_size = get_warp_size();
        const auto block_size = (std::min(size, (long)block_dim_option)  + warp_size - 1) / warp_size * warp_size;
        return make(
            type,
            operation,
            size,
            block_size
        );
    }

    // We need one word of memory per block to store the reduction result of each pass.
    // We iterate, reducing the number of elements by a factor of block_size each time, until
    // we are left with only one element: the result of the recursive reduction over the entire vector.
    static int compute_size_of_temp(const int n_elems, const int block_size) {
        assert(block_size > 1);
        if (n_elems <= block_size) {
            return 0;
        }
        // We start from 0 rather than 1 as in the scan algorithms, because we don't need to store the
        // final result in the temp buffer
        int size_temp = 0;

        for(int n_elems_remaining = (n_elems + block_size - 1) / block_size;
            n_elems_remaining > 1;
            n_elems_remaining = (n_elems_remaining + block_size - 1) / block_size
        ) {
            size_temp += n_elems_remaining;
        }
        return size_temp;
    }

    inline static Vector_reduction_recursive_spec make(
        const auto& type,
        const auto& operation,
        const long size,
        const long block_size
    ) {
        const auto [scalar_size, cuda_occupancy_kernel]  = (
            type == "half" ? std::make_tuple(sizeof(__half), (void*)vector_reduction_by_blocks_parallel<half, cuda_sum_op<half>>) :
            type == "single" || type == "float" ? std::make_tuple(sizeof(float), (void*)vector_reduction_by_blocks_parallel<float, cuda_sum_op<float>>) :
            type == "double" ? std::make_tuple(sizeof(double), (void*)vector_reduction_by_blocks_parallel<double, cuda_sum_op<double>>) :
            type == "int8" ? std::make_tuple(sizeof(int8_t), (void*)vector_reduction_by_blocks_parallel<int8_t, cuda_sum_op<int8_t>>) :
            type == "int16" ? std::make_tuple(sizeof(int16_t), (void*)vector_reduction_by_blocks_parallel<int16_t, cuda_sum_op<int16_t>>) :
            type == "int32" ? std::make_tuple(sizeof(int32_t), (void*)vector_reduction_by_blocks_parallel<int32_t, cuda_sum_op<int32_t>>) :
            type == "int64" ? std::make_tuple(sizeof(int64_t), (void*)vector_reduction_by_blocks_parallel<int64_t, cuda_sum_op<int64_t>>) :
            type == "uint8" ? std::make_tuple(sizeof(uint8_t), (void*)vector_reduction_by_blocks_parallel<uint8_t, cuda_sum_op<uint8_t>>) :
            type == "uint16" ? std::make_tuple(sizeof(uint16_t), (void*)vector_reduction_by_blocks_parallel<uint16_t, cuda_sum_op<uint16_t>>) :
            type == "uint32" ? std::make_tuple(sizeof(uint32_t), (void*)vector_reduction_by_blocks_parallel<uint32_t, cuda_sum_op<uint32_t>>) :
            type == "uint64" ? std::make_tuple(sizeof(uint64_t), (void*)vector_reduction_by_blocks_parallel<uint64_t, cuda_sum_op<uint64_t>>) :
            std::make_tuple(0, nullptr)
        );
        if (cuda_occupancy_kernel == nullptr) {
            std::cerr << "[ERROR] Invalid type: " << type << std::endl;
            throw cxxopts::exceptions::exception("Invalid type: " + type);
        }
        return make(type, operation, scalar_size, cuda_occupancy_kernel, size, block_size);
    }

    template <CUDA_scalar Number_, typename Operation_= cuda_sum_op<Number_>>
    inline static Vector_reduction_recursive_spec make(
        const long size,
        const long block_size
    ) {
        const auto scalar_size = sizeof(Number_);
        const auto cuda_occupancy_kernel = vector_reduction_by_blocks_parallel<Number_, Operation_>;
        return make(type_name(type_t<Number_>{}), Operation_::name, scalar_size, (void*)cuda_occupancy_kernel, size, block_size);
    }

    protected:
    inline static Vector_reduction_recursive_spec make(
        const auto& type,
        const auto& operation,
        const size_t scalar_size,
        const void* const cuda_occupancy_kernel,
        const long size,
        const long block_size
    ) {
        if (block_size < 1) {
            throw std::invalid_argument("block_size must be at least 1");
        };
        if (cuda_occupancy_kernel == nullptr) {
            throw std::invalid_argument("cuda_occupancy_kernel must not be nullptr");
        };
        if (scalar_size <= 0) {
            throw std::invalid_argument("scalar_size must be positive");
        };

        int max_block_size = 0;
        int opt_grid_size = 0;
        int max_active_blocks_per_multiprocessor = 0;
        const auto shared_mem_guess = (compute_n_warps_per_block(block_size/4)+1) * scalar_size;
        cuda_check_error(cudaOccupancyMaxPotentialBlockSize(
            &opt_grid_size,
            &max_block_size,
            cuda_occupancy_kernel,
            shared_mem_guess,
            block_size
        ), "cudaOccupancyMaxPotentialBlockSize");

        // opt_grid_size is a good-to-know number, but we have to compute grid_dim based on the size of vector
        const auto grid_dim = dim3((size+max_block_size-1)/max_block_size);
        const auto block_dim = dim3(max_block_size);
        const auto shared_mem_size = (compute_n_warps_per_block(max_block_size)+1) * scalar_size;
        const auto n_temp = compute_size_of_temp(size, max_block_size);
        cuda_check_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_multiprocessor,
            cuda_occupancy_kernel,
            block_dim.x,
            shared_mem_size
        ), "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        std::cout << "[INFO] max_active_blocks_per_multiprocessor: " << max_active_blocks_per_multiprocessor
                  << " at block_size:" << max_block_size << std::endl;
        std::cout << "[INFO] max_block_size: " << max_block_size << std::endl;
        std::cout << "[INFO] opt_grid_size: " << opt_grid_size << std::endl;
        std::cout << "[INFO] grid_dim: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
        std::cout << "[INFO] block_dim: " << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << std::endl;
        std::cout << "[INFO] shared_mem_size: " << shared_mem_size << std::endl;
        return Vector_reduction_recursive_spec(type, operation, size, block_dim, grid_dim, shared_mem_size, n_temp);
    }

    protected:
    Vector_reduction_recursive_spec(
        const auto& type,
        const auto& operation,
        const long size,
        const dim3 block_dim,
        const dim3 grid_dim,
        const size_t shared_mem_size,
        const int n_temp
    ) : device_prop_(get_default_device_prop()),
        type_(type),
        operation_(operation),  // Store the actual operation string
        m_(0),  // unused
        n_(size),
        k_(0),  // unused
        n_A_(size),
        n_C_(1),
        n_temp_(n_temp),
        block_dim_(block_dim),
        grid_dim_(grid_dim),
        shared_mem_size_(shared_mem_size)
    {}
};

static_assert(Check_vector_kernel_spec_1In_1Out<Vector_reduction_recursive_spec>::check_passed, "Vector_reduction_recursive_spec is not a valid kernel spec");


template <CUDA_scalar Number_, typename Operation_= cuda_sum_op<Number_>>
class Vector_reduction_recursive_kernel {
    public:
    using Number = Number_;
    using Operation = Operation_;
    using Kernel_spec = Vector_reduction_recursive_spec;

    const Kernel_spec spec_;

    Vector_reduction_recursive_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void reduce_recursive(
        const Number* const input_buffer,
        const int input_index,
        const int input_n_elems,
        Number* const temp_buffer,
        const int temp_index,
        const int temp_size,
        Number* const reduced_result,
        cudaStream_t stream
    ) {
        const int input_n_blocks = (input_n_elems + spec_.block_dim_.x - 1)/spec_.block_dim_.x;
        assert(temp_size >= temp_index + input_n_blocks || input_n_blocks == 1);

        vector_reduction_by_blocks_parallel<Number, Operation><<<input_n_blocks, spec_.block_dim_, spec_.shared_mem_size_, stream>>>(
            input_buffer + input_index,
            input_n_elems,

            // If the kernel launch entails only one block, then no further reduction will be necessary,
            // and we can write the result directly to the final result pointer.
            (input_n_blocks == 1) ? reduced_result : temp_buffer + temp_index
        );

        // if no further reduction is needed, return
        if (input_n_blocks <= 1) return;
        reduce_recursive(
            temp_buffer,
            temp_index,
            input_n_blocks,
            temp_buffer,
            temp_index + input_n_blocks,
            temp_size,
            reduced_result,
            stream
        );
        // By the powers of recursion, reduced_result is fully scanned
    }

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        Number* const gpu_data_temp,
        cudaStream_t stream
    ) {
        reduce_recursive(
            gpu_data_A,
            0,
            spec_.n_A_,
            gpu_data_temp,
            0,
            spec_.n_temp_,
            gpu_data_C,
            stream
        );
    }

    Eigen::Vector<Number, Eigen::Dynamic> run_host_kernel(
        const Eigen::Map<Eigen::Vector<Number, Eigen::Dynamic>>& A
    ) {
        // Compute cumulative max for a vector (treat matrix as a flattened vector)
        Eigen::Vector<Number, Eigen::Dynamic> result(1);
        Number accu = A(0);
        result(0) = accu;
        for (long i = 1; i < A.size(); ++i) {
            accu = Operation::apply(A(i),accu);
        }
        result(0) = accu;
        return result;
    }
};

static_assert(Check_vector_kernel_1In_1Out_template<Vector_reduction_recursive_kernel>::check_passed, "Vector_reduction_recursive_kernel is not a valid kernel template");
