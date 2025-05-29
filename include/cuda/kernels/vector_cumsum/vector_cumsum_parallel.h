// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/vector_cumsum_parallel.h

#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

constexpr static unsigned int MAX_BLOCK_SIZE = 1024;
constexpr static unsigned int WARP_SIZE = 32;
constexpr static unsigned int MAX_N_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;
constexpr static unsigned int LAST_LANE = WARP_SIZE - 1;

template <CUDA_floating_point CUDA_FLOAT>
__global__ void vector_cumsum_by_blocks_parallel(
    const CUDA_FLOAT* A,
    CUDA_FLOAT* C,
    const unsigned int n  // size of vector
) {
    __shared__ CUDA_FLOAT shm[MAX_N_WARPS]; // for writing, index this using `wid_block` (warp id)

    // const unsigned int n_blocks = gridDim.x;

    // tid_xxx represent the index of the thread within grid, block, and warp
    // const unsigned int bid_grid = blockIdx.x;
    // const unsigned short tid_block = threadIdx.x;
    const unsigned int tid_grid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("tid_grid: %d, n: %d, blockDim.x: %d, gridDim.x: %d\n", tid_grid, n, blockDim.x, gridDim.x);

    CUDA_FLOAT value = 0;

    // bid_grid (block ID relative to the whole grid) can be >= n_blocks when we call ourselves
    // recursively on a reduced dataset. In this case, we can skip directliy to the synchronization,
    const unsigned short tid_warp = threadIdx.x % WARP_SIZE;
    const unsigned short wid_block = threadIdx.x / WARP_SIZE;
    const unsigned short n_warps_per_block = blockDim.x / WARP_SIZE;

    // the size of shm must be at least the number of warps in the block

    // Load data, only if index is within bounds
    if (tid_grid < n) {
        value = A[tid_grid];
    }

    // Scan warp using warp-shuffle
    // Each warp produces a scan of 2*WARP_SIZE values
    // (stored in the `value` register of each of the threads of the warp)
    // Initialize: subtree_size = 1, subtree_index = tid_warp (lane)
    // At each step:
    //   shufl value from highest lane of previous subtree: from_lane = std::max(0, (subtree_id-1) * subtree_size)
    //   if (subtree_id % 2 == 1): add shuffled value to local value, offset*=2
    //   all threads: double scanned_size, halve subtree_id
    // e.g.
    // 1  2  3  4   5  6  7   8  9  10 11 12 13 14   15  16
    //   \|    \|     \|     \|    \|    \|    \|       \|
    // 1  3  3  7   5 11  7  15  9  19 11 23 13 27   15  31
    //     \-\-\|      \--\--\|      \-\-\|      \---\--\|
    // 1  3  6 10   5 11  18 26  9  19 30 42 13 27   42  58
    //           \--\--\--\--\|            \--\--\---\--\|
    // 1  3  6 10   15 21 28 36  9  19 30 42 55 69   84  100
    //                        \-\---\--\--\--\--\----\--\|
    // 1  3  6 10   15 21 28 36 45  55 66 78 91 105  120 136

    // e.g.
    // 0  1  2  3   4  5  6   7  8   9 10 11 12 13   14  15
    //   \|    \|     \|     \|    \|    \|    \|       \|
    // 0  1  2  5   4  9  6  13  8  17 10 21 12 25   14  29
    //     \-\-\|      \--\--\|      \-\-\|      \---\--\|
    // 0  1  3  6   4  9  15 22  8  17 27 38 12 25   39  54
    //           \--\--\--\--\|            \--\--\---\--\|
    // 0  1  3  6   10 15 21 28  8  17 27 38 50 63   77  92
    //                        \-\---\--\--\--\--\----\--\|
    // 0  1  3  6   10 15 21 28 36  45 55 66 78 91  105 120


    for (int subtree_size = 1, subtree_id = tid_warp;
            subtree_size < WARP_SIZE;
            subtree_size <<= 1, subtree_id /= 2) {
        const int from_lane = max(0, subtree_id * subtree_size - 1);
        const CUDA_FLOAT received_value = __shfl_sync(0xFFFFFFFF, value, from_lane);
        if (subtree_id % 2 == 1) {
            value += received_value;
        }
    }

    if (n_warps_per_block == 1) {
        // Only one warp. We have already compute the result. We just need to copy it to C
        C[tid_grid] = value;
    } else {
        // n_warps_per_block > 1
        // We need to do a warp-shuffle scan across the warps. We will use shared memory
        // to allow each warp within a block to send it terminal value to the master warp
        // which will then perform the scan.

        // Now each warp's values contain the cumsums for the warp. The last lane of the warp contains the warp total.
        if (tid_warp == LAST_LANE) {
            shm[wid_block] = value;
        }

        __syncthreads(); // Now shm contains input data for the block-level warp-shuffle scan

        // Now the warp totals live in shm. We need to scan shm to compute
        // We use the same algorithm as above, but we execute with only one warp,
        // as the shared memory size is equal to warpSize (1024/32 == 32)
        static_assert(MAX_N_WARPS == WARP_SIZE, "MAX_N_WARPS != WARP_SIZE (at compile time)");
        CUDA_FLOAT shm_value = 0;
        if (wid_block == 0) {
            // We pick warp 0 to perform the warp-shuffle scan on shared memory.
            if (tid_warp < n_warps_per_block) {
                shm_value = shm[tid_warp];
            }
            for (int subtree_size = 1, subtree_id = tid_warp;
                subtree_size < WARP_SIZE;
                subtree_size <<= 1, subtree_id /= 2) {
                const int from_lane = max(0, subtree_id * subtree_size - 1);
                const CUDA_FLOAT received_value = __shfl_sync(0xFFFFFFFF, shm_value, from_lane);
                if (subtree_id % 2 == 1) {
                    shm_value += received_value;
                }
            }
            shm[tid_warp] = shm_value;
        }

        __syncthreads(); // Now shm contains output data from the block-level warp-shuffle scan

        // We need to read shm into all the warps, and let each lane update itself based on the
        // difference between the value written by the warp to shm and the value read back in.
        CUDA_FLOAT warp_delta_value = 0;
        if (tid_warp == LAST_LANE) {
            const CUDA_FLOAT updated_value = shm[wid_block];
            warp_delta_value = updated_value - value;
            value = updated_value; // only for the last lane!
        }
        warp_delta_value = __shfl_sync(0xFFFFFFFF, warp_delta_value, LAST_LANE);
        if (tid_warp != LAST_LANE) {
            // All the other lanes
            value += warp_delta_value;
        }

        // Now the `value` variables contain the the scanned values: we need to write them back to C
        C[tid_grid] = value;
    }
}

struct Vector_cumsum_parallel_spec {
    const cudaDeviceProp device_prop_;

    const std::string type_;

    const unsigned int m_;    // unused for vector cumsum
    const unsigned int n_;    // size of vector
    const unsigned int k_;    // unused for vector cumsum

    const unsigned int n_rows_A_;
    const unsigned int n_cols_A_;

    const unsigned int n_rows_C_;
    const unsigned int n_cols_C_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 0;    // unused
    constexpr static int DEFAULT_N = 3000; // size of vector
    constexpr static int DEFAULT_K = 0;    // unused
    constexpr static int DEFAULT_BLOCK_DIM_X = 1024;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("n", "Size of vector", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N)))
            ("block_dim,x", "Number of threads in the x dimension per block", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("type", "Numeric type (half, single/float, double)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Vector_cumsum_parallel_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        const auto n = options_parsed["n"].as<int>();
        const auto block_dim_option = options_parsed["block_dim"].as<int>();
        const auto block_size = (std::min(n, block_dim_option)  + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        return Vector_cumsum_parallel_spec(
            type,
            n,
            block_size
        );
    }

    inline Vector_cumsum_parallel_spec(
        const std::string& type,
        const unsigned int n,
        const unsigned int block_size
    ) : device_prop_(get_default_device_prop()),
        type_(type),
        m_(0),  // unused
        n_(n),
        k_(0),  // unused
        n_rows_A_(1),
        n_cols_A_(n),
        n_rows_C_(1),
        n_cols_C_(n),
        block_dim_(block_size),
        grid_dim_((n + block_size - 1) / block_size)
    {}
};

static_assert(Check_kernel_spec_1In_1Out<Vector_cumsum_parallel_spec>::check_passed, "Vector_cumsum_parallel_spec is not a valid kernel spec");


template <CUDA_floating_point Number_>
class Vector_cumsum_parallel_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Vector_cumsum_parallel_spec;

    const Kernel_spec spec_;
    dim3 block_dim_;
    dim3 grid_dim_;

    Vector_cumsum_parallel_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,
        Number* const gpu_data_C,
        cudaStream_t stream
    ) {
        int max_block_size = 0;
        int opt_grid_size = 0;
        int max_active_blocks_per_multiprocessor = 0;
        cudaOccupancyMaxPotentialBlockSize(
            &max_block_size,
            &opt_grid_size,
            vector_cumsum_by_blocks_parallel<Number>,
            0,
            0
        );
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_multiprocessor,
            vector_cumsum_by_blocks_parallel<Number>,
            max_block_size,
            0
        );
        // block_dim_ = dim3(max_block_size);
        // grid_dim_ = dim3(opt_grid_size);

        std::cout << "[INFO] max_active_blocks_per_multiprocessor: " << max_active_blocks_per_multiprocessor
                  << "at block_size:" << max_block_size << std::endl;
        std::cout << "[INFO] max_block_size: " << max_block_size << std::endl;
        std::cout << "[INFO] opt_grid_size: " << opt_grid_size << std::endl;
        std::cout << "[INFO] grid_dim_: " << spec_.grid_dim_.x << ", " << spec_.grid_dim_.y << ", " << spec_.grid_dim_.z << std::endl;
        std::cout << "[INFO] block_dim_: " << spec_.block_dim_.x << ", " << spec_.block_dim_.y << ", " << spec_.block_dim_.z << std::endl;
        vector_cumsum_by_blocks_parallel<<<spec_.grid_dim_, spec_.block_dim_, 0, stream>>>(
            gpu_data_A,
            gpu_data_C,
            spec_.n_
        );
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A
    ) {
        // Compute cumulative sum for a vector (treat matrix as a flattened vector)
        Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(A.rows(), A.cols());
        Number sum = 0;
        for (int i = 0; i < A.size(); ++i) {
            sum += A(i);
            result(i) = sum;
        }
        return result;
    }
};
static_assert(Check_kernel_1In_1Out_template<Vector_cumsum_parallel_kernel>::check_passed, "Vector_cumsum_parallel is not a valid kernel template");
