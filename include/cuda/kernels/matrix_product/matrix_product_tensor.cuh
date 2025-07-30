// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_product_tensor.hpp

#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <limits>
#include <mma.h>
#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <string>
#include <tuple>
#include <type_traits>
#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/kernel_api/matrix_2in_1out.cuh"
#include "cuda/type_traits.cuh"
#include "cuda/cuda_utils.cuh"

struct Matrix_product_tensor_spec {
    const std::string type_;

    unsigned m_;    // Rows of first matrix
    unsigned k_;    // Columns of first matrix and rows of second matrix
    unsigned n_;    // Columns of second matrix

    const long n_rows_A_;
    const long n_cols_A_;

    const long n_rows_B_;
    const long n_cols_B_;

    const long n_rows_C_;
    const long n_cols_C_;

    const long n_rows_temp_ = 0;
    const long n_cols_temp_ = 0;

    const unsigned block_dim_tiles_x;
    const unsigned block_dim_tiles_y;

    // TODO: Kernel_spec API should not define block_dim_ and grid_dim_
    const dim3 block_dim_; // Not meaningful here, computed in Kernel class
    const dim3 grid_dim_;  // Not meaningful here, computed in Kernel class
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static int DEFAULT_M = 3000; // Rows of first matrix
    constexpr static int DEFAULT_K = 300;  // Columns of first matrix / Rows of second matrix
    constexpr static int DEFAULT_N = 1000; // Columns of second matrix
    constexpr static int DEFAULT_BLOCK_DIM_WARPS_X = 4;
    constexpr static int DEFAULT_BLOCK_DIM_WARPS_Y = 4;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of rows in first matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("k", "Number of columns in first matrix and rows of the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("n", "Number of columns in the second matrix", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("block-dim-tiles-x,x", "Number of tiles in the x dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_WARPS_X)))
            ("block-dim-tiles-y,y", "Number of tiles in the y dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_WARPS_Y)))
            ("type", "Numeric type (half*, int8*, single/float, double, uint8) (* = tensor cores)", cxxopts::value<std::string>()->default_value("float"));
        ;
    }

    inline static Matrix_product_tensor_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option - now accepts all types supported by the main program
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" &&
            type != "int8" && type != "int16" && type != "int32" && type != "int64" &&
            type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        const auto m = options_parsed["m"].as<long>();
        const auto n = options_parsed["n"].as<long>();
        const auto k = options_parsed["k"].as<long>();
        if (m <= 0) {
            std::cerr << "[ERROR] -m, -n, -k must be positive" << std::endl;
            throw cxxopts::exceptions::exception("Invalid -m: " + m);
        }
        if (n <= 0) {
            std::cerr << "[ERROR] -m, -n, -k must be positive" << std::endl;
            throw cxxopts::exceptions::exception("Invalid -n: " + n);
        }
        if (k <= 0) {
            std::cerr << "[ERROR] -m, -n, -k must be positive" << std::endl;
            throw cxxopts::exceptions::exception("Invalid -k: " + k);
        }
        constexpr auto max_mnk = std::numeric_limits<unsigned>::max();
        if (m > max_mnk) {
            std::cerr << "[ERROR] -m, -n, -k may be at most " << max_mnk << std::endl;
            throw cxxopts::exceptions::exception("Invalid -m: " + m);
        }
        if (n > max_mnk) {
            std::cerr << "[ERROR] -m, -n, -k may be at most " << max_mnk << std::endl;
            throw cxxopts::exceptions::exception("Invalid -n: " + n);
        }
        if (k > max_mnk) {
            std::cerr << "[ERROR] -m, -n, -k may be at most " << max_mnk << std::endl;
            throw cxxopts::exceptions::exception("Invalid -k: " + k);
        }

        return Matrix_product_tensor_spec(
            type,
            (unsigned)m,
            (unsigned)n,
            (unsigned)k,
            options_parsed["block-dim-tiles-x"].as<unsigned>(),
            options_parsed["block-dim-tiles-y"].as<unsigned>()
        );
    }

    inline Matrix_product_tensor_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k,
        const long block_dim_tiles_x,
        const long block_dim_tiles_y
    ) : type_(type),
        m_(m),
        k_(k),
        n_(n),
        n_rows_A_(m),
        n_cols_A_(k),
        n_rows_B_(k),
        n_cols_B_(n),
        n_rows_C_(m),
        n_cols_C_(n),
        block_dim_tiles_x(block_dim_tiles_x),
        block_dim_tiles_y(block_dim_tiles_y),
        block_dim_(0),
        grid_dim_(0)
    {}
};

static_assert(Check_matrix_kernel_spec_2In_1Out<Matrix_product_tensor_spec>::check_passed, "Matrix_product_tensor_spec is not a valid kernel spec");

// Looks like nvcc does not actually enforce this.
constexpr static long wmma_alignment_requirement_bits = 256;
constexpr static long wmma_alignment_requirement_bytes = wmma_alignment_requirement_bits/8;

template <typename T>
struct wmma_config;

template <>
struct wmma_config<std::int8_t> {
    using argument_type = std::int8_t;
    using operand_type = std::int8_t;
    using accumulator_type = int;
    using result_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");
};

template <>
struct wmma_config<std::uint8_t> {
    using argument_type = uint8_t;
    using operand_type = std::uint8_t;
    using accumulator_type = int;
    using result_type = int;
    using temp_type = int;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");
};

template <>
struct wmma_config<__half> {
    using argument_type = __half;
    using operand_type = __half;
    using accumulator_type = __half;
    using result_type = __half;
    using temp_type = __half;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 16;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");
};

template <>
struct wmma_config<float> {
    // tf32 is just a placeholder type declared as a struct without a definition (an incomplete type)
    using argument_type = float;
    using operand_type = nvcuda::wmma::precision::tf32;
    using accumulator_type = float;
    using result_type = float;
    using temp_type = float;
    constexpr static unsigned M = 16;
    constexpr static unsigned N = 16;
    constexpr static unsigned K = 8;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");
};

template <>
struct wmma_config<double> {
    using argument_type = double;
    using operand_type = double;
    using accumulator_type = double;
    using result_type = double;
    using temp_type = double;
    constexpr static unsigned M = 8;
    constexpr static unsigned N = 8;
    constexpr static unsigned K = 4;
    constexpr static unsigned temp_tile_bytes = std::max(std::max(M*K, K*N)*sizeof(argument_type), M*N*sizeof(result_type));

    // static_assert(M*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(K*sizeof(argument_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_a");
    // static_assert(M*sizeof(result_type) % wmma_alignment_requirement_bytes == 0, "alignment requirement not met for matrix_c");
};

template <typename Use, int M, int N, int K, typename T, typename Layout, CUDA_scalar Number>
__device__ inline void safe_load_operand_sync(
    nvcuda::wmma::fragment<Use, M, N, K, T, Layout> &fragment, // M by K, or K by N, or M by N depending on Use
    const Number* const source,
    const unsigned source_nrows,
    const unsigned source_ncols,
    const unsigned tile_id_row,
    const unsigned tile_id_col,
    const auto tid_warp,
    Number* temp_tile
) {
    constexpr nvcuda::wmma::layout_t layout = (
        std::is_same_v<Layout, nvcuda::wmma::row_major> ? nvcuda::wmma::mem_row_major : nvcuda::wmma::mem_col_major
    );
    constexpr auto tile_nrows = (
        std::is_same_v<Use, nvcuda::wmma::matrix_a> ? M
        : std::is_same_v<Use, nvcuda::wmma::matrix_b> ? K
        : M
    );
    constexpr auto tile_ncols = (
        std::is_same_v<Use, nvcuda::wmma::matrix_a> ?  K
        : std::is_same_v<Use, nvcuda::wmma::matrix_b> ? N
        : N
    );
    // Both tile_nrows and tile_ncols are <= warpSize
    // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes:~:text=and%20Matrix%20Sizes-,%EF%83%81,-Tensor%20Cores%20support

    const auto source_row0 = (long)tile_id_row * tile_nrows;
    const auto source_col0 = (long)tile_id_col * tile_ncols;

    const auto matrix_width = (
        layout == nvcuda::wmma::mem_row_major
        ? source_ncols
        : source_nrows
    );

    // Partial or misaligned tile: we need to copy it to temp memory, then load from temp memory
    // Let's coalesce memory by accessing contiguous memory. We must account for Layout.
    const auto [copy_width, copy_depth, matrix_depth] = (
        layout == nvcuda::wmma::mem_row_major
        ? std::make_tuple(tile_ncols, tile_nrows, source_nrows)
        : std::make_tuple(tile_nrows, tile_ncols, source_ncols)
    );

    const auto depth_per_warp = warpSize / copy_width;

    const auto col_in_tile = tid_warp % copy_width;  // inner loop variable (width)
    const auto row_in_tile0 = tid_warp / copy_width; // starting point for outer loop variable (depth)

    const auto col_in_matrix = source_col0 + col_in_tile;
    if (col_in_matrix < source_ncols) {
        for (auto row_in_tile = row_in_tile0; row_in_tile < tile_nrows; row_in_tile += depth_per_warp) {
            const auto row_in_matrix = source_row0 + row_in_tile;
            const auto offset_in_matrix = row_in_matrix * matrix_width + col_in_matrix;
            const auto offset_in_tile = row_in_tile * copy_width + col_in_tile;
            const Number zero(0);
            // printf("[w%d] Reading from %d -> %d, row_in_matrix=%d, col_in_matrix=%d\n",
            //     tid_warp, offset_in_matrix, offset_in_tile, row_in_matrix, col_in_matrix
            // );
            temp_tile[offset_in_tile] = (
                (col_in_matrix < matrix_width && row_in_matrix < matrix_depth)
                ? source[offset_in_matrix]
                : zero
            );
        }
    }
    // Now the partial tile has been copied to temp_tile, so we can call load_matrix_sync
    nvcuda::wmma::load_matrix_sync(fragment, temp_tile, tile_ncols);
}

template <int M, int N, int K, typename T, CUDA_scalar Number>
__device__ inline void safe_store_accumulator_sync(
    Number* const dest,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T, /* Layout = */ void> &fragment, // M by N
    const unsigned dest_nrows,
    const unsigned dest_ncols,
    const unsigned tile_id_row,
    const unsigned tile_id_col,
    const auto tid_warp,
    Number* temp_tile
) {
    constexpr nvcuda::wmma::layout_t layout = nvcuda::wmma::mem_row_major;
    constexpr auto tile_nrows = M;
    constexpr auto tile_ncols = N;
    // Both tile_nrows and tile_ncols are <= warpSize
    // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes:~:text=and%20Matrix%20Sizes-,%EF%83%81,-Tensor%20Cores%20support

    const auto dest_row0 = (long)tile_id_row * tile_nrows;
    const auto dest_col0 = (long)tile_id_col * tile_ncols;
    if (!(dest_row0 < dest_nrows && dest_col0 < dest_ncols)) {
        assert(false);
    }

    const auto matrix_width = (
        layout == nvcuda::wmma::mem_row_major
        ? dest_ncols
        : dest_nrows
    );


    // Partial or misaligned tile: we need to copy it to temp memory, then load from temp memory
    // Let's coalesce memory by accessing contiguous memory. We must account for Layout.
    const auto [copy_width, copy_depth, matrix_depth] = (
        layout == nvcuda::wmma::mem_row_major
        ? std::make_tuple(tile_ncols, tile_nrows, dest_nrows)
        : std::make_tuple(tile_nrows, tile_ncols, dest_ncols)
    );

    const auto depth_per_warp = warpSize / copy_width;

    const auto col_in_tile = tid_warp % copy_width;  // inner loop variable (width)
    const auto row_in_tile0 = tid_warp / copy_width; // starting point for outer loop variable (depth)

    nvcuda::wmma::store_matrix_sync(temp_tile, fragment, tile_ncols, layout);
    // Now the partial tile has been copied to temp_tile. We have to manually copy it to dest

    const auto col_in_matrix = dest_col0 + col_in_tile;
    if (col_in_matrix < dest_ncols) {
        for (auto row_in_tile = row_in_tile0; row_in_tile < tile_nrows; row_in_tile += depth_per_warp) {
            const auto row_in_matrix = dest_row0 + row_in_tile;
            const auto offset_in_matrix = row_in_matrix * matrix_width + col_in_matrix;
            const auto offset_in_tile = row_in_tile * copy_width + col_in_tile;
            if (col_in_matrix < matrix_width && row_in_matrix < matrix_depth) {
                dest[offset_in_matrix] = temp_tile[offset_in_tile];
            }
        }
    }
}

// WMMA tensor core matrix multiplication kernel
template <typename Number, typename Wmma_config = wmma_config<Number>>
__global__ void matrix_product_tensor_wmma(
    const typename Wmma_config::argument_type* const A, // m by k
    const typename Wmma_config::argument_type* const B, // k by n
    typename Wmma_config::result_type* const C,         // m by n
    const int m,
    const int k,
    const int n
) {
    extern __shared__ char temp_tile[];
    // Block is 3D: warp lane, tile_col in block, tile_row in block
    // Grid is 2D: block col in grid, block row in grid
    assert(blockDim.x == warpSize);
    const unsigned tid_warp = threadIdx.x;
    constexpr int M = Wmma_config::M;
    constexpr int N = Wmma_config::N;
    constexpr int K = Wmma_config::K;

    // Calculate which tile this warp handles
    const int x_wid_grid = blockIdx.x * blockDim.y + threadIdx.y;
    const int y_wid_grid = blockIdx.y * blockDim.z + threadIdx.z;
    const int wid_in_block = threadIdx.z * blockDim.y + threadIdx.y;

    if (x_wid_grid * N >= n || y_wid_grid * M >= m) {
        return;
    }

    // Declare WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, typename Wmma_config::operand_type, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, typename Wmma_config::operand_type, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, typename Wmma_config::accumulator_type> c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0);


    // Loop over inner dimension one tile at a time
    const auto k_tiles = (k + K - 1) / K;
    for (int k_tile_i = 0; k_tile_i < k_tiles; ++k_tile_i) {
        // Load fragments from input arrays with bounds checking
        using namespace nvcuda::wmma;
        auto argument_tile = (typename Wmma_config::argument_type*)(temp_tile + wid_in_block * Wmma_config::temp_tile_bytes);
        safe_load_operand_sync<matrix_a, M, N, K, typename Wmma_config::operand_type, row_major, typename Wmma_config::argument_type>(
            a_frag, A, m, k, y_wid_grid, k_tile_i, tid_warp, argument_tile
        );
        __syncthreads();
        safe_load_operand_sync<matrix_b, M, N, K, typename Wmma_config::operand_type, row_major, typename Wmma_config::argument_type>(
            b_frag, B, k, n, k_tile_i, x_wid_grid, tid_warp, argument_tile
        );

        // Perform tensor core matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // Store accumulator to result array with bounds checking
    auto result_tile = (typename Wmma_config::result_type*)(temp_tile + wid_in_block * Wmma_config::temp_tile_bytes);
    safe_store_accumulator_sync(C, c_frag, m, n, y_wid_grid, x_wid_grid, tid_warp, result_tile);
}

template <CUDA_scalar Number_>
class Matrix_product_tensor_kernel {
    public:
    using Number = Number_;
    using Wmma_config = wmma_config<Number>;
    using NumberA = typename Wmma_config::argument_type;
    using NumberB = typename Wmma_config::argument_type;
    using NumberC = typename Wmma_config::result_type;
    using NumberInternal = typename Wmma_config::operand_type;

    using Kernel_spec = Matrix_product_tensor_spec;

    const Kernel_spec spec_;
    Matrix_product_tensor_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const NumberA* const gpu_data_A,
        const NumberB* const gpu_data_B,
        NumberC*       const gpu_data_C,
        Number*        const gpu_data_temp,
        cudaStream_t stream
    ) {
        const auto warp_size = get_warp_size();
        // We need one warp per destination tile
        const auto c_ncols = spec_.n_, c_nrows = spec_.m_;
        const auto c_ncol_tiles = (c_ncols + Wmma_config::N - 1) / Wmma_config::N;
        const auto c_nrow_tiles = (c_nrows + Wmma_config::M - 1) / Wmma_config::M;
        const auto c_ncol_blocks = (c_ncol_tiles + spec_.block_dim_tiles_x - 1) / spec_.block_dim_tiles_x;
        const auto c_nrow_blocks = (c_nrow_tiles + spec_.block_dim_tiles_y - 1) / spec_.block_dim_tiles_y;

        const dim3 block_dim{warp_size, spec_.block_dim_tiles_x, spec_.block_dim_tiles_y};
        const dim3 grid_dim{c_ncol_blocks, c_nrow_blocks};
        const auto n_warps_per_block = spec_.block_dim_tiles_x * spec_.block_dim_tiles_y;
        const auto shared_memory_size = n_warps_per_block * Wmma_config::temp_tile_bytes;

        std::cout << "[INFO] c_nrow_tiles: " << c_nrow_tiles << " c_ncol_tiles: " << c_ncol_tiles << std::endl;
        std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
        std::cout << "[INFO] block_dim_: " << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << std::endl;
        std::cout << "[INFO] kernel launch: matrix_product_tensor_wmma" << std::endl;

        // Launch tensor core matrix multiplication kernel
        matrix_product_tensor_wmma<Number><<<
            grid_dim,
            block_dim,
            shared_memory_size,
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_C, spec_.m_, spec_.k_, spec_.n_);
    }
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<NumberA, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,
        const Eigen::Map<Eigen::Matrix<NumberB, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B
    ) {
        return (A * B).eval();
    }

};
static_assert(Check_matrix_kernel_2In_1Out_template<Matrix_product_tensor_kernel>::check_passed, "Matrix_product_tensor is not a valid kernel template");
