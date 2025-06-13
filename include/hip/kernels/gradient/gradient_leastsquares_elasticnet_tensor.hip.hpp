// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/matrix/gradient_leastsquares_elasticnet_tensor.hpp

#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp16.h>
#include <cxxopts.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "hip/kernel_api/matrix_3in_1out.hip.hpp"
#include "hip/type_traits.hip.hpp"
#include "hip/type_traits.hip.hpp"

/*
This kernel computes the gradient of the loss of a linear regression
with a custom loss function and an elastic net regularization term.

There are 3 inputs: A, B, and M. The output is ∇_M L.

We interpret A as the matrix (a tensor really) of observed independent variables, B as the matrix of
observed dependent variables. Both tensors contain nrow observations.

M is the model coefficient matrix, such that

B = <A, M> + E

whre <A, M> denotes the inner product between tensor A and matrix M, producing a matrix result.

The loss function is:

L_[alpha, lambda](M) = SUM_i SUM_j E(i,j)^2 + ElasticNet_[alpha, lambda](M)

We want to compute the partial derivatives of L with respect to the elements of M.

The gradient is:
∇_M L = -2A^T(B - AM) + α·sign(M) + 2λM
*/

// Device function for tensor core matrix multiplication
template <typename Number>
__device__ void wmma_matrix_multiply_device(
    const Number* A,
    const Number* B,
    Number* M,
    const int m,
    const int n,
    const int k,
    const int lda,
    const int ldb,
    const int ldc
) {
    // WMMA fragment dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Calculate which 16x16 tile this warp handles
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int row_base = block_row * WMMA_M;
    const int col_base = block_col * WMMA_N;

    if constexpr (std::is_same_v<Number, __half>) {
        using namespace rocwmma;

        if (warpId == 0) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

            fill_fragment(c_frag, Number(0.0));

            for (int i = 0; i < n; i += WMMA_K) {
                if (row_base < m && col_base < k &&
                    row_base + WMMA_M <= m && col_base + WMMA_N <= k &&
                    i + WMMA_K <= n) {

                    load_matrix_sync(a_frag, A + row_base * lda + i, lda);
                    load_matrix_sync(b_frag, B + i * ldb + col_base, ldb);
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }

            if (row_base < m && col_base < k &&
                row_base + WMMA_M <= m && col_base + WMMA_N <= k) {
                store_matrix_sync(M + row_base * ldc + col_base, c_frag, ldc, mem_row_major);
            }
        }
    } else {
        // Fallback for non-tensor core types
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < k) {
            Number sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += A[row * lda + i] * B[i * ldb + col];
            }
            M[row * ldc + col] = sum;
        }
    }
}

// Optimized gradient kernel using tensor operations and efficient parallelization
template <HIP_scalar HIP_Number>
__global__ void gradient_leastsquares_elasticnet_tensor_optimized(
    const HIP_Number* A,         // m x n matrix (independent variables)
    const HIP_Number* B,         // m x k matrix (dependent variables)
    const HIP_Number* M,         // n x k matrix (model coefficients)
    HIP_Number* grad_M,          // n x k matrix (gradient output)
    HIP_Number* temp_AM,         // m x k temporary matrix for A*M
    HIP_Number* temp_residual,   // m x k temporary matrix for residuals
    const long m,                 // number of observations
    const long n,                 // number of features
    const long k,                 // number of outputs
    const HIP_Number alpha,      // L1 regularization parameter
    const HIP_Number lambda      // L2 regularization parameter
) {
    // Strategy: Use multiple phases
    // Phase 1: Compute A*M using tensor cores
    // Phase 2: Compute residuals B - A*M
    // Phase 3: Compute A^T * residuals using tensor cores
    // Phase 4: Add regularization terms

    const int total_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                         (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);

    // Phase 1: Compute A*M -> temp_AM (m x k)
    // Use tensor cores if possible
    if constexpr (std::is_same_v<HIP_Number, __half>) {
        wmma_matrix_multiply_device(A, M, temp_AM, m, n, k, n, k, k);
    } else {
        // Fallback: naive matrix multiplication
        for (int idx = thread_id; idx < m * k; idx += total_threads) {
            int row = idx / k;
            int col = idx % k;

            HIP_Number sum = 0;
            for (int j = 0; j < n; ++j) {
                sum += A[row * n + j] * M[j * k + col];
            }
            temp_AM[row * k + col] = sum;
        }
    }

    __syncthreads();

    // Phase 2: Compute residuals B - A*M -> temp_residual
    for (int idx = thread_id; idx < m * k; idx += total_threads) {
        temp_residual[idx] = B[idx] - temp_AM[idx];
    }

    __syncthreads();

    // Phase 3: Compute A^T * residuals -> data term of gradient
    // A^T is n x m, residuals is m x k, result is n x k
    if constexpr (std::is_same_v<HIP_Number, __half>) {
        // For tensor cores, we need to handle A^T multiplication carefully
        // This is more complex and may need a separate kernel launch
        // For now, use naive approach
        for (int idx = thread_id; idx < n * k; idx += total_threads) {
            int row = idx / k;  // feature index
            int col = idx % k;  // output index

            HIP_Number sum = 0;
            for (int i = 0; i < m; ++i) {
                sum += A[i * n + row] * temp_residual[i * k + col];
            }
            grad_M[row * k + col] = HIP_Number(-2) * sum;
        }
    } else {
        for (int idx = thread_id; idx < n * k; idx += total_threads) {
            int row = idx / k;  // feature index
            int col = idx % k;  // output index

            HIP_Number sum = 0;
            for (int i = 0; i < m; ++i) {
                sum += A[i * n + row] * temp_residual[i * k + col];
            }
            grad_M[row * k + col] = HIP_Number(-2) * sum;
        }
    }

    __syncthreads();

    // Phase 4: Add regularization terms
    for (int idx = thread_id; idx < n * k; idx += total_threads) {
        HIP_Number m_val = M[idx];
        HIP_Number reg_term = 0;

        // L1 regularization: α * sign(M)
        const HIP_Number zero = HIP_Number(0);
        if constexpr (std::is_unsigned_v<HIP_Number>) {
            reg_term += alpha * (m_val > zero ? 1 : zero);
        } else if (m_val > zero) {
            reg_term += alpha;
        } else if (m_val < zero) {
            reg_term -= alpha;
        }

        // L2 regularization: 2λ * M
        reg_term += HIP_Number(2) * lambda * m_val;

        grad_M[idx] += reg_term;
    }
}

// Warp-reduction based gradient kernel for better parallelization
template <HIP_scalar HIP_Number>
__global__ void gradient_leastsquares_elasticnet_tensor_warp_reduce(
    const HIP_Number* A,     // m x n matrix (independent variables)
    const HIP_Number* B,     // m x k matrix (dependent variables)
    const HIP_Number* M,     // n x k matrix (model coefficients)
    HIP_Number* grad_M,      // n x k matrix (gradient output)
    const long m,             // number of observations
    const long n,             // number of features
    const long k,             // number of outputs
    const HIP_Number alpha,  // L1 regularization parameter
    const HIP_Number lambda  // L2 regularization parameter
) {
    // Each block handles one (feature, output) pair
    // Threads within block handle different observations (sheets)
    const int feature_idx = blockIdx.y;
    const int output_idx = blockIdx.x;
    const int obs_stride = blockDim.x * blockDim.y;
    const int thread_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    if (feature_idx >= n || output_idx >= k) return;

    HIP_Number* shm = static_cast<HIP_Number*>(get_dynamic_shared_memory(alignof(HIP_Number)));

    HIP_Number local_sum = 0;

    // Each thread accumulates over a subset of observations
    for (int obs_idx = thread_in_block; obs_idx < m; obs_idx += obs_stride) {
        // Compute prediction for this observation
        HIP_Number prediction = 0;
        for (int j = 0; j < n; ++j) {
            prediction += A[obs_idx * n + j] * M[j * k + output_idx];
        }

        // Compute residual
        HIP_Number residual = B[obs_idx * k + output_idx] - prediction;

        // Accumulate gradient contribution: -2 * A[obs_idx, feature_idx] * residual
        local_sum += A[obs_idx * n + feature_idx] * residual;
    }

    // Warp-level reduction
    const int warp_id = thread_in_block / 32;
    const int lane_id = thread_in_block % 32;

    // Reduce within warp using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(__activemask(), local_sum, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shm[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        const int num_warps = (obs_stride + 31) / 32;
        local_sum = (lane_id < num_warps) ? shm[lane_id] : HIP_Number(0);

        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(__activemask(), local_sum, offset);
        }

        // First thread writes final result
        if (lane_id == 0) {
            HIP_Number grad = HIP_Number(-2) * local_sum;

            // Add regularization terms
            HIP_Number m_val = M[feature_idx * k + output_idx];
            const HIP_Number zero = HIP_Number(0);

            // L1 regularization
            if constexpr (std::is_unsigned_v<HIP_Number>) {
                grad += alpha * (m_val > zero ? 1 : zero);
            } else if (m_val > zero) {
                grad += alpha;
            } else if (m_val < zero) {
                grad -= alpha;
            }

            // L2 regularization
            grad += HIP_Number(2) * lambda * m_val;

            grad_M[feature_idx * k + output_idx] = grad;
        }
    }
}

struct Gradient_leastsquares_elasticnet_tensor_spec {
    const std::string type_;

    const long m_;    // Number of observations
    const long n_;    // Number of features
    const long k_;    // Number of outputs

    const long n_rows_A_;   // m (observations)
    const long n_cols_A_;   // n (features)

    const long n_rows_B_;   // m (observations)
    const long n_cols_B_;   // k (outputs)

    const long n_rows_C_;   // n (features)
    const long n_cols_C_;   // k (outputs)

    const long n_rows_D_;   // n (features) - gradient output
    const long n_cols_D_;   // k (outputs)

    const long n_rows_temp_;  // For temporary matrices
    const long n_cols_temp_;

    const double alpha_;  // L1 regularization parameter
    const double lambda_; // L2 regularization parameter

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 1000;  // Number of observations
    constexpr static int DEFAULT_N = 100;   // Number of features
    constexpr static int DEFAULT_K = 10;    // Number of outputs
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;
    constexpr static double DEFAULT_ALPHA = 0.01;
    constexpr static double DEFAULT_LAMBDA = 0.01;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of outputs", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("alpha", "L1 regularization parameter", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_ALPHA)))
            ("lambda", "L2 regularization parameter", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_LAMBDA)))
            ("block_dim_x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block_dim_y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
    }

    inline static Gradient_leastsquares_elasticnet_tensor_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Gradient_leastsquares_elasticnet_tensor_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["alpha"].as<double>(),
            options_parsed["lambda"].as<double>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Gradient_leastsquares_elasticnet_tensor_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k,
        const double alpha,
        const double lambda,
        const long block_dim_x,
        const long block_dim_y
    ) : type_(type),
        m_(m),
        n_(n),
        k_(k),
        n_rows_A_(m),
        n_cols_A_(n),
        n_rows_B_(m),
        n_cols_B_(k),
        n_rows_C_(n),
        n_cols_C_(k),
        n_rows_D_(n),
        n_cols_D_(k),
        n_rows_temp_(m),  // For temporary matrices (A*M and residuals)
        n_cols_temp_(k),
        alpha_(alpha),
        lambda_(lambda),
        block_dim_(block_dim_x, block_dim_y),
        grid_dim_(
            k_,  // Each block handles one (feature, output) pair - columns are outputs
            n_   // rows are features
        ),
        dynamic_shared_mem_words_((block_dim_x * block_dim_y + 31) / 32)  // Space for warp reduction
    {}
};

static_assert(Check_matrix_kernel_spec_3In_1Out<Gradient_leastsquares_elasticnet_tensor_spec>::check_passed, "Gradient_leastsquares_elasticnet_tensor_spec is not a valid 3In1Out kernel spec");

template <HIP_scalar Number_>
class Gradient_leastsquares_elasticnet_tensor_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Gradient_leastsquares_elasticnet_tensor_spec;

    const Kernel_spec spec_;

    Gradient_leastsquares_elasticnet_tensor_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,  // independent variables (m x n)
        const Number* const gpu_data_B,  // dependent variables (m x k)
        const Number* const gpu_data_M,  // coefficients (n x k)
        Number* const gpu_data_grad_M,  // gradient output (n x k)
        Number* const gpu_data_temp,     // temporary storage (2 * m * k for AM and residuals)
        hipStream_t stream
    ) {
        // Use optimized warp-reduction kernel for better performance
        gradient_leastsquares_elasticnet_tensor_warp_reduce<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_M, gpu_data_grad_M,
            spec_.m_, spec_.n_, spec_.k_,
            Number(spec_.alpha_), Number(spec_.lambda_));
    }

    template<typename Matrix_like_A, typename Matrix_like_B, typename Matrix_like_M>
    requires is_matrix_like<Matrix_like_A> && is_matrix_like<Matrix_like_B> && is_matrix_like<Matrix_like_M> &&
             std::is_same_v<typename Matrix_like_A::Scalar, Number> &&
             std::is_same_v<typename Matrix_like_B::Scalar, Number> &&
             std::is_same_v<typename Matrix_like_M::Scalar, Number>
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Matrix_like_A& A,  // m x n
        const Matrix_like_B& B,  // m x k
        const Matrix_like_M& M   // n x k
    ) {
        // Compute prediction: A * M (m x k)
        auto predictions = A * M;

        // Compute residuals: B - A*M (m x k)
        auto residuals = B - predictions;

        // Compute gradient: -2 * A^T * residuals (n x k)
        auto grad_data_term = Number(-2) * A.transpose() * residuals;

        // Add L1 regularization: α * sign(M)
        auto grad_l1 = Number(spec_.alpha_) * M.unaryExpr([](const Number& x) -> Number {
            Number zero = Number(0);
            if (x > zero) return Number(1);
            else if (x < zero) return Number(-1);
            else return zero;  // subgradient choice for x=0
        });

        // Add L2 regularization: 2λ * M
        auto grad_l2 = Number(2) * Number(spec_.lambda_) * M;

        // Total gradient
        return (grad_data_term + grad_l1 + grad_l2).eval();
    }
};

static_assert(Check_matrix_kernel_3In_1Out_template<Gradient_leastsquares_elasticnet_tensor_kernel>::check_passed, "Gradient_leastsquares_elasticnet_tensor is not a valid 3In1Out kernel template");
