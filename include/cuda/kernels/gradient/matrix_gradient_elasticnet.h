// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/matrix/matrix_gradient_elasticnet.h

#pragma once
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "cuda/kernel_api.h"
#include "cuda/type_traits.h"

/*
This kernel computes the gradient of the loss of a linear regression
with a custom loss function and an elastic net regularization term.

There are 3 inputs: A, B, and C. The output is M.

We interpret A as the matrix (a tensor really) of observed independent variables, B as the matrix of
observed dependent variables. Both tensors contain nrow observations.

C is the model coefficient matrix, such that

B = <A, C> + E

whre <A, C> denotes the inner product between tensor A and matrix C, producing a matrix result.

The loss function is:

L_[alpha, lambda](C) = SUM_i SUM_j E(i,j)^2 + ElasticNet_[alpha, lambda](C)

We want to compute the partial derivatives of L with respect to the elements of C.

The gradient is:
∇_C L = -2A^T(B - AC) + α·sign(C) + 2λC
*/

template <CUDA_scalar CUDA_Number>
__global__ void matrix_gradient_elasticnet(
    const CUDA_Number* A,     // m x n matrix (independent variables)
    const CUDA_Number* B,     // m x k matrix (dependent variables)
    const CUDA_Number* C,     // n x k matrix (coefficients)
    CUDA_Number* M,           // n x k matrix (gradient output)
    const long m,             // number of observations
    const long n,             // number of features
    const long k,             // number of outputs
    const CUDA_Number alpha,  // L1 regularization parameter
    const CUDA_Number lambda  // L2 regularization parameter
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // coefficient matrix column (output dimension)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // coefficient matrix row (feature dimension)

    if (row < n && col < k) {
        // Compute gradient for coefficient C[row,col]
        CUDA_Number grad = 0.0;

        // Compute residual term: -2 * A^T * (B - A*C)
        // For element (row,col), this is: -2 * sum_i A[i,row] * (B[i,col] - sum_j A[i,j]*C[j,col])
        for (int i = 0; i < m; ++i) {
            // Compute prediction for observation i, output col: sum_j A[i,j]*C[j,col]
            CUDA_Number prediction = 0.0;
            for (int j = 0; j < n; ++j) {
                prediction += A[i * n + j] * C[j * k + col];
            }

            // Compute residual: B[i,col] - prediction
            CUDA_Number residual = B[i * k + col] - prediction;

            // Add contribution to gradient: -2 * A[i,row] * residual
            grad -= CUDA_Number(2) * A[i * n + row] * residual;
        }

        // Add L1 regularization term: α * sign(C[row,col])
        CUDA_Number c_val = C[row * k + col];
        const CUDA_Number zero = CUDA_Number(0);
        if constexpr (std::is_unsigned_v<CUDA_Number>) {
            grad += alpha * (c_val > zero ? 1 : zero);
        } else if (c_val > zero) {
            grad += alpha;
        } else if (c_val < zero) {
            grad -= alpha;
        }
        // If c_val == 0, sign is undefined, so we use subgradient (can be any value in [-α, α])
        // We choose 0 for simplicity

        // Add L2 regularization term: 2λ * C[row,col]
        grad += CUDA_Number(2) * lambda * c_val;

        // Store result
        M[row * k + col] = grad;
    }
}

struct Matrix_gradient_elasticnet_spec {
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

    const long n_rows_temp_;
    const long n_cols_temp_;

    const double alpha_;  // L1 regularization parameter
    const double lambda_; // L2 regularization parameter

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;

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

    inline static Matrix_gradient_elasticnet_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return Matrix_gradient_elasticnet_spec(
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

    inline Matrix_gradient_elasticnet_spec(
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
        n_rows_temp_(0),
        n_cols_temp_(0),
        alpha_(alpha),
        lambda_(lambda),
        block_dim_(block_dim_x, block_dim_y),
        grid_dim_(
            (k_ + block_dim_.x - 1) / block_dim_.x,  // columns of output (k)
            (n_ + block_dim_.y - 1) / block_dim_.y   // rows of output (n)
        )
    {}
};

static_assert(Check_matrix_kernel_spec_3In_1Out<Matrix_gradient_elasticnet_spec>::check_passed, "Matrix_gradient_elasticnet_spec is not a valid 3In1Out kernel spec");

template <CUDA_scalar Number_>
class Matrix_gradient_elasticnet_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Matrix_gradient_elasticnet_spec;

    const Kernel_spec spec_;

    Matrix_gradient_elasticnet_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_A,  // independent variables (m x n)
        const Number* const gpu_data_B,  // dependent variables (m x k)
        const Number* const gpu_data_C,  // coefficients (n x k)
        Number* const gpu_data_D,        // gradient output (n x k)
        Number* const gpu_data_temp,     // temporary storage (unused)
        cudaStream_t stream
    ) {
        matrix_gradient_elasticnet<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(gpu_data_A, gpu_data_B, gpu_data_C, gpu_data_D,
            spec_.m_, spec_.n_, spec_.k_,
            Number(spec_.alpha_), Number(spec_.lambda_));
    }

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& A,  // m x n
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& B,  // m x k
        const Eigen::Map<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& C   // n x k
    ) {
        // Compute prediction: A * C (m x k)
        auto predictions = A * C;

        // Compute residuals: B - A*C (m x k)
        auto residuals = B - predictions;

        // Compute gradient: -2 * A^T * residuals (n x k)
        auto grad_data_term = Number(-2) * A.transpose() * residuals;

        // Add L1 regularization: α * sign(C)
        auto grad_l1 = Number(spec_.alpha_) * C.unaryExpr([](const Number& x) -> Number {
            Number zero = Number(0);
            if (x > zero) return Number(1);
            else if (x < zero) return Number(-1);
            else return zero;  // subgradient choice for x=0
        });

        // Add L2 regularization: 2λ * C
        auto grad_l2 = Number(2) * Number(spec_.lambda_) * C;

        // Total gradient
        return (grad_data_term + grad_l1 + grad_l2).eval();
    }
};

static_assert(Check_matrix_kernel_3In_1Out_template<Matrix_gradient_elasticnet_kernel>::check_passed, "Matrix_gradient_elasticnet is not a valid 3In1Out kernel template");
