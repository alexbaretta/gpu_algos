// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/hip/kernels/gradient/gradient_leastsquares_elasticnet_optimizer.hip.hpp

#pragma once
#include <hip/hip_runtime.h>
#include <mma.h>
#include <hip_fp16.h>
#include <cxxopts.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "hip/kernel_api/matrix_3in_1out.hpp"
#include "hip/type_traits.hpp"
#include "common/type_traits.hpp"
#include "hip/kernels/gradient/gradient_leastsquares_elasticnet_tensor.hpp"

/*
This kernel performs gradient descent optimization for linear regression
with elastic net regularization using line search.

The optimization process:
1. Compute gradient using gradient_leastsquares_elasticnet_tensor
2. Perform line search along negative gradient direction
3. Line search explores n_points_line_search quadratically spaced points:
   M_candidate = M_i - eta * |M_i|_2 * j^2 * gradient_direction
   where j ranges from 1 to n_points_line_search
4. Select M that minimizes loss among candidates
5. Repeat until convergence

The loss function is:
L_[alpha, lambda](M) = SUM_i SUM_j E(i,j)^2 + ElasticNet_[alpha, lambda](M)
where E = B - A*M (residuals)
*/

// Device function to compute loss for a given M
template <typename Number>
__device__ Number compute_loss_device(
    const Number* A,          // m x n matrix
    const Number* B,          // m x k matrix
    const Number* M,          // n x k matrix
    const long m,             // observations
    const long n,             // features
    const long k,             // outputs
    const Number alpha,       // L1 parameter
    const Number lambda       // L2 parameter
) {
    const int total_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                         (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);

    Number local_loss = 0;

    // Compute squared error term
    for (int obs_idx = thread_id; obs_idx < m; obs_idx += total_threads) {
        for (int out_idx = 0; out_idx < k; ++out_idx) {
            // Compute prediction for this (observation, output) pair
            Number prediction = 0;
            for (int feat_idx = 0; feat_idx < n; ++feat_idx) {
                prediction += A[obs_idx * n + feat_idx] * M[feat_idx * k + out_idx];
            }

            // Compute squared residual
            Number residual = B[obs_idx * k + out_idx] - prediction;
            local_loss += residual * residual;
        }
    }

    // Reduce across threads (simplified - assumes single block)
    // In practice, would need proper block-level reduction
    __syncthreads();

    // Add regularization terms (computed by thread 0 only)
    if (thread_id == 0) {
        for (int idx = 0; idx < n * k; ++idx) {
            Number m_val = M[idx];

            // L1 regularization: α * |M|
            local_loss += alpha * (m_val >= 0 ? m_val : -m_val);

            // L2 regularization: λ * M^2
            local_loss += lambda * m_val * m_val;
        }
    }

    return local_loss;
}

// Kernel to compute Frobenius norm of a matrix
template <typename Number>
__global__ void compute_frobenius_norm(
    const Number* M,
    Number* norm_squared,
    const long size
) {
    const int total_threads = blockDim.x * gridDim.x;
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    Number local_sum = 0;

    for (int idx = thread_id; idx < size; idx += total_threads) {
        Number val = M[idx];
        local_sum += val * val;
    }

    // Simplified reduction - in practice would use proper warp/block reduction
    atomicAdd(norm_squared, local_sum);
}

// Line search kernel
template <typename Number>
__global__ void line_search_kernel(
    const Number* A,                    // m x n matrix
    const Number* B,                    // m x k matrix
    const Number* M_current,            // n x k current coefficients
    const Number* gradient,             // n x k gradient
    Number* M_candidates,               // n_points * n * k candidate matrices
    Number* losses,                     // n_points loss values
    const Number frobenius_norm,       // |M_current|_2
    const Number eta,                   // learning rate
    const long m, const long n, const long k,
    const int n_points_line_search,
    const Number alpha, const Number lambda
) {
    const int point_idx = blockIdx.x;  // Which line search point

    if (point_idx >= n_points_line_search) return;

    const int total_threads = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // Compute step size for this point: eta * frobenius_norm * (point_idx + 1)^2
    Number step_size = eta * frobenius_norm * (point_idx + 1) * (point_idx + 1);

    // Compute candidate M: M_current - step_size * gradient
    Number* M_candidate = M_candidates + point_idx * n * k;

    for (int idx = thread_id; idx < n * k; idx += total_threads) {
        M_candidate[idx] = M_current[idx] - step_size * gradient[idx];
    }

    __syncthreads();

    // Compute loss for this candidate (simplified - would need proper reduction)
    if (thread_id == 0) {
        losses[point_idx] = compute_loss_device(A, B, M_candidate, m, n, k, alpha, lambda);
    }
}

// Main optimization kernel
template <HIP_scalar HIP_Number>
__global__ void gradient_leastsquares_elasticnet_optimizer(
    const HIP_Number* A,               // m x n matrix (independent variables)
    const HIP_Number* B,               // m x k matrix (dependent variables)
    HIP_Number* M,                     // n x k matrix (coefficients - input/output)
    HIP_Number* temp_gradient,         // n x k temporary gradient storage
    HIP_Number* temp_candidates,       // n_points * n * k temporary candidates
    HIP_Number* temp_losses,           // n_points temporary loss values
    HIP_Number* temp_norm,             // temporary norm storage
    const long m,                       // number of observations
    const long n,                       // number of features
    const long k,                       // number of outputs
    const int max_iterations,           // maximum optimization iterations
    const int n_points_line_search,     // number of line search points
    const HIP_Number eta,              // learning rate
    const HIP_Number alpha,            // L1 regularization parameter
    const HIP_Number lambda,           // L2 regularization parameter
    const HIP_Number tol,              // absolute tolerance
    const HIP_Number rtol              // relative tolerance
) {
    // This would be the main optimization loop
    // In practice, this might be better implemented as separate kernel launches
    // for better control and debugging

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Step 1: Compute gradient (would call gradient kernel)
        // Step 2: Compute Frobenius norm
        // Step 3: Perform line search
        // Step 4: Update M with best candidate
        // Step 5: Check convergence

        // Placeholder - actual implementation would be more complex
        break;
    }
}

struct Gradient_leastsquares_elasticnet_optimizer_spec {
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

    const long n_rows_D_;   // n (features) - optimized coefficients output
    const long n_cols_D_;   // k (outputs)

    const long n_rows_temp_;  // For temporary storage
    const long n_cols_temp_;

    const int max_iterations_;        // Maximum optimization iterations
    const int n_points_line_search_;  // Number of line search points
    const double eta_;                // Learning rate
    const double alpha_;              // L1 regularization parameter
    const double lambda_;             // L2 regularization parameter
    const double tol_;                // Absolute tolerance
    const double rtol_;               // Relative tolerance

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static int DEFAULT_M = 1000;
    constexpr static int DEFAULT_N = 100;
    constexpr static int DEFAULT_K = 10;
    constexpr static int DEFAULT_MAX_ITERATIONS = 100;
    constexpr static int DEFAULT_N_POINTS_LINE_SEARCH = 10;
    constexpr static int DEFAULT_BLOCK_DIM_X = 16;
    constexpr static int DEFAULT_BLOCK_DIM_Y = 16;
    constexpr static double DEFAULT_ALPHA = 0.01;
    constexpr static double DEFAULT_LAMBDA = 0.01;
    constexpr static double DEFAULT_TOL = 1e-6;
    constexpr static double DEFAULT_RTOL = 1e-4;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("m", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_M)))
            ("n", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_N)))
            ("k", "Number of outputs", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_K)))
            ("max_iterations", "Maximum optimization iterations", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_MAX_ITERATIONS)))
            ("n_points_line_sear.hpp", "Number of line search points", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_N_POINTS_LINE_SEARCH)))
            ("eta", "Learning rate", cxxopts::value<double>())
            ("alpha", "L1 regularization parameter", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_ALPHA)))
            ("lambda", "L2 regularization parameter", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_LAMBDA)))
            ("tol", "Absolute tolerance", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_TOL)))
            ("rtol", "Relative tolerance", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_RTOL)))
            ("block_dim_x,x", "Number of threads in the x dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_X)))
            ("block_dim_y,y", "Number of threads in the y dimension per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM_Y)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
    }

    inline static Gradient_leastsquares_elasticnet_optimizer_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }

        // Default eta to 1/n_points_line_search^2 if not specified
        double eta;
        if (options_parsed.count("eta")) {
            eta = options_parsed["eta"].as<double>();
        } else {
            int n_points = options_parsed["n_points_line_sear.hpp"].as<int>();
            eta = 1.0 / (n_points * n_points);
        }

        return Gradient_leastsquares_elasticnet_optimizer_spec(
            type,
            options_parsed["m"].as<long>(),
            options_parsed["n"].as<long>(),
            options_parsed["k"].as<long>(),
            options_parsed["max_iterations"].as<int>(),
            options_parsed["n_points_line_sear.hpp"].as<int>(),
            eta,
            options_parsed["alpha"].as<double>(),
            options_parsed["lambda"].as<double>(),
            options_parsed["tol"].as<double>(),
            options_parsed["rtol"].as<double>(),
            options_parsed["block_dim_x"].as<long>(),
            options_parsed["block_dim_y"].as<long>()
        );
    }

    inline Gradient_leastsquares_elasticnet_optimizer_spec(
        const std::string& type,
        const long m,
        const long n,
        const long k,
        const int max_iterations,
        const int n_points_line_search,
        const double eta,
        const double alpha,
        const double lambda,
        const double tol,
        const double rtol,
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
        n_rows_temp_(n * k * n_points_line_search + n * k + n_points_line_search + 1),  // candidates + gradient + losses + norm
        n_cols_temp_(1),
        max_iterations_(max_iterations),
        n_points_line_search_(n_points_line_search),
        eta_(eta),
        alpha_(alpha),
        lambda_(lambda),
        tol_(tol),
        rtol_(rtol),
        block_dim_(block_dim_x, block_dim_y),
        grid_dim_(
            (k_ + block_dim_x - 1) / block_dim_x,
            (n_ + block_dim_y - 1) / block_dim_y
        ),
        dynamic_shared_mem_words_(0)
    {}
};

static_assert(Check_matrix_kernel_spec_3In_1Out<Gradient_leastsquares_elasticnet_optimizer_spec>::check_passed, "Gradient_leastsquares_elasticnet_optimizer_spec is not a valid 3In1Out kernel spec");

template <HIP_scalar Number_>
class Gradient_leastsquares_elasticnet_optimizer_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Gradient_leastsquares_elasticnet_optimizer_spec;

    const Kernel_spec spec_;

    // Embedded gradient kernel for computing gradients
    Gradient_leastsquares_elasticnet_tensor_kernel<Number> gradient_kernel_;

    Gradient_leastsquares_elasticnet_optimizer_kernel(
        const Kernel_spec spec
    ) : spec_(spec),
        gradient_kernel_(Gradient_leastsquares_elasticnet_tensor_spec(
            spec.type_, spec.m_, spec.n_, spec.k_,
            spec.alpha_, spec.lambda_, 16, 16
        )) {}

    void run_device_kernel(
        const Number* const gpu_data_A,  // independent variables (m x n)
        const Number* const gpu_data_B,  // dependent variables (m x k)
        const Number* const gpu_data_M,  // initial coefficients (n x k)
        Number* const gpu_data_result,   // optimized coefficients output (n x k)
        Number* const gpu_data_temp,     // temporary storage
        hipStream_t stream
    ) {
        // Implementation would involve multiple kernel launches:
        // 1. Copy initial M to result
        // 2. Iterative optimization loop with gradient computation and line search
        // 3. Convergence checking

        // For now, placeholder implementation
        hipMemcpyAsync(gpu_data_result, gpu_data_M,
                       spec_.n_ * spec_.k_ * sizeof(Number),
                       hipMemcpyDeviceToDevice, stream);
    }

    template<typename Matrix_like_A, typename Matrix_like_B, typename Matrix_like_M>
    requires is_matrix_like<Matrix_like_A> && is_matrix_like<Matrix_like_B> && is_matrix_like<Matrix_like_M> &&
             std::is_same_v<typename Matrix_like_A::Scalar, Number> &&
             std::is_same_v<typename Matrix_like_B::Scalar, Number> &&
             std::is_same_v<typename Matrix_like_M::Scalar, Number>
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> run_host_kernel(
        const Matrix_like_A& A,  // m x n
        const Matrix_like_B& B,  // m x k
        const Matrix_like_M& M   // n x k initial
    ) {
        auto M_current = M.eval();
        Number prev_loss = std::numeric_limits<Number>::max();

        for (int iter = 0; iter < spec_.max_iterations_; ++iter) {
            // Compute gradient directly with M_current
            auto gradient = gradient_kernel_.run_host_kernel(A, B, M_current);

            // Compute Frobenius norm of M_current
            const auto frobenius_norm = std::sqrt(double(M_current.squaredNorm()));

            // Line search
            Number best_loss = std::numeric_limits<Number>::max();
            auto best_M = M_current;

            for (int j = 1; j <= spec_.n_points_line_search_; ++j) {
                const Number step_size = Number(spec_.eta_) * Number(frobenius_norm * j * j);
                auto M_candidate = M_current - step_size * gradient;

                // Compute loss for candidate
                auto predictions = A * M_candidate;
                auto residuals = B - predictions;
                Number data_loss = residuals.squaredNorm();

                // Add regularization
                Number l1_loss = Number(spec_.alpha_) * M_candidate.cwiseAbs().sum();
                Number l2_loss = Number(spec_.lambda_) * M_candidate.squaredNorm();
                Number total_loss = data_loss + l1_loss + l2_loss;

                if (total_loss < best_loss) {
                    best_loss = total_loss;
                    best_M = M_candidate;
                }
            }

            // Check convergence
            Number improvement = prev_loss - best_loss;
            Number relative_improvement = improvement / (prev_loss + Number(1e-10));

            if (improvement < Number(spec_.tol_) || relative_improvement < Number(spec_.rtol_)) {
                break;
            }

            M_current = best_M;
            prev_loss = best_loss;
        }

        return M_current;
    }
};

static_assert(Check_matrix_kernel_3In_1Out_template<Gradient_leastsquares_elasticnet_optimizer_kernel>::check_passed, "Gradient_leastsquares_elasticnet_optimizer_kernel is not a valid 3In1Out kernel template");
