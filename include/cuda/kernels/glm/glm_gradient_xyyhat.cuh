// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/glm/glm_gradient_xyyhat.hpp

#pragma once

#include <cassert>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/type_traits.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/kernel_api/tensor3d_3in_1out.cuh"

#include "glm_predict_naive.cuh"

/*
This kernel computes the gradient of the loss of a linear regression
with a squared error loss function and an elastic net regularization term.

### Notation
We will use the following convention with respect to tensor indices:
- `feature ∈ 1..nfeatures` indexes over features  (previously, `k`)
- `target ∈ 1..ntargets` indexes over targets     (previously, `h`)
- `task ∈ 1..ntasks` indexes over tasks           (previously, `i`)
- `obs ∈ 1..nobs` indexes over observations       (previously, `j`)

where we interpret the `nobs` as the number of observations (the length of the data set),
`ntasks` as the number of interrelated regression tasks per observation, `nfeatures` as
the number of covariates (independent variables), and `ntargets` as the number of
target variables.

### Data Format
The input to the algorithm is a triplet of tensors of rank 3: `(X, Y, M)` such that
- `X` with shape `(nfeatures, ntasks, nobs)` is the tensor of independent variables
- `Y` with shape `(ntargets, ntasks, nobs)` is the tensor of dependent variables
- `M` with shape `(nfeatures, ntargets, ntasks)` is a linear model predicting Y given X.

The output of the model is computed as follows:

`Ŷ[target, task, obs] = SUM_feature M[feature,target,task] * X[feature,task,obs]`.

Notice that this cannot be computed as a tensor contraction (Einstein summation), because a contraction
reduces the total number of dimensions by 2. So for example we could compute the following contraction,
(using Einstein notation):

M[feature,target,task] X[feature,task,obs]

but it would result in a tensor having only two dimensions: {target,obs}, which is not what we want.

We could also perform the following contraction:
M[feature,target,task] X[feature,task',obs]

but it would result in a tensor having four dimensions: {target,task,task',obs}, which is also not
what we want.

If we wanted to use tensor notation, we would first have to use a contraction over the shared `feature`
dimension, with distinct `task` and `task'` dimensions, followed by a selection for `task == task'`

Ŷ[target,task,obs] = (M[feature,target,task] X[feature,task',obs])[target,task,task,obs}

The squared-error loss function is:
`L(M) = SUM_target,task,obs E[target,task,obs]^2 = SUM_target,task,obs (Ŷ[target,task,obs] - Y[target,task,obs])^2 = SUM_target,task,obs ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs])^2`

The ElasticNet regularization term is:
`ElastNet(M|alpha,lambda) = lambda * (alpha * L1(M) + (1-alpha) * L2(M)^2)`
`                         = lambda * (alpha * SUM_task,obs,feature |M[feature,target,task]| + (1-alpha) * SUM_task,obs,feature M[feature,target,task]^2)`

We propose a multitask generalization of ElasticNet that exploits the assumed inherent similarity between tasks
to allow each task to borrow predictiveness from the other task.

The regularized loss function is:
`L_r(M|alpha,lambda) = SUM_target,task,obs (Ŷ[target,task,obs] - Y[target,task,obs])^2 + lambda * (alpha * SUM_task,obs,k |M[feature,target,task]| + (1-alpha) * SUM_task,obs,k M[feature,target,task]^2)`

We want to compute the gradient L_r with respect to the elements of M.

The gradient is:
dL/dM[feature',target',task'] =
    = d/dM[feature',target',task'] SUM_target,task,obs ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs])^2
    = SUM_target,task,obs d/dM[feature',target',task'] ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs])^2
    = SUM_target,task,obs 2 * ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs]) * d/dM[feature',target',task'] ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs])
    = SUM_target,task,obs 2 * ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs]) * ( (SUM_feature d/dM[feature',target',task'] M[feature,target,task] * X[feature,task,obs]) - d/dM[feature',target',task'] Y[target,task,obs])
    = SUM_target,task,obs 2 * ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs]) * ( (SUM_feature delta[feature=feature',target=target',task=task'] * X[feature,task,obs])   - 0)
    = SUM_target,task,obs 2 * ( (SUM_feature M[feature,target,task] * X[feature,task,obs]) - Y[target,task,obs]) * delta[target=target',task=task'] * X[feature',task,obs]
    = 2 * SUM_obs ( (SUM_feature M[feature,target',task'] * X[feature,task',obs]) - Y[target',task',obs]) * X[feature',task',obs]
    = 2 * SUM_obs ( Ŷ[target',task',obs] - Y[target',task',obs]) * X[feature',task',obs]
*/

namespace glm {

    template <CUDA_scalar Number>
    using transform_ptr = Number (*const)(Number);

    template <CUDA_scalar Number>
    __global__ void glm_gradient_XYYhat_block(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        const Number* const Yhat,  // (ntargets, ntasks, nobs)
        Number* const grad_M,      // (nfeatures, ntargets, ntasks)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs
    ) {
        // We need one word of dynamic shm per warp per block
        Number* shm  = get_dynamic_shared_memory<Number>();

        assert(blockDim.x % WARP_SIZE == 0); // Number of threads is a multiple of a warp
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one block per location in grad_M to sum over obs and over feature
        const auto bid_grid = blockIdx.x; // We use a 1D grid
        // const auto& tid_block = threadIdx.x; // We use a 1D block
        const auto tid_warp = threadIdx.x % WARP_SIZE;
        const auto wid_block = threadIdx.x / WARP_SIZE;
        const auto n_warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto M_output_size = M_sheet_size * ntasks;

        const auto& nobs_per_block = blockDim.x;
        assert(nobs_per_block > 0 && nobs_per_block % WARP_SIZE == 0);

        for (auto grad_M_idx = bid_grid; grad_M_idx < M_output_size; grad_M_idx += gridDim.x) {
            const auto dst_task = grad_M_idx / M_sheet_size;
            const auto dst_task_idx = grad_M_idx % M_sheet_size;
            const auto dst_target = dst_task_idx / nfeatures;
            const auto dst_feature = dst_task_idx % nfeatures;

            Number sum_obs{0};
            for (long obs = 0; obs < nobs; obs += nobs_per_block ) {
                const auto Y_idx = dst_target + dst_task * ntargets + obs * Y_sheet_size;
                const Number yhat = Yhat[Y_idx];
                const Number y = Y[Y_idx];

                // compute ( Yhat[target',task',obs] - Y[target',task',obs]) * X[feature',task',obs]
                sum_obs += (
                    // compute ( Yhat[target',task',obs] - Y[target',task',obs])
                    yhat - y
                ) * (
                    // compute X[feature',task',obs]
                    X[dst_feature + dst_task * nfeatures + obs * X_sheet_size]
                );
            }
            // Now we need to reduce/sum over the threads of the block to get the aggregate sum_feature
            // First a warp shuffle reduction down to lane 0
            for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                sum_obs += __shfl_down_sync(__activemask(), sum_obs, reduced_lanes);
            }

            // Finally, we sum over shared memory using a single warp
            if (wid_block == 0) {
                sum_obs = (tid_warp < n_warps_per_block) ? shm[tid_warp] : Number(0);
                for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                    sum_obs += __shfl_down_sync(__activemask(), sum_obs, reduced_lanes);
                }
                if (tid_warp == 0) {
                    grad_M[grad_M_idx] = Number(2) * sum_obs;
                }
            }
        }
    }

    template <CUDA_scalar Number>
    __global__ void glm_gradient_XYYhat_naive(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        const Number* const Yhat,  // (ntargets, ntasks, nobs)
        Number* const grad_M,      // (nfeatures, ntargets, ntasks)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs
    ) {
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one thread per location in grad_M to sum over obs and over feature
        const auto tid_grid = threadIdx.x + blockIdx.x * blockDim.x; // We use a 1D grid
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto M_output_size = M_sheet_size * ntasks;

        const auto nthreads_per_grid = gridDim.x * blockDim.x;

        for (auto grad_M_idx = tid_grid; grad_M_idx < M_output_size; grad_M_idx += nthreads_per_grid) {
            const auto dst_task = grad_M_idx / M_sheet_size;
            const auto dst_task_idx = grad_M_idx % M_sheet_size;
            const auto dst_target = dst_task_idx / nfeatures;
            const auto dst_feature = dst_task_idx % nfeatures;

            Number sum_obs{0};
            for (long obs = 0; obs < nobs; obs += 1) {
                const auto Y_idx = dst_target + dst_task * ntargets + obs * Y_sheet_size;
                const Number yhat = Yhat[Y_idx];
                const Number y = Y[Y_idx];

                // compute ( Yhat[target',task',obs] - Y[target',task',obs]) * X[feature',task',obs]
                sum_obs += (
                    // compute ( Yhat[target',task',obs] - Y[target',task',obs])
                    yhat - y
                ) * (
                    // compute X[feature',task',obs]
                    X[dst_feature + dst_task * nfeatures + obs * X_sheet_size]
                );
            }
            grad_M[grad_M_idx] = Number(2) * sum_obs;
        }
    }

    template <CUDA_scalar Number>
    __global__ void glm_gradient_XYYhat_fixed_grid(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        const Number* const Yhat,  // (ntargets, ntasks, nobs)
        Number* const grad_M,      // (nfeatures, ntargets, ntasks)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs
    ) {
        const auto dst_feature = blockIdx.x * blockDim.x + threadIdx.x;
        const auto dst_target = blockIdx.y * blockDim.y + threadIdx.y;
        const auto dst_task = blockIdx.z * blockDim.z + threadIdx.z;

        if (dst_feature < nfeatures && dst_target < ntargets && dst_task < ntasks) {
            const auto X_sheet_size = nfeatures * ntasks;
            const auto Y_sheet_size = ntargets * ntasks;
            const auto M_sheet_size = nfeatures * ntargets;
            const auto grad_M_idx = dst_feature + dst_target * nfeatures + dst_task * M_sheet_size;

            Number sum_obs{0};
            for (long obs = 0; obs < nobs; obs += 1) {
                const auto X_idx = dst_feature + dst_task * nfeatures + obs * X_sheet_size;
                const auto Y_idx = dst_target + dst_task * ntargets + obs * Y_sheet_size;

                // compute ( Yhat[target',task',obs] - Y[target',task',obs]) * X[feature',task',obs]
                sum_obs += (
                    // compute ( Yhat[target',task',obs] - Y[target',task',obs])
                    Yhat[Y_idx] - Y[Y_idx]
                ) * (
                    // compute X[feature',task',obs]
                    X[X_idx]
                );
            }
            grad_M[grad_M_idx] = Number(2) * sum_obs;
        }
    } // glm_gradient_XYYhat_fixed_grid

    template <CUDA_scalar Number>
    __global__ void glm_gradient_XYYhat_warp(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        const Number* const Yhat,  // (ntargets, ntasks, nobs)
        Number* const grad_M,      // (nfeatures, ntargets, ntasks)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs
    ) {
        // We need one word of dynamic shm per warp per block
        Number* shm  = get_dynamic_shared_memory<Number>();

        assert(blockDim.x % WARP_SIZE == 0); // Number of threads is a multiple of a warp
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one warp per location in grad_M to sum over obs and over feature
        // const auto bid_grid = blockIdx.x;
        // We use a 1D grid
        const auto tid_grid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto wid = tid_grid/WARP_SIZE;
        // const auto& tid_block = threadIdx.x; // We use a 1D block
        const auto tid_warp = threadIdx.x % WARP_SIZE;
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto M_output_size = M_sheet_size * ntasks;

        const auto nthreads = (long)gridDim.x * (long)blockDim.x;
        static_assert(std::same_as<decltype(nthreads), const long>);
        const auto nwarps = nthreads / WARP_SIZE;

        Number sum_obs;
        for (auto grad_M_idx = wid; grad_M_idx < M_output_size; grad_M_idx += nwarps) {
            sum_obs = 0;
            const auto dst_task = grad_M_idx / M_sheet_size;
            const auto dst_task_idx = grad_M_idx % M_sheet_size;
            const auto dst_target = dst_task_idx / nfeatures;
            const auto dst_feature = dst_task_idx % nfeatures;
            assert(grad_M_idx == dst_feature + dst_target * nfeatures + dst_task * M_sheet_size);
            for (long obs = tid_warp; obs < nobs; obs += WARP_SIZE) {
                const auto Y_idx = dst_target + dst_task * ntargets + obs * Y_sheet_size;
                const Number yhat = Yhat[Y_idx];
                const Number y = Y[Y_idx];

                // compute ( Yhat[target',task',obs] - Y[target',task',obs]) * X[feature',task',obs]
                sum_obs += (
                    // compute ( Yhat[target',task',obs] - Y[target',task',obs])
                    yhat - y
                ) * (
                    // compute X[feature',task',obs]
                    X[dst_feature + dst_task * nfeatures + obs * X_sheet_size]
                );
            }
            // Now we need to reduce/sum over the threads of the warp to get the aggregate sum_feature
            // First a warp shuffle reduction down to lane 0
            printf("[DEBUG] lane=%d, grad_M_idx=%u, sum_obs=%f\n", tid_warp, grad_M_idx, float(sum_obs));
            for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                sum_obs += __shfl_down_sync(__activemask(), sum_obs, reduced_lanes);
            }

            // Now lane 0 can write the total to the output array
            if (tid_warp == 0) {
                printf("[DEBUG] reduced: grad_M_idx=%ud, grad_m=%f\n", grad_M_idx, float(Number(2)*sum_obs));
                assert(grad_M[grad_M_idx] == Number(0));
                grad_M[grad_M_idx] = Number(2) * sum_obs;
            }
        }
    } // glm_gradient_XYYhat_warp


} // namespace glm

struct Glm_gradient_xyyhat_spec {
    const std::string type_;
    const std::string gpu_algo_;
    const std::string cpu_algo_;

    const long nfeatures_;
    const long ntargets_;
    const long ntasks_;
    const long nobs_;
    const long niterations_;

    // X tensor (covariates) with dimensions (nfeatures, ntasks, nobs)
    const long n_cols_A_;   // nfeatures
    const long n_rows_A_;   // ntasks
    const long n_sheets_A_; // nobs

    // M tensor (model) with dimensions (nfeatures, ntargets, ntasks)
    const long n_cols_B_;   // nfeatures
    const long n_rows_B_;   // ntargets
    const long n_sheets_B_; // ntasks

    // Y tensor (responses) with dimensions (ntargets, ntasks, nobs)
    const long n_cols_C_;   // ntargets
    const long n_rows_C_;   // ntasks
    const long n_sheets_C_; // nobs

    // grad_M tensor (gradient) with dimensions (nfeatures, ntargets, ntasks)
    const long n_cols_D_;   // nfeatures
    const long n_rows_D_;   // ntargets
    const long n_sheets_D_; // ntasks

    // Yhat tensor (predictions) with dimensions (ntargets, ntasks, nobs)
    const long n_cols_temp_;
    const long n_rows_temp_;
    const long n_sheets_temp_;

    const unsigned block_dim_;
    const unsigned fixed_grid_block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;
    // const bool optimize_launch_;

    constexpr static long DEFAULT_NOBS = 1000;
    constexpr static long DEFAULT_NTASKS = 10;
    constexpr static long DEFAULT_NTARGETS = 25;
    constexpr static long DEFAULT_NFEATURES = 25;
    constexpr static long DEFAULT_BLOCK_DIM = 256;
    constexpr static std::string DEFAULT_GPU_ALGO = "naive";

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("nfeatures,X", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NFEATURES)))
            ("ntargets,Y", "Number of targets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTARGETS)))
            ("ntasks,T", "Number of tasks", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTASKS)))
            ("nobs,N", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NOBS)))
            ("block-dim,n", "Number of threads per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            // ("optimize-launch", "Use occupancy API to determine optimal launch configuration")
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"))

            // Commenting out --cpu-algo as the Eigen matrix implementation is broken. More debugging necessary before it can be enabled.
            ("gpu-algo", "GPU algo to use (fixed-grid, naive, warp, block)", cxxopts::value<std::string>()->default_value(DEFAULT_GPU_ALGO))
            ("cpu-algo", "CPU algo variant to benchmark against (nested-loop, matrix)", cxxopts::value<std::string>()->default_value("nested-loop"))
            ;
        }

    inline static Glm_gradient_xyyhat_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate gpu_algo
        const auto gpu_algo = options_parsed["gpu-algo"].as<std::string>();
        if (gpu_algo != "fixed-grid" && gpu_algo != "naive" && gpu_algo != "warp" && gpu_algo != "block") {
            std::cerr << "[ERROR] --gpu-algo must be one of: fixed-grid, naive, warp, block" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --gpu-algo: " + gpu_algo);
        }
        // Validate cpu_algo
        const auto& cpu_algo = options_parsed["cpu-algo"].as<std::string>();
        if (cpu_algo != "nested-loop" && cpu_algo != "matrix") {
            std::cerr << "[ERROR] --nested-loop must be one of: nested-loop, matrix" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --nested-loop: " + cpu_algo);
        }
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        // Parse and transform block-dim
        const auto block_dim = std::min(options_parsed["block-dim"].as<long>(), 1024L);
        return {
            type,
            gpu_algo,
            cpu_algo,
            options_parsed["nfeatures"].as<long>(),
            options_parsed["ntargets"].as<long>(),
            options_parsed["ntasks"].as<long>(),
            options_parsed["nobs"].as<long>(),
            block_dim,
            // options_parsed.count("optimize-launch") > 0
        };
    }

    inline Glm_gradient_xyyhat_spec(
        const std::string& type,
        const std::string& gpu_algo,
        const std::string& cpu_algo,
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const long block_dim
        // const bool optimize_launch
    ) : type_(type),
        gpu_algo_(gpu_algo),
        cpu_algo_(cpu_algo),
        nfeatures_(nfeatures),
        ntargets_(ntargets),
        ntasks_(ntasks),
        nobs_(nobs),
        niterations_(nobs * ntasks * ntargets),

        // X
        n_cols_A_(nfeatures),
        n_rows_A_(ntasks),
        n_sheets_A_(nobs),

        // Y
        n_cols_B_(ntargets),
        n_rows_B_(ntasks),
        n_sheets_B_(nobs),

        // M
        n_cols_C_(nfeatures),
        n_rows_C_(ntargets),
        n_sheets_C_(ntasks),

        // grad_M
        n_cols_D_(nfeatures),
        n_rows_D_(ntargets),
        n_sheets_D_(ntasks),

        // Yhat
        n_cols_temp_(ntargets),
        n_rows_temp_(ntasks),
        n_sheets_temp_(nobs),

        block_dim_(block_dim),
        fixed_grid_block_dim_(std::floor(std::cbrt(block_dim_))),
        grid_dim_((gpu_algo == "fixed-grid") ? dim3(
            (nfeatures + fixed_grid_block_dim_ - 1)/fixed_grid_block_dim_,
            (ntargets + fixed_grid_block_dim_ - 1)/fixed_grid_block_dim_,
            (ntasks + fixed_grid_block_dim_ - 1)/fixed_grid_block_dim_
        )
        : (gpu_algo == "naive") ? dim3(niterations_)
        : dim3(niterations_ * 32)),
        dynamic_shared_mem_words_(0)
        // optimize_launch_(optimize_launch)
    {
        assert(block_dim_ > 0);
        assert(grid_dim_.x > 0);
        assert(grid_dim_.y > 0);
        assert(grid_dim_.z > 0);
    }
};

static_assert(Check_tensor3d_kernel_spec_3In_1Out<Glm_gradient_xyyhat_spec>::check_passed, "Glm_gradient_xyyhat_spec is not a valid 3In1Out kernel spec");

template <CUDA_scalar Number_>
class Glm_gradient_xyyhat_kernel {
    public:
    using Number = Number_;
    using Printable_Number = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;
    using Kernel_spec = Glm_gradient_xyyhat_spec;

    const Kernel_spec spec_;

    Glm_gradient_xyyhat_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_X,
        const Number* const gpu_data_Y,
        const Number* const gpu_data_M,
        Number* const gpu_data_grad_M,
        Number* const gpu_data_Yhat,
        cudaStream_t stream
    ) {
        long block_dim;
        dim3 grid_dim;

        // We initialize block_dim and grid_dim after declaring them because we might want to add support --optimize-launch
        block_dim = spec_.block_dim_;
        grid_dim = spec_.grid_dim_;
        const auto dynamic_shared_mem_words = compute_n_warps_per_block(block_dim);
        const auto predict_shm_size = dynamic_shared_mem_words * sizeof(Number);
        const dim3 fixed_grid_block_dim3{spec_.fixed_grid_block_dim_, spec_.fixed_grid_block_dim_, spec_.fixed_grid_block_dim_};

        const Glm_predict_naive_spec glm_predict_naive_spec{
            spec_.type_,
            "fixed-grid",
            spec_.nfeatures_,
            spec_.ntargets_,
            spec_.ntasks_,
            spec_.nobs_,
            spec_.fixed_grid_block_dim_
        };
        // Compute Yhat
        std::cout << "[INFO] kernel launch: glm::glm_predict_fixed_grid<<<" << glm_predict_naive_spec.grid_dim_.x << ", " << glm_predict_naive_spec.block_dim_.x << ", " << predict_shm_size
            << ">>>(..., " << glm_predict_naive_spec.nfeatures_ << ", " << glm_predict_naive_spec.ntargets_ << ", " << glm_predict_naive_spec.ntasks_ << ", " << glm_predict_naive_spec.nobs_ << ")" << std::endl;
        std::cout << "[INFO] niterations = " << glm_predict_naive_spec.nobs_ * glm_predict_naive_spec.ntasks_ * glm_predict_naive_spec.ntargets_ << std::endl;
        glm::glm_predict_fixed_grid<<<
            glm_predict_naive_spec.grid_dim_,
            glm_predict_naive_spec.block_dim_,
            predict_shm_size,
            stream
        >>>(
            gpu_data_X,
            // gpu_data_Y,
            gpu_data_M,
            gpu_data_Yhat,
            glm_predict_naive_spec.nfeatures_, glm_predict_naive_spec.ntargets_, glm_predict_naive_spec.ntasks_, glm_predict_naive_spec.nobs_,
            static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
        );

        // Compute grad_M
        if (spec_.gpu_algo_ == "fixed-grid") {
            const int dynamic_shared_mem_words = 0;
            const int shm_size = dynamic_shared_mem_words * sizeof(Number);

            std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
            std::cout << "[INFO] block_dim_: " << fixed_grid_block_dim3.x << ", " << fixed_grid_block_dim3.y << ", " << fixed_grid_block_dim3.z << std::endl;
            std::cout << "[INFO] kernel launch: glm::glm_gradient_XYYhat_fixed_grid<<<(" << spec_.grid_dim_.x << ", " << spec_.grid_dim_.y << ", " << spec_.grid_dim_.z << "), ("
                << spec_.block_dim_ << "), " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_gradient_XYYhat_fixed_grid<<<
                grid_dim,
                fixed_grid_block_dim3,
                shm_size,
                stream
            >>>(
                gpu_data_X,
                gpu_data_Y,
                gpu_data_Yhat,
                gpu_data_grad_M,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_
            );
        } else if (spec_.gpu_algo_ == "naive") {
            const int dynamic_shared_mem_words = 0;
            const int shm_size = dynamic_shared_mem_words * sizeof(Number);
            std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
            std::cout << "[INFO] block_dim_: " << block_dim << std::endl;
            std::cout << "[INFO] kernel launch: glm::glm_gradient_XYYhat_naive<<<" << grid_dim.x << ", " << block_dim << ", " << shm_size
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_gradient_XYYhat_naive<<<
                grid_dim,
                block_dim,
                shm_size,
                stream
            >>>(
                gpu_data_X,
                gpu_data_Y,
                gpu_data_Yhat,
                gpu_data_grad_M,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_
            );
        } else if (spec_.gpu_algo_ == "warp") {
            const int dynamic_shared_mem_words = spec_.dynamic_shared_mem_words_;
            const int shm_size = dynamic_shared_mem_words * sizeof(Number);
            std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
            std::cout << "[INFO] block_dim_: " << block_dim << std::endl;
            std::cout << "[INFO] kernel launch: glm::glm_gradient_XYYhat_warp<<<" << grid_dim.x << ", " << block_dim << ", " << shm_size
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_gradient_XYYhat_warp<<<
                grid_dim,
                block_dim,
                shm_size,
                stream
            >>>(
                gpu_data_X,
                gpu_data_Y,
                gpu_data_Yhat,
                gpu_data_grad_M,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_
            );
        }  else if (spec_.gpu_algo_ == "block") {
            const int dynamic_shared_mem_words = spec_.dynamic_shared_mem_words_;
            const int shm_size = dynamic_shared_mem_words * sizeof(Number);
            std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
            std::cout << "[INFO] block_dim_: " << block_dim << std::endl;
            std::cout << "[INFO] kernel launch: glm::glm_gradient_XYYhat_block<<<" << grid_dim.x << ", " << block_dim << ", " << shm_size
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_gradient_XYYhat_block<<<
                grid_dim,
                block_dim,
                shm_size,
                stream
            >>>(
                gpu_data_X,
                gpu_data_Y,
                gpu_data_Yhat,
                gpu_data_grad_M,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_
            );
        } else {
            std::cerr << "[ERROR] --gpu-algo must be one of: fixed-grid, naive, warp, block" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --gpu-algo: " + spec_.gpu_algo_);
            assert(false); // We should have printed this error when parsing the command line
        }
    }

    Tensor3D<Number> run_host_kernel(
        const Tensor3D<Number>& X,
        const Tensor3D<Number>& Y,
        const Tensor3D<Number>& M
    ) {
/*
dL/dM[feature',target',task'] =
    = 2 * SUM_obs (
                     (SUM_feature M[feature,target',task'] * X[feature,task',obs])
                    - Y[target',task',obs]
                ) * X[feature',task',obs]
*/
        Tensor3D<Number> grad_M{spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_};

        if (spec_.cpu_algo_ == "nested-loop") {
            #pragma omp parallel for
            for (int dst_feature = 0; dst_feature < spec_.nfeatures_; ++dst_feature) {
                for (int dst_target = 0; dst_target < spec_.ntargets_; ++dst_target) {
                    for (int dst_task = 0; dst_task < spec_.ntasks_; ++dst_task) {
                        Number sum_obs{0};
                        for (int obs = 0; obs < spec_.nobs_; ++obs) {
                            const auto sum_feature = M.row_at(dst_target, dst_task).dot(X.row_at(dst_task, obs));
                            sum_obs += (sum_feature - Y.at(dst_target, dst_task, obs)) * X.at(dst_feature, dst_task, obs);
                        }
                        grad_M.at(dst_feature, dst_target, dst_task) = Number(2) * sum_obs;
                    }
                }
            }
        } else {
            // Tensor dimensions: innermost, intermediate, outermost
            // X: (nfeatures, ntasks, nobs)
            // Y: (ntargets, ntasks, nobs)
            // M: (nfeatures, ntargets, ntasks)
            // const Eigen::IOFormat eigen_format(4, 0, ", ", "\n", "  [", "]");
            for (int task = 0; task < spec_.ntasks_; ++task) {
                // Matrix dimensions: [nrows (outer), ncols (inner)]
                auto X_task_matrix = X.chip_at_dim1(task);  // tensor:(nfeatures, nobs) -> matrix:[nobs, nfeatures]
                auto Y_task_matrix = Y.chip_at_dim1(task);  // tensor:(ntargets,  nobs) -> matrix:[nobs, ntargets]
                auto M_task_matrix = M.sheet_at(task);    // tensor:(nfeatures, ntargets) -> matrix:[ntargets, nfeatures]
                auto grad_M_task_matrix = grad_M.sheet_at(task); // tensor:(nfeatures, ntargets) -> matrix:[ntargets, nfeatures]
                assert(X_task_matrix.rows() == spec_.nobs_);
                assert(X_task_matrix.cols() == spec_.nfeatures_);
                assert(Y_task_matrix.rows() == spec_.nobs_);
                assert(Y_task_matrix.cols() == spec_.ntargets_);
                assert(M_task_matrix.rows() == spec_.ntargets_);
                assert(M_task_matrix.cols() == spec_.nfeatures_);
                assert(grad_M_task_matrix.rows() == spec_.ntargets_);
                assert(grad_M_task_matrix.cols() == spec_.nfeatures_);
                // for each slice: grad_M = (M*X^T - Y^T)*X
                // grad_M_task_matrix.noalias() = X_task_matrix.transpose() * (X_task_matrix * M_task_matrix - Y_task_matrix);
                grad_M_task_matrix.noalias() = 2*(M_task_matrix * X_task_matrix.transpose() - Y_task_matrix.transpose()) * X_task_matrix;
                // std::cout << "X_task_matrix =\n" << X_task_matrix.template cast<Printable_Number>().format(eigen_format)
                //     << "\n Y_task_matrix =\n" << Y_task_matrix.template cast<Printable_Number>().format(eigen_format)
                //     << "\n M_task_matrix =\n" << M_task_matrix.template cast<Printable_Number>().format(eigen_format)
                //     << "\n grad_M_task_matrix =\n" << grad_M_task_matrix.template cast<Printable_Number>().format(eigen_format)
                //     << std::endl;
            }
        }
        return grad_M;
    }
};

static_assert(Check_tensor3d_kernel_3In_1Out_template<Glm_gradient_xyyhat_kernel>::check_passed, "Glm_gradient_xyyhat_kernel is not a valid 3In1Out kernel template");
