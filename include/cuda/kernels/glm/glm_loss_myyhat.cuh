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


// source path: include/cuda/kernels/glm/glm_loss_myyhat.cuh

#pragma once

#include <cassert>
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
`L_r(M|alpha,lambda) = SUM_target,task,obs (Ŷ[target,task,obs] - Y[target,task,obs])^2 + lambda * (alpha * SUM_feature,target,task |M[feature,target,task]| + (1-alpha) * SUM_feature,target,task M[feature,target,task]^2)`
*/

namespace glm {

    template <CUDA_scalar Number>
    using transform_ptr = Number (*const)(Number);

    template <CUDA_scalar Number>
    __global__ void glm_loss_MYYhat(
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        const Number* const Yhat,  // (ntargets, ntasks, nobs)
        Number* const loss,        // 1 element, use atomics
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const Number lambda,
        const Number alpha
    ) {
        Number* warp_loss = get_dynamic_shm<Number>(); // One block per warp per block
        const auto size_M = nfeatures * ntargets * ntasks;
        const auto size_Y = ntargets * ntasks * nobs;

        const auto tid_grid = blockIdx.x * blockDim.x + threadIdx.x; // 1D grid
        const auto wid_block = threadIdx.x / WARP_SIZE;
        const auto tid_warp = threadIdx.x % WARP_SIZE;

        Number thread_loss{Number(0)};
        if (tid_grid < size_M) {
            const auto m = M[tid_grid];
            thread_loss = lambda * (alpha * abs(m) + (Number(1) - alpha) * m * m);
        } else {
            const auto Y_idx = tid_grid - size_M;
            if (Y_idx < size_Y) {
                const auto error = Yhat[Y_idx] - Y[Y_idx];
                thread_loss = error * error;
            }
        }

        // Warp-shuffle reduction: compute total loss for this warp
        for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
            thread_loss += __shfl_down_sync(__activemask(), thread_loss, reduced_lanes);
        }

        // Copy reduced loss to shared memory
        if (tid_warp == 0) {
            warp_loss[wid_block] = thread_loss;
        }
        __syncthreads();

        // Use wid_block == 0 to sum the warp-level losses recorded in shared memory
        if (wid_block == 0) {
            thread_loss = warp_loss[tid_warp];

            // Warp-shuffle reduction: compute total loss for this block
            for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                thread_loss += __shfl_down_sync(__activemask(), thread_loss, reduced_lanes);
            }

            // Write to global memory
            if (tid_warp == 0) {
                atomicAdd(loss, thread_loss);
            }
        }
    } // glm_loss_MYYhat

} // namespace glm

struct Glm_loss_myyhat_spec {
    const std::string type_;
    const std::string cpu_algo_;
    const double lambda_;
    const double alpha_;
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

    const long size_Y_;
    const long size_M_;

    // temp memory to hold the partial results: 1 element per block
    const long size_temp_;


    const long block_dim_;
    const long grid_dim_;
    const size_t dynamic_shm_words_;
    // const bool optimize_launch_;

    constexpr static long DEFAULT_NOBS = 1000;
    constexpr static long DEFAULT_NTASKS = 10;
    constexpr static long DEFAULT_NTARGETS = 25;
    constexpr static long DEFAULT_NFEATURES = 25;
    constexpr static long DEFAULT_BLOCK_DIM = 512;
    constexpr static double DEFAULT_LAMBDA = 0.0;
    constexpr static double DEFAULT_ALPHA = 0.0; // We default to pure L2 regularization
    constexpr static std::string DEFAULT_GPU_ALGO = "naive";

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("nfeatures,X", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NFEATURES)))
            ("ntargets,Y", "Number of targets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTARGETS)))
            ("ntasks,T", "Number of tasks", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTASKS)))
            ("nobs,N", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NOBS)))
            ("lambda", "Regularization parameter", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_LAMBDA)))
            ("alpha", "L1 mixing parameter (default to 0.0 for pure L2 regularization)", cxxopts::value<double>()->default_value(std::to_string(DEFAULT_ALPHA)))
            ("block-dim", "Number of threads per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            // ("optimize-launch", "Use occupancy API to determine optimal launch configuration")
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"))
            ;
        }

    inline static Glm_loss_myyhat_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        return {
            type,
            options_parsed["nfeatures"].as<long>(),
            options_parsed["ntargets"].as<long>(),
            options_parsed["ntasks"].as<long>(),
            options_parsed["nobs"].as<long>(),
            options_parsed["lambda"].as<double>(),
            options_parsed["alpha"].as<double>(),
            options_parsed["block-dim"].as<unsigned>(),
            // options_parsed.count("optimize-launch") > 0
        };
    }

    inline Glm_loss_myyhat_spec(
        const std::string& type,
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const double lambda,
        const double alpha,
        const unsigned block_dim
        // const bool optimize_launch
    ) : type_(type),
        lambda_(lambda),
        alpha_(alpha),
        nfeatures_(nfeatures),
        ntargets_(ntargets),
        ntasks_(ntasks),
        nobs_(nobs),
        niterations_(nobs * ntasks * ntargets),

        // M tensor (model) with dimensions (nfeatures, ntargets, ntasks)
        n_cols_A_(nfeatures),
        n_rows_A_(ntasks),
        n_sheets_A_(nobs),

        // Y tensor (responses) with dimensions (ntargets, ntasks, nobs)
        n_cols_B_(nfeatures),
        n_rows_B_(ntargets),
        n_sheets_B_(ntasks),

        // Yhat tensor (responses) with dimensions (ntargets, ntasks, nobs)
        n_cols_C_(ntargets),
        n_rows_C_(ntasks),
        n_sheets_C_(nobs),

        // loss
        n_cols_D_(1),
        n_rows_D_(1),
        n_sheets_D_(1),

        // temp memory
        size_Y_(ntargets * ntasks * nobs),
        size_M_(nfeatures * ntargets * ntasks),
        size_temp_((size_M_ + size_Y_ + block_dim - 1)/block_dim),
        block_dim_(block_dim),
        grid_dim_((size_M_ + size_Y_ + block_dim_ - 1)/block_dim_),
        dynamic_shm_words_(0)
        // optimize_launch_(optimize_launch)
    {
        assert(block_dim_ > 0);

        assert(grid_dim_ > 0);
        assert(lambda_ >= 0);
        assert(alpha_ >= 0);
        assert(alpha_ <= 1);
    }
};

// static_assert(Check_tensor3d_kernel_spec_3In_1Out<Glm_loss_myyhat_spec>::check_passed, "Glm_loss_myyhat_spec is not a valid 3In1Out kernel spec");

template <CUDA_scalar Number_>
class Glm_loss_myyhat_kernel {
    public:
    using Number = Number_;
    using Printable_Number = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;
    using Kernel_spec = Glm_loss_myyhat_spec;

    const Kernel_spec spec_;
    const Number lambda_;
    const Number alpha_;

    Glm_loss_myyhat_kernel(
        const Kernel_spec spec
    ) : spec_(spec),
        lambda_(spec.lambda_),
        alpha_(spec.alpha_) {}

    void run_device_kernel(
        const Number* const gpu_data_M,
        const Number* const gpu_data_Y,
        const Number* const gpu_data_Yhat,
        Number* const gpu_data_loss,
        Number* const gpu_data_temp, // unused
        cudaStream_t stream
    ) {
        long block_dim;
        long grid_dim;

        // We initialize block_dim and grid_dim after declaring them because we might want to add support --optimize-launch
        block_dim = spec_.block_dim_;
        grid_dim = spec_.grid_dim_;
        const auto dynamic_shm_words = compute_n_warps_per_block(block_dim);
        const auto shm_size = dynamic_shm_words * sizeof(Number);
        cuda_check_error(cudaMemsetAsync(gpu_data_loss, 0, sizeof(Number), stream), "cudaMemsetAsync");

        std::cout << "[INFO] kernel launch: glm::glm_loss_MYYhat<<<(" << spec_.grid_dim_ << "), ("
                << spec_.block_dim_ << "), " << shm_size
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
        glm::glm_loss_MYYhat<<<
            grid_dim,
            block_dim,
            shm_size,
            stream
        >>>(
            gpu_data_M,
            gpu_data_Y,
            gpu_data_Yhat,
            gpu_data_loss,
            spec_.nfeatures_,
            spec_.ntargets_,
            spec_.ntasks_,
            spec_.nobs_,
            lambda_, alpha_
        );
    }

    Tensor3D<Number> run_host_kernel(
        const Tensor3D<Number>& M,
        const Tensor3D<Number>& Y,
        const Tensor3D<Number>& Yhat
    ) {
        /*
        dL/dM[feature',target',task'] =
            = 2 * SUM_obs (
                (SUM_feature M[feature,target',task'] * X[feature,task',obs])
                - Y[target',task',obs]
            ) * X[feature',task',obs]
        */
        Tensor3D<Number> loss{1, 1, 1};

        const auto M_data = M.data();
        const auto Y_data = Y.data();
        const auto Yhat_data = Yhat.data();

        Number error_loss{0};
        for (int i = 0; i < spec_.size_Y_; ++i) {
            const auto error = Yhat_data[i] - Y_data[i];
            error_loss += error * error;
        }

        Number l1_loss = 0;
        for (int i = 0; i < spec_.size_M_; ++i) {
            l1_loss += abs(M_data[i]);
        }
        Number l2_loss = 0;
        for (int i = 0; i < spec_.size_M_; ++i) {
            l2_loss += M_data[i] * M_data[i];
        }
        const Number regularization_loss = lambda_ * (alpha_ * l1_loss + (Number(1) - alpha_) * l2_loss);

        loss.at(0, 0, 0) = error_loss + regularization_loss;
        return loss;
    }
};

static_assert(Check_tensor3d_kernel_3In_1Out_template<Glm_loss_myyhat_kernel>::check_passed, "Glm_loss_myyhat_kernel is not a valid 3In1Out kernel template");
