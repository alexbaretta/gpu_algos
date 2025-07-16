// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/glm/glm_predict_naive.hpp

#pragma once
#include <cassert>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/type_traits.cuh"
#include "cuda_utils.cuh"
#include "cuda/kernel_api/tensor3d_2in_1out.cuh"

/*
This kernel evaluates a Tensor3Dlinear model on a Tensor3D of covariates, resulting in a Tensor3D of predictions.

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
*/

namespace glm {

    template <CUDA_scalar Number>
    using transform_ptr = Number (*const)(Number);

    template <CUDA_scalar Number>
    __global__ void glm_predict_block(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        Number* const Yhat,        // (ntargets, ntasks, nobs)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const transform_ptr<Number> transform
    ) {
        // We need one word of dynamic shm per warp per block
        Number* shm  = get_dynamic_shared_memory<Number>();

        assert(blockDim.x % WARP_SIZE == 0); // Number of threads is a multiple of a warp
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one block per location in Yhat
        const auto& bid_grid = blockIdx.x; // We use a 1D grid
        const auto& tid_block = threadIdx.x; // We use a 1D block
        const auto tid_warp = threadIdx.x % WARP_SIZE;
        const auto wid_block = threadIdx.x / WARP_SIZE;
        const auto n_warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto Y_output_size = Y_sheet_size * nobs;

        const auto Yhat_idx_per_block = blockDim.x / WARP_SIZE;
        assert(Yhat_idx_per_block > 0 && blockDim.x % WARP_SIZE == 0);
        const auto& nfeatures_per_block = WARP_SIZE;
        assert(nfeatures_per_block > 0 && nfeatures_per_block == WARP_SIZE);

        for (auto Yhat_idx = bid_grid; Yhat_idx < Y_output_size; Yhat_idx += Yhat_idx_per_block) {
            const auto dst_obs = Yhat_idx / Y_sheet_size;
            const auto dst_obs_idx = Yhat_idx % Y_sheet_size;
            const auto dst_task = dst_obs_idx / ntargets;
            const auto dst_target = dst_obs_idx % ntargets;

            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs']
            Number sum_feature{0};
            for (long feature = tid_block; feature < nfeatures; feature += nfeatures_per_block) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[feature + dst_target * nfeatures + dst_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[feature + dst_task * nfeatures + dst_obs * X_sheet_size]
                );
            }
            // Now we need to reduce/sum over the threads of the block to get the aggregate sum_feature
            // First a warp shuffle reduction down to lane 0
            for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                sum_feature += __shfl_down_sync(__activemask(), sum_feature, reduced_lanes);
            }
            // Next we perform a block-level reduction via shm
            // Only lane 0 posts its sum_feature value to shm
            if (tid_warp == 0) { shm[wid_block] = sum_feature; }
            __syncthreads();

            // Finally, we sum over shared memory using a single warp
            if (wid_block == 0) {
                sum_feature = (tid_warp < n_warps_per_block) ? shm[tid_warp] : Number(0);
                for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                    sum_feature += __shfl_down_sync(__activemask(), sum_feature, reduced_lanes);
                }
                if (tid_warp == 0) {
                    Yhat[Yhat_idx] = sum_feature;
                }
            }
        }
    }

    template <CUDA_scalar Number>
    __global__ void glm_predict_warp(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        Number* const Yhat,        // (ntargets, ntasks, nobs)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const transform_ptr<Number> transform
    ) {
        assert(blockDim.x % WARP_SIZE == 0); // Number of threads is a multiple of a warp
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one block per location in Yhat
        const auto tid_warp = threadIdx.x % WARP_SIZE;
        const auto tid_grid = threadIdx.x + blockIdx.x * blockDim.x;
        const auto wid_grid = tid_grid / WARP_SIZE;
        const auto nthreads_per_grid = blockDim.x * gridDim.x;
        const auto nwarps_per_grid = nthreads_per_grid / WARP_SIZE;
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto Y_output_size = Y_sheet_size * nobs;

        for (auto Yhat_idx = wid_grid; Yhat_idx < Y_output_size; Yhat_idx += nwarps_per_grid) {
            const auto dst_obs = Yhat_idx / Y_sheet_size;
            const auto dst_obs_idx = Yhat_idx % Y_sheet_size;
            const auto dst_task = dst_obs_idx / ntargets;
            const auto dst_target = dst_obs_idx % ntargets;

            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs']
            Number sum_feature{0};
            for (long feature = tid_warp; feature < nfeatures; feature += WARP_SIZE) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[feature + dst_target * nfeatures + dst_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[feature + dst_task * nfeatures + dst_obs * X_sheet_size]
                );
            }
            // We need to reduce/sum over the threads of the block to get the aggregate sum_feature
            // First a warp shuffle reduction down to lane 0
            for (int reduced_lanes = 1; reduced_lanes < WARP_SIZE; reduced_lanes <<= 1) {
                sum_feature += __shfl_down_sync(__activemask(), sum_feature, reduced_lanes);
            }
            if (tid_warp == 0) {
                Yhat[Yhat_idx] = sum_feature;
            }
        }
    }

    template <CUDA_scalar Number, int COALESCED_ACCESS_SIZE = 32 /* bytes*/, int GROUP_SIZE = COALESCED_ACCESS_SIZE/sizeof(Number)>
    __global__ void glm_predict_group(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        Number* const Yhat,        // (ntargets, ntasks, nobs)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const transform_ptr<Number> transform
    ) {
        assert(blockDim.x % GROUP_SIZE == 0); // Number of threads is a multiple of a group
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);

        // We use one block per location in Yhat
        const auto tid_group = threadIdx.x % GROUP_SIZE;
        const auto tid_grid = threadIdx.x + blockIdx.x * blockDim.x;
        const auto groupid_grid = tid_grid / GROUP_SIZE;
        const auto nthreads_per_grid = blockDim.x * gridDim.x;
        const auto nwarps_per_grid = nthreads_per_grid / GROUP_SIZE;
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto Y_output_size = Y_sheet_size * nobs;

        for (auto Yhat_idx = groupid_grid; Yhat_idx < Y_output_size; Yhat_idx += nwarps_per_grid) {
            const auto dst_obs = Yhat_idx / Y_sheet_size;
            const auto dst_obs_idx = Yhat_idx % Y_sheet_size;
            const auto dst_task = dst_obs_idx / ntargets;
            const auto dst_target = dst_obs_idx % ntargets;

            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs']
            Number sum_feature{0};
            for (long feature = tid_group; feature < nfeatures; feature += GROUP_SIZE) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[feature + dst_target * nfeatures + dst_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[feature + dst_task * nfeatures + dst_obs * X_sheet_size]
                );
            }
            // We need to reduce/sum over the threads of the block to get the aggregate sum_feature
            // First a warp shuffle reduction down to lane 0
            for (int reduced_lanes = 1; reduced_lanes < GROUP_SIZE; reduced_lanes <<= 1) {
                sum_feature += __shfl_down_sync(__activemask(), sum_feature, reduced_lanes);
            }
            if (tid_group == 0) {
                Yhat[Yhat_idx] = sum_feature;
            }
        }
    }


    template <CUDA_scalar Number>
    __global__ void glm_predict_naive(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        Number* const Yhat,        // (ntargets, ntasks, nobs)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const transform_ptr<Number> transform
    ) {
        // We use one thread per location in Yhat
        const auto& tid_grid = threadIdx.x + blockIdx.x * blockDim.x; // We use a 1D block
        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;
        const auto Y_output_size = Y_sheet_size * nobs;

        for (auto Yhat_idx = tid_grid; Yhat_idx < Y_output_size; Yhat_idx += gridDim.x) {
            const auto dst_obs = Yhat_idx / Y_sheet_size;
            const auto dst_obs_idx = Yhat_idx % Y_sheet_size;
            const auto dst_task = dst_obs_idx / ntargets;
            const auto dst_target = dst_obs_idx % ntargets;

            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs]
            Number sum_feature{0};
            for (long feature = 0; feature < nfeatures; feature += 1) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[feature + dst_target * nfeatures + dst_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[feature + dst_task * nfeatures + dst_obs * X_sheet_size]
                );
            }
            Yhat[Yhat_idx] = sum_feature;
        }
    }

    template <CUDA_scalar Number>
    __global__ void glm_predict_fixed_grid(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        Number* const Yhat,        // (ntargets, ntasks, nobs)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const transform_ptr<Number> transform
    ) {
        // We use one thread per location in Yhat
        const auto dst_target = blockIdx.x * blockDim.x + threadIdx.x;
        const auto dst_task = blockIdx.y * blockDim.y + threadIdx.y;
        const auto dst_obs = blockIdx.z * blockDim.z + threadIdx.z;
        if (dst_target < ntargets && dst_task < ntasks && dst_obs < nobs) {
            const auto X_sheet_size = nfeatures * ntasks;
            const auto Y_sheet_size = ntargets * ntasks;
            const auto M_sheet_size = nfeatures * ntargets;

            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs]
            Number sum_feature{0};
            for (long feature = 0; feature < nfeatures; feature += 1) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[feature + dst_target * nfeatures + dst_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[feature + dst_task * nfeatures + dst_obs * X_sheet_size]
                );
            }
            Yhat[dst_target + dst_task * ntargets + dst_obs * Y_sheet_size] = sum_feature;
        }
    }
} // namespace glm


struct Glm_predict_naive_spec {
    const std::string type_;
    const std::string gpu_algo_;

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

    // Yhat tensor (predictions) with dimensions (ntargets, ntasks, nobs)
    const long n_cols_C_;   // ntargets
    const long n_rows_C_;   // ntasks
    const long n_sheets_C_; // nobs

    const long n_cols_temp_;
    const long n_rows_temp_;
    const long n_sheets_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_;

    constexpr static long DEFAULT_NOBS = 1000;
    constexpr static long DEFAULT_NTASKS = 10;
    constexpr static long DEFAULT_NTARGETS = 25;
    constexpr static long DEFAULT_NFEATURES = 25;
    constexpr static long DEFAULT_BLOCK_DIM = 32;
    constexpr static std::string DEFAULT_GPU_ALGO = "naive";

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("nfeatures,X", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NFEATURES)))
            ("ntargets,Y", "Number of targets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTARGETS)))
            ("ntasks,T", "Number of tasks", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTASKS)))
            ("nobs,N", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NOBS)))
            ("block-dim,n", "Number of threads per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("gpu-algo", "GPU algo to use (fixed-grid, naive, vector, warp, group, block)", cxxopts::value<std::string>()->default_value(DEFAULT_GPU_ALGO))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
    }

    inline static Glm_predict_naive_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate gpu_algo
        const auto gpu_algo = options_parsed["gpu-algo"].as<std::string>();
        if (gpu_algo != "fixed-grid" && gpu_algo != "naive" && gpu_algo != "group" && gpu_algo != "warp" && gpu_algo != "block") {
            std::cerr << "[ERROR] --gpu-algo must be one of: fixed-grid, naive, warp, group, block" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --gpu-algo: " + gpu_algo);
        }
        // Validate the type option
        const auto type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        // Validate block-dim
        const auto block_dim = options_parsed["block-dim"].as<long>();
        if (gpu_algo == "fixed-grid" && block_dim > 10) {
            std::cerr << "[ERROR] --block-dim " << block_dim << " too big when --gpu-algo fixed-grid (must be <= 10)" << std::endl;
            throw cxxopts::exceptions::exception("--block-dim " + std::to_string(block_dim) + " too big when --gpu-algo fixed-grid (must be <= 10)");
        }
        return {
            type,
            gpu_algo,
            options_parsed["nfeatures"].as<long>(),
            options_parsed["ntargets"].as<long>(),
            options_parsed["ntasks"].as<long>(),
            options_parsed["nobs"].as<long>(),
            block_dim,
        };
    }

    inline Glm_predict_naive_spec(
        const std::string& type,
        const std::string& gpu_algo,
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const long block_dim
    ) : type_(type),
        gpu_algo_(gpu_algo),
        nfeatures_(nfeatures),
        ntargets_(ntargets),
        ntasks_(ntasks),
        nobs_(nobs),
        niterations_(nobs * ntasks * ntargets),

        n_cols_A_(nfeatures),
        n_rows_A_(ntasks),
        n_sheets_A_(nobs),

        n_cols_B_(nfeatures),
        n_rows_B_(ntargets),
        n_sheets_B_(ntasks),

        n_cols_C_(ntargets),
        n_rows_C_(ntasks),
        n_sheets_C_(nobs),

        n_cols_temp_(0),
        n_rows_temp_(0),
        n_sheets_temp_(0),

        block_dim_(gpu_algo == "fixed-grid" ? dim3(block_dim, block_dim, block_dim) : dim3(block_dim)),
        grid_dim_(gpu_algo == "fixed-grid" ? dim3(
            (ntargets + block_dim - 1)/block_dim,
            (ntasks + block_dim - 1)/block_dim,
            (nobs + block_dim - 1)/block_dim
        ) : dim3(niterations_ * 32)),
        dynamic_shared_mem_words_(block_dim)
    {
        assert(block_dim_.x > 0);
        assert(block_dim_.y > 0);
        assert(block_dim_.z > 0);

        assert(grid_dim_.x > 0);
        assert(grid_dim_.y > 0);
        assert(grid_dim_.z > 0);
    }
};

static_assert(Check_tensor3d_kernel_spec_2In_1Out<Glm_predict_naive_spec>::check_passed, "Glm_predict_naive_spec is not a valid 2In1Out kernel spec");

template <CUDA_scalar Number_>
class Glm_predict_naive_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Glm_predict_naive_spec;

    const Kernel_spec spec_;

    Glm_predict_naive_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_X,
        // const Number* const gpu_data_Y,
        const Number* const gpu_data_M,
        Number* const gpu_data_Yhat,
        Number* const gpu_data_temp,     // temporary storage (unused)
        cudaStream_t stream
    ) {
        if (spec_.gpu_algo_ == "fixed-grid") {
            const int dynamic_shared_mem_words = 0;
            std::cout << "[INFO] kernel launch: glm::glm_predict_fixed_grid<<<(" << spec_.grid_dim_.x << ", " << spec_.grid_dim_.y << ", " << spec_.grid_dim_.z << "), ("
                << spec_.block_dim_.x << ", " << spec_.block_dim_.y << ", " << spec_.block_dim_.z << "), " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_predict_fixed_grid<<<
                spec_.grid_dim_,
                spec_.block_dim_,
                dynamic_shared_mem_words * sizeof(Number),
                stream
            >>>(
                gpu_data_X,
                // gpu_data_Y,
                gpu_data_M,
                gpu_data_Yhat,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
                static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
            );
        } else if (spec_.gpu_algo_ == "naive") {
            const int dynamic_shared_mem_words = 0;
            std::cout << "[INFO] kernel launch: glm::glm_predict_naive<<<" << spec_.grid_dim_.x << ", " << spec_.block_dim_.x << ", " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_predict_naive<<<
                spec_.grid_dim_,
                spec_.block_dim_,
                dynamic_shared_mem_words * sizeof(Number),
                stream
            >>>(
                gpu_data_X,
                // gpu_data_Y,
                gpu_data_M,
                gpu_data_Yhat,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
                static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
            );
        } else if (spec_.gpu_algo_ == "group") {
            const int dynamic_shared_mem_words = 0;
            std::cout << "[INFO] kernel launch: glm::glm_predict_group<<<" << spec_.grid_dim_.x << ", " << spec_.block_dim_.x << ", " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_predict_group<<<
                spec_.grid_dim_,
                spec_.block_dim_,
                dynamic_shared_mem_words * sizeof(Number),
                stream
            >>>(
                gpu_data_X,
                // gpu_data_Y,
                gpu_data_M,
                gpu_data_Yhat,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
                static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
            );
        } else if (spec_.gpu_algo_ == "warp") {
            const int dynamic_shared_mem_words = 0;
            std::cout << "[INFO] kernel launch: glm::glm_predict_warp<<<" << spec_.grid_dim_.x << ", " << spec_.block_dim_.x << ", " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_predict_warp<<<
                spec_.grid_dim_,
                spec_.block_dim_,
                dynamic_shared_mem_words * sizeof(Number),
                stream
            >>>(
                gpu_data_X,
                // gpu_data_Y,
                gpu_data_M,
                gpu_data_Yhat,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
                static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
            );
        } else if (spec_.gpu_algo_ == "block") {
            const int dynamic_shared_mem_words = compute_n_warps_per_block(spec_.block_dim_);
            std::cout << "[INFO] kernel launch: glm::glm_predict_block<<<" << spec_.grid_dim_.x << ", " << spec_.block_dim_.x << ", " << dynamic_shared_mem_words
                << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
            std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
            glm::glm_predict_block<<<
                spec_.grid_dim_,
                spec_.block_dim_,
                dynamic_shared_mem_words * sizeof(Number),
                stream
            >>>(
                gpu_data_X,
                // gpu_data_Y,
                gpu_data_M,
                gpu_data_Yhat,
                spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
                static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
            );
        } else {
            std::cerr << "[ERROR] --gpu-algo must be one of: fixed-grid, naive, warp, group, block" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --gpu-algo: " + spec_.gpu_algo_);
            assert(false); // We should have printed this error when parsing the command line
        }
    }

    Tensor3D<Number> run_host_kernel(
        const Tensor3D<Number>& X,
        // const Tensor3D<Number>& Y,
        const Tensor3D<Number>& M
    ) {
        Tensor3D<Number> Yhat{spec_.ntargets_, spec_.ntasks_, spec_.nobs_};

        #pragma omp parallel for
        for (int obs = 0; obs < spec_.nobs_; ++obs) {
            for (int task = 0; task < spec_.ntasks_; ++task) {
                for (int target = 0; target < spec_.ntargets_; ++target) {
                    Yhat.at(target, task, obs) = X.row_at(task, obs).dot(M.row_at(target, task));
                }
            }
        }
        return Yhat;
    }
};

static_assert(Check_tensor3d_kernel_2In_1Out_template<Glm_predict_naive_kernel>::check_passed, "Glm_predict_naive_kernel is not a valid 2In1Out kernel template");
