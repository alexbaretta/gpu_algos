// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/glm/glm_gradient_naive.cuh

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
*/

namespace glm {

    /*
    The parallelization strategy here is to allocate one thread per element of gradM.
    Ignoring the need to round up: gridDim = { nfeatures / blockDim.x, ntargets / blockDim.y, ntasks / blockDim.z }
    We index M and gradM as follows:
        M[i_feature][i_target][i_task],
    where i_feature < nfeatures, i_target < ntargets, i_task < ntasks.

    We handle the nobs dimension through a for loop.
        for (int w = 0; w < nobs; ++w)
    */
    template <CUDA_scalar Number>
    __global__ void glm_gradient_naive_fixed_grid(
        const Number* const X,     // (nfeatures, ntasks, nobs)
        const Number* const M,     // (nfeatures, ntargets, ntasks)
        const Number* const Y,     // (ntargets, ntasks, nobs)
        Number* const grad_M,      // (nfeatures, ntargets, ntasks)
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs
    ) {

        const long i_feature = blockIdx.x * blockDim.x + threadIdx.x;
        const long i_target = blockIdx.y * blockDim.y + threadIdx.y;
        const long i_task = blockIdx.z * blockDim.z + threadIdx.z;
        if (i_feature >= nfeatures || i_target >= ntargets || i_task >= ntasks) return;

        const auto X_sheet_size = nfeatures * ntasks;
        const auto Y_sheet_size = ntargets * ntasks;
        const auto M_sheet_size = nfeatures * ntargets;

        const auto M_idx = i_task * M_sheet_size + i_target * nfeatures + i_feature; // This is the index into grad_M
        #ifdef DEBUG
        const auto M_size = nfeatures * ntargets * ntargets;
        if (M_idx >= M_size) {
            printf("M_idx = %ld >= %ld when i_feature=%ld < %ld, i_target=%ld < %ld, i_task=%ld < %ld\n",
                M_idx, M_size, i_feature, nfeatures, i_target, ntargets, i_task, ntasks
            );
        }
        #endif

        Number sum_obs{0};
        for (int i_obs = 0; i_obs < nobs; ++i_obs ) {
            // compute SUM_feature M[feature,target',task'] * X[feature,task',obs]
            Number sum_feature{0};
            for (int j_feature = 0; j_feature < nfeatures; j_feature += 1) {
                // compute M[feature,target',task'] * X[feature,task',obs]
                sum_feature += (
                    // compute M[feature,target',task']
                    M[j_feature + i_target * nfeatures + i_task * M_sheet_size]
                ) * (
                    // compute X[feature,task',obs]
                    X[j_feature + i_task * nfeatures + i_obs * X_sheet_size]
                );
            }

            // compute ( (SUM_feature ...) - Y[target',task',obs]) * X[feature',task',obs]
            sum_obs += (
                // compute ( (SUM_feature ...) - Y[target',task',obs])
                sum_feature - Y[i_target + i_task * ntargets + i_obs * Y_sheet_size]
            ) * (
                // compute X[feature',task',obs]
                X[i_feature + i_task * nfeatures + i_obs * X_sheet_size]
            );
            grad_M[M_idx] = Number(2) * sum_obs;
        }
    }


} // namespace glm

struct Glm_gradient_naive_spec {
    const std::string type_;
    const std::string cpu_algo_;

    const long nfeatures_;
    const long ntargets_;
    const long ntasks_;
    const long nobs_;

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

    // temp tensor, unused
    const long n_cols_temp_;
    const long n_rows_temp_;
    const long n_sheets_temp_;

    const dim3 block_dim_;
    const dim3 grid_dim_;
    const size_t dynamic_shared_mem_words_ = 0;
    const bool optimize_launch_;

    constexpr static long DEFAULT_NOBS = 1000;
    constexpr static long DEFAULT_NTASKS = 10;
    constexpr static long DEFAULT_NTARGETS = 25;
    constexpr static long DEFAULT_NFEATURES = 25;
    constexpr static long DEFAULT_BLOCK_DIM = 8;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("nfeatures,X", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NFEATURES)))
            ("ntargets,Y", "Number of targets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTARGETS)))
            ("ntasks,T", "Number of tasks", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTASKS)))
            ("nobs,N", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NOBS)))
            ("block-dim-x", "Number of threads in the x dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("block-dim-y", "Number of threads in the y dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("block-dim-z", "Number of threads in the z dimension per block", cxxopts::value<unsigned>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("optimize-launch", "Use occupancy API to determine optimal launch configuration")
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"))
            ("cpu-algo", "CPU algo variant to benchmark against (nested-loop, matrix)", cxxopts::value<std::string>()->default_value("matrix"))
            ;
        }

    inline static Glm_gradient_naive_spec make(
        const cxxopts::ParseResult& options_parsed
    ) {
        // Validate the type option
        const auto& type = options_parsed["type"].as<std::string>();
        if (type != "half" && type != "single" && type != "float" && type != "double" && type != "int8" && type != "int16" && type != "int32" && type != "int64" && type != "uint8" && type != "uint16" && type != "uint32" && type != "uint64") {
            std::cerr << "[ERROR] --type must be one of: half, single/float, double, int<n>, uint<n>" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --type: " + type);
        }
        const auto& cpu_algo = options_parsed["cpu-algo"].as<std::string>();
        if (cpu_algo != "nested-loop" && cpu_algo != "matrix") {
            std::cerr << "[ERROR] --nested-loop must be one of: nested-loop, matrix" << std::endl;
            throw cxxopts::exceptions::exception("Invalid --nested-loop: " + cpu_algo);
        }
        const dim3 block_dim{
            options_parsed["block-dim-x"].as<unsigned>(),
            options_parsed["block-dim-y"].as<unsigned>(),
            options_parsed["block-dim-z"].as<unsigned>()
        };
        return {
            type,
            cpu_algo,
            options_parsed["nfeatures"].as<long>(),
            options_parsed["ntargets"].as<long>(),
            options_parsed["ntasks"].as<long>(),
            options_parsed["nobs"].as<long>(),
            block_dim,
            options_parsed.count("optimize-launch") > 0
        };
    }

    inline Glm_gradient_naive_spec(
        const std::string& type,
        const std::string& cpu_algo,
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const dim3 block_dim,
        const bool optimize_launch
    ) : type_(type),
        cpu_algo_(cpu_algo),
        nfeatures_(nfeatures),
        ntargets_(ntargets),
        ntasks_(ntasks),
        nobs_(nobs),

        // X tensor (covariates) with dimensions (nfeatures, ntasks, nobs)
        n_cols_A_(nfeatures),
        n_rows_A_(ntasks),
        n_sheets_A_(nobs),

        // M tensor (model) with dimensions (nfeatures, ntargets, ntasks)
        n_cols_B_(nfeatures),
        n_rows_B_(ntargets),
        n_sheets_B_(ntasks),

        // Y tensor (responses) with dimensions (ntargets, ntasks, nobs)
        n_cols_C_(ntargets),
        n_rows_C_(ntasks),
        n_sheets_C_(nobs),

        // grad_M tensor (model) with dimensions (nfeatures, ntargets, ntasks)
        n_cols_D_(nfeatures),
        n_rows_D_(ntargets),
        n_sheets_D_(ntasks),

        // temp tensor, unused
        n_cols_temp_(0),
        n_rows_temp_(0),
        n_sheets_temp_(0),

        block_dim_(block_dim),
        grid_dim_(
            (nfeatures + block_dim.x - 1) / block_dim.x,
            (ntargets + block_dim.y - 1) / block_dim.y,
            (ntasks + block_dim.z - 1) / block_dim.z
        ),
        optimize_launch_(optimize_launch)
    {
        assert(block_dim_.x > 0);
        assert(block_dim_.y > 0);
        assert(block_dim_.z > 0);

        assert(grid_dim_.x > 0);
        assert(grid_dim_.y > 0);
        assert(grid_dim_.z > 0);
    }
};

static_assert(Check_tensor3d_kernel_spec_3In_1Out<Glm_gradient_naive_spec>::check_passed, "Glm_gradient_naive_spec is not a valid 3In1Out kernel spec");

template <CUDA_scalar Number_>
class Glm_gradient_naive_kernel {
    public:
    using Number = Number_;
    using Kernel_spec = Glm_gradient_naive_spec;

    const Kernel_spec spec_;

    Glm_gradient_naive_kernel(
        const Kernel_spec spec
    ) : spec_(spec) {}

    void run_device_kernel(
        const Number* const gpu_data_X,
        const Number* const gpu_data_M,
        const Number* const gpu_data_Y,
        Number* const gpu_data_grad_M,
        Number* const gpu_data_temp,     // temporary storage (unused)
        cudaStream_t stream
    ) {
        dim3 block_dim;
        dim3 grid_dim;
        if (spec_.optimize_launch_) {
            const int tentative_shm_size = 0;

            int max_block_size = 0;
            int opt_grid_size = 0;
            int max_active_blocks_per_multiprocessor = 0;
            cuda_check_error(cudaOccupancyMaxPotentialBlockSize(
                &max_block_size,
                &opt_grid_size,
                glm::glm_gradient_naive_fixed_grid<Number>,
                tentative_shm_size, // We don't know shm_size yet
                0
            ), "cudaOccupancyMaxPotentialBlockSize");
            cuda_check_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks_per_multiprocessor,
                glm::glm_gradient_naive_fixed_grid<Number>,
                max_block_size,
                tentative_shm_size  // We don't know shm_size yet
            ), "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
            std::cout << "[INFO] max_active_blocks_per_multiprocessor: " << max_active_blocks_per_multiprocessor
                << " at block_size:" << max_block_size << std::endl;
            std::cout << "[INFO] max_block_size: " << max_block_size << std::endl;
            std::cout << "[INFO] opt_grid_size: " << opt_grid_size << std::endl;
            block_dim = dim3(max_block_size);
            grid_dim = dim3(opt_grid_size);
        } else {
            block_dim = spec_.block_dim_;
            grid_dim = spec_.grid_dim_;
        }
        const int shm_size = 0;

        std::cout << "[INFO] grid_dim_: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;
        std::cout << "[INFO] block_dim_: " << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << std::endl;
        std::cout << "[INFO] kernel launch: glm::glm_gradient_naive_fixed_grid<<<("
            << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), ("
            << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), " << shm_size
            << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
        std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
        glm::glm_gradient_naive_fixed_grid<<<
            grid_dim,
            block_dim,
            shm_size,
            stream
        >>>(
            gpu_data_X,
            gpu_data_M,
            gpu_data_Y,
            gpu_data_grad_M,
            spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_
        );
    }

    Tensor3D<Number> run_host_kernel(
        const Tensor3D<Number>& X,
        const Tensor3D<Number>& M,
        const Tensor3D<Number>& Y
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
            for (int i_feature = 0; i_feature < spec_.nfeatures_; ++i_feature) {
                for (int i_target = 0; i_target < spec_.ntargets_; ++i_target) {
                    for (int i_task = 0; i_task < spec_.ntasks_; ++i_task) {
                        Number sum_obs{0};
                        for (int i_obs = 0; i_obs < spec_.nobs_; ++i_obs) {
                            const auto sum_feature = M.row_at(i_target, i_task).dot(X.row_at(i_task, i_obs));
                            sum_obs += (sum_feature - Y.at(i_target, i_task, i_obs)) * X.at(i_feature, i_task, i_obs);
                        }
                        grad_M.at(i_feature, i_target, i_task) = Number(2) * sum_obs;
                    }
                }
            }
        } else {
            // Tensor dimensions: innermost, intermediate, outermost
            // X: (nfeatures, ntasks, nobs)
            // Y: (ntargets, ntasks, nobs)
            // M: (nfeatures, ntargets, ntasks)
            // const Eigen::IOFormat eigen_format(4, 0, ", ", "\n", "  [", "]");
            for (int i_task = 0; i_task < spec_.ntasks_; ++i_task) {
                // Matrix dimensions: [nrows (outer), ncols (inner)]
                auto X_task_matrix = X.chip_at_dim1(i_task);  // tensor:(nfeatures, nobs) -> matrix:[nobs, nfeatures]
                auto Y_task_matrix = Y.chip_at_dim1(i_task);  // tensor:(ntargets,  nobs) -> matrix:[nobs, ntargets]
                auto M_task_matrix = M.sheet_at(i_task);    // tensor:(nfeatures, ntargets) -> matrix:[ntargets, nfeatures]
                auto grad_M_task_matrix = grad_M.sheet_at(i_task); // tensor:(nfeatures, ntargets) -> matrix:[ntargets, nfeatures]
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
            }
        }
        return grad_M;
    }
};

static_assert(Check_tensor3d_kernel_3In_1Out_template<Glm_gradient_naive_kernel>::check_passed, "Glm_gradient_naive_kernel is not a valid 3In1Out kernel template");
