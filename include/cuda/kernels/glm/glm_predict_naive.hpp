// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

// source path: include/cuda/kernels/glm/glm_predict_naive.hpp

#pragma once
#include <cassert>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "common/types/tensor3d.hpp"
#include "cuda/kernel_api/tensor3d_2in_1out.hpp"
#include "cuda/type_traits.hpp"

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
        // Number of warps: nobs * ntasks * ntargets
        // We use 1 warp to sum over k in 1..nfeatures.
        // -> Number of threads = nobs * ntasks * ntargets * WARP_SIZE
        const auto tid_grid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto wid_grid = tid_grid / WARP_SIZE;
        const auto tid_warp = tid_grid % WARP_SIZE;
        const auto Y_output_size = nobs * ntasks * ntargets;
        const auto grid_size = gridDim.x * blockDim.x; // Must be a multiple of WARP_SIZE!
        const auto Y_sheet_size = ntasks * nobs;
        assert(grid_size % WARP_SIZE == 0);
        const auto nwarps = grid_size / WARP_SIZE;
        for (auto Y_idx = wid_grid; Y_idx < Y_output_size; Y_idx += nwarps) {
            const auto obs  = Y_idx / Y_sheet_size;      // h = obs
            const auto obs_idx = Y_idx % Y_sheet_size;   // idx in the `obs`-th sheet
            const auto task = obs_idx / ntargets;      // i = task
            const auto target = obs_idx % ntargets;  // idx in the `task`-th row

            const auto X_row_idx = obs * ntasks * nfeatures    + task * nfeatures;
            const auto M_row_idx = task * ntargets * nfeatures + target * nfeatures;
            // const auto Y_idx     = obs * ntasks * ntargets     + task * ntargets     + target;

            // Compute Ŷ[obs, task, target] = ∑_k M[task, target, feature] * X[obs, task, feature]
            Number lane_value{0};
            for (int feature = tid_warp; feature < nfeatures; feature += WARP_SIZE) {
                // k = feature
                lane_value += M[M_row_idx + feature] * X[X_row_idx + feature];
            }

            // Now we need to do warp shuffle sum
            // Unrolling lane0:
            // n = 1 : lane_value += lane1
            // n = 2 : lane_value += (lane2 + lane3)
            // n = 4 : lane_value += (lane4 + lane5 + lane6 + lane7)
            // n = 8 : lane_value += (lane8 + ... + lane15)
            // n = 16: lane_value += (lane16 + ... + lane31)
            for (int n = 1; n < WARP_SIZE; n <<= 1) {
                lane_value += __shfl_sync(__activemask(), lane_value, tid_warp + n);
            }
            // Now lane_value contains Ŷ[obs, task, target]
            if (transform) {
                Yhat[Y_idx] = transform(lane_value);
            } else {
                Yhat[Y_idx] = lane_value;
            }
        }
    }

} // namespace glm

struct Glm_predict_naive_spec {
    const std::string type_;

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
    const size_t dynamic_shared_mem_words_ = 0;

    constexpr static long DEFAULT_NOBS = 1000;
    constexpr static long DEFAULT_NTASKS = 10;
    constexpr static long DEFAULT_NTARGETS = 25;
    constexpr static long DEFAULT_NFEATURES = 25;
    constexpr static long DEFAULT_BLOCK_DIM = 256;

    inline static void add_kernel_spec_options(cxxopts::Options& options) {
        options.add_options()
            ("nfeatures,X", "Number of features", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NFEATURES)))
            ("ntargets,Y", "Number of targets", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTARGETS)))
            ("ntasks,T", "Number of tasks", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NTASKS)))
            ("nobs,N", "Number of observations", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_NOBS)))
            ("block_dim,n", "Number of threads per block", cxxopts::value<long>()->default_value(std::to_string(DEFAULT_BLOCK_DIM)))
            ("type", "Numeric type (half, single/float, double, int<n>, uint<n>)", cxxopts::value<std::string>()->default_value("float"));
    }

    inline static Glm_predict_naive_spec make(
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
            options_parsed["block_dim"].as<long>(),
        };
    }

    inline Glm_predict_naive_spec(
        const std::string& type,
        const long nfeatures,
        const long ntargets,
        const long ntasks,
        const long nobs,
        const long block_dim
    ) : type_(type),
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

        block_dim_(block_dim),
        grid_dim_((niterations_ + block_dim - 1)/ block_dim * 32)
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
        std::cout << "[INFO] kernel launch: glm::glm_predict_naive<<<" << spec_.grid_dim_.x << ", " << spec_.block_dim_.x << ", " << spec_.dynamic_shared_mem_words_ * sizeof(Number)
            << ">>>(..., " << spec_.nfeatures_ << ", " << spec_.ntargets_ << ", " << spec_.ntasks_ << ", " << spec_.nobs_ << ")" << std::endl;
        std::cout << "[INFO] niterations = " << spec_.nobs_ * spec_.ntasks_ * spec_.ntargets_ << std::endl;
        glm::glm_predict_naive<<<
            spec_.grid_dim_,
            spec_.block_dim_,
            spec_.dynamic_shared_mem_words_ * sizeof(Number),
            stream
        >>>(
            gpu_data_X,
            // gpu_data_Y,
            gpu_data_M,
            gpu_data_Yhat,
            spec_.nfeatures_, spec_.ntargets_, spec_.ntasks_, spec_.nobs_,
            static_cast<glm::transform_ptr<Number>>(nullptr) // Transform: logistic or other activation function
        );
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
