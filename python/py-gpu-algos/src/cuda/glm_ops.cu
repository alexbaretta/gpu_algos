/*
 * CUDA GLM Operations Bindings for py-gpu-algos
 *
 * This module provides Python bindings for GLM operations using pybind11.
 * It instantiates GLM kernels for 3D tensor operations with all supported numeric types.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

// Include CUDA kernel headers
#include "cuda/kernels/glm/glm_predict_naive.cuh"
#include "cuda/kernels/glm/glm_gradient_naive.cuh"
#include "cuda/kernels/glm/glm_gradient_xyyhat.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/check_errors.cuh"
#include "common/types/tensor3d.hpp"

namespace py = pybind11;

// Helper function to convert numpy 3D array to Tensor3D
template<typename T>
Tensor3D<T> numpy_to_tensor3d(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 3) {
        throw std::invalid_argument("Array must be 3-dimensional for Tensor3D conversion");
    }

    long cols = buf.shape[0];
    long rows = buf.shape[1];
    long sheets = buf.shape[2];

    Tensor3D<T> tensor(cols, rows, sheets);
    const T* src = static_cast<const T*>(buf.ptr);

    // Copy data from numpy array to Tensor3D
    for (long sheet = 0; sheet < sheets; ++sheet) {
        for (long row = 0; row < rows; ++row) {
            for (long col = 0; col < cols; ++col) {
                long numpy_idx = col * rows * sheets + row * sheets + sheet;
                tensor.at(col, row, sheet) = src[numpy_idx];
            }
        }
    }

    return tensor;
}

// Helper function to convert Tensor3D to numpy 3D array
template<typename T>
py::array_t<T> tensor3d_to_numpy(const Tensor3D<T>& tensor) {
    long cols = tensor.cols();
    long rows = tensor.rows();
    long sheets = tensor.sheets();

    auto result = py::array_t<T>({cols, rows, sheets});
    auto buf = result.request();
    T* dst = static_cast<T*>(buf.ptr);

    // Copy data from Tensor3D to numpy array
    for (long sheet = 0; sheet < sheets; ++sheet) {
        for (long row = 0; row < rows; ++row) {
            for (long col = 0; col < cols; ++col) {
                long numpy_idx = col * rows * sheets + row * sheets + sheet;
                dst[numpy_idx] = tensor.at(col, row, sheet);
            }
        }
    }

    return result;
}

// Helper function to launch glm_predict_naive kernels
template<typename T>
py::array_t<T> glm_predict_naive_cuda_impl(
    const py::array_t<T>& X,  // (nfeatures, ntasks, nobs)
    const py::array_t<T>& M   // (nfeatures, ntargets, ntasks)
) {
    // Validate input arrays
    auto X_buf = X.request();
    auto M_buf = M.request();

    if (X_buf.ndim != 3 || M_buf.ndim != 3) {
        throw std::invalid_argument("Input arrays must be 3-dimensional");
    }

    if (!X.flags() & py::array::c_style) {
        throw std::invalid_argument("Array X must be C-contiguous");
    }

    if (!M.flags() & py::array::c_style) {
        throw std::invalid_argument("Array M must be C-contiguous");
    }

    // Get tensor dimensions
    long nfeatures = X_buf.shape[0];
    long ntasks = X_buf.shape[1];
    long nobs = X_buf.shape[2];

    long nfeatures_M = M_buf.shape[0];
    long ntargets = M_buf.shape[1];
    long ntasks_M = M_buf.shape[2];

    // Validate dimensions
    if (nfeatures != nfeatures_M || ntasks != ntasks_M) {
        throw std::invalid_argument("Incompatible tensor dimensions between X and M");
    }

    // Create output array Yhat: (ntargets, ntasks, nobs)
    auto result = py::array_t<T>({ntargets, ntasks, nobs});
    auto result_buf = result.request();

    // Get data pointers
    const T* X_ptr = static_cast<const T*>(X_buf.ptr);
    const T* M_ptr = static_cast<const T*>(M_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Glm_predict_naive_spec spec(
        "float",      // Type string (will be overridden by template parameter)
        "fixed-grid", // GPU algorithm
        nfeatures, ntargets, ntasks, nobs,
        8            // Default block dimension
    );

    // Create kernel instance
    Glm_predict_naive_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_X = nullptr;
    T* d_M = nullptr;
    T* d_result = nullptr;

    size_t size_X = nfeatures * ntasks * nobs * sizeof(T);
    size_t size_M = nfeatures * ntargets * ntasks * sizeof(T);
    size_t size_result = ntargets * ntasks * nobs * sizeof(T);

    cuda_check_error(cudaMalloc(&d_X, size_X), "cudaMalloc for tensor X");
    cuda_check_error(cudaMalloc(&d_M, size_M), "cudaMalloc for tensor M");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result tensor");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_X, X_ptr, size_X, cudaMemcpyHostToDevice), "cudaMemcpy X to device");
        cuda_check_error(cudaMemcpy(d_M, M_ptr, size_M, cudaMemcpyHostToDevice), "cudaMemcpy M to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_X, d_M, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_X) cudaFree(d_X);
        if (d_M) cudaFree(d_M);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_X), "cudaFree X");
    cuda_check_error(cudaFree(d_M), "cudaFree M");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch glm_gradient_naive kernels
template<typename T>
py::array_t<T> glm_gradient_naive_cuda_impl(
    const py::array_t<T>& X,  // (nfeatures, ntasks, nobs)
    const py::array_t<T>& Y,  // (ntargets, ntasks, nobs)
    const py::array_t<T>& M   // (nfeatures, ntargets, ntasks)
) {
    // Validate input arrays
    auto X_buf = X.request();
    auto Y_buf = Y.request();
    auto M_buf = M.request();

    if (X_buf.ndim != 3 || Y_buf.ndim != 3 || M_buf.ndim != 3) {
        throw std::invalid_argument("Input arrays must be 3-dimensional");
    }

    if (!X.flags() & py::array::c_style) {
        throw std::invalid_argument("Array X must be C-contiguous");
    }

    if (!Y.flags() & py::array::c_style) {
        throw std::invalid_argument("Array Y must be C-contiguous");
    }

    if (!M.flags() & py::array::c_style) {
        throw std::invalid_argument("Array M must be C-contiguous");
    }

    // Get tensor dimensions
    long nfeatures = X_buf.shape[0];
    long ntasks = X_buf.shape[1];
    long nobs = X_buf.shape[2];

    long ntargets = Y_buf.shape[0];
    long ntasks_Y = Y_buf.shape[1];
    long nobs_Y = Y_buf.shape[2];

    long nfeatures_M = M_buf.shape[0];
    long ntargets_M = M_buf.shape[1];
    long ntasks_M = M_buf.shape[2];

    // Validate dimensions
    if (ntasks != ntasks_Y || nobs != nobs_Y) {
        throw std::invalid_argument("Incompatible tensor dimensions between X and Y");
    }
    if (nfeatures != nfeatures_M || ntargets != ntargets_M || ntasks != ntasks_M) {
        throw std::invalid_argument("Incompatible tensor dimensions between tensors and M");
    }

    // Create output array grad_M: (nfeatures, ntargets, ntasks)
    auto result = py::array_t<T>({nfeatures, ntargets, ntasks});
    auto result_buf = result.request();

    // Get data pointers
    const T* X_ptr = static_cast<const T*>(X_buf.ptr);
    const T* Y_ptr = static_cast<const T*>(Y_buf.ptr);
    const T* M_ptr = static_cast<const T*>(M_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Glm_gradient_naive_spec spec(
        "float",      // Type string (will be overridden by template parameter)
        "nested-loop", // CPU algorithm
        nfeatures, ntargets, ntasks, nobs,
        256,         // Block dimension
        false        // Don't optimize launch
    );

    // Create kernel instance
    Glm_gradient_naive_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_X = nullptr;
    T* d_Y = nullptr;
    T* d_M = nullptr;
    T* d_result = nullptr;

    size_t size_X = nfeatures * ntasks * nobs * sizeof(T);
    size_t size_Y = ntargets * ntasks * nobs * sizeof(T);
    size_t size_M = nfeatures * ntargets * ntasks * sizeof(T);
    size_t size_result = nfeatures * ntargets * ntasks * sizeof(T);

    cuda_check_error(cudaMalloc(&d_X, size_X), "cudaMalloc for tensor X");
    cuda_check_error(cudaMalloc(&d_Y, size_Y), "cudaMalloc for tensor Y");
    cuda_check_error(cudaMalloc(&d_M, size_M), "cudaMalloc for tensor M");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result tensor");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_X, X_ptr, size_X, cudaMemcpyHostToDevice), "cudaMemcpy X to device");
        cuda_check_error(cudaMemcpy(d_Y, Y_ptr, size_Y, cudaMemcpyHostToDevice), "cudaMemcpy Y to device");
        cuda_check_error(cudaMemcpy(d_M, M_ptr, size_M, cudaMemcpyHostToDevice), "cudaMemcpy M to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_X, d_Y, d_M, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_X) cudaFree(d_X);
        if (d_Y) cudaFree(d_Y);
        if (d_M) cudaFree(d_M);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_X), "cudaFree X");
    cuda_check_error(cudaFree(d_Y), "cudaFree Y");
    cuda_check_error(cudaFree(d_M), "cudaFree M");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch glm_gradient_xyyhat kernels
template<typename T>
py::array_t<T> glm_gradient_xyyhat_cuda_impl(
    const py::array_t<T>& X,  // (nfeatures, ntasks, nobs)
    const py::array_t<T>& Y,  // (ntargets, ntasks, nobs)
    const py::array_t<T>& M   // (nfeatures, ntargets, ntasks)
) {
    // Validate input arrays
    auto X_buf = X.request();
    auto Y_buf = Y.request();
    auto M_buf = M.request();

    if (X_buf.ndim != 3 || Y_buf.ndim != 3 || M_buf.ndim != 3) {
        throw std::invalid_argument("Input arrays must be 3-dimensional");
    }

    if (!X.flags() & py::array::c_style) {
        throw std::invalid_argument("Array X must be C-contiguous");
    }

    if (!Y.flags() & py::array::c_style) {
        throw std::invalid_argument("Array Y must be C-contiguous");
    }

    if (!M.flags() & py::array::c_style) {
        throw std::invalid_argument("Array M must be C-contiguous");
    }

    // Get tensor dimensions
    long nfeatures = X_buf.shape[0];
    long ntasks = X_buf.shape[1];
    long nobs = X_buf.shape[2];

    long ntargets = Y_buf.shape[0];
    long ntasks_Y = Y_buf.shape[1];
    long nobs_Y = Y_buf.shape[2];

    long nfeatures_M = M_buf.shape[0];
    long ntargets_M = M_buf.shape[1];
    long ntasks_M = M_buf.shape[2];

    // Validate dimensions
    if (ntasks != ntasks_Y || nobs != nobs_Y) {
        throw std::invalid_argument("Incompatible tensor dimensions between X and Y");
    }
    if (nfeatures != nfeatures_M || ntargets != ntargets_M || ntasks != ntasks_M) {
        throw std::invalid_argument("Incompatible tensor dimensions between tensors and M");
    }

    // Create output array grad_M: (nfeatures, ntargets, ntasks)
    auto result = py::array_t<T>({nfeatures, ntargets, ntasks});
    auto result_buf = result.request();

    // Get data pointers
    const T* X_ptr = static_cast<const T*>(X_buf.ptr);
    const T* Y_ptr = static_cast<const T*>(Y_buf.ptr);
    const T* M_ptr = static_cast<const T*>(M_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Glm_gradient_xyyhat_spec spec(
        "float",      // Type string (will be overridden by template parameter)
        "fixed-grid", // GPU algorithm
        "nested-loop", // CPU algorithm
        nfeatures, ntargets, ntasks, nobs,
        8            // Block dimension
    );

    // Create kernel instance
    Glm_gradient_xyyhat_kernel<T> kernel(spec);

    // Allocate GPU memory (including temp space for Yhat)
    T* d_X = nullptr;
    T* d_Y = nullptr;
    T* d_M = nullptr;
    T* d_result = nullptr;
    T* d_temp = nullptr;  // For Yhat intermediate results

    size_t size_X = nfeatures * ntasks * nobs * sizeof(T);
    size_t size_Y = ntargets * ntasks * nobs * sizeof(T);
    size_t size_M = nfeatures * ntargets * ntasks * sizeof(T);
    size_t size_result = nfeatures * ntargets * ntasks * sizeof(T);
    size_t size_temp = ntargets * ntasks * nobs * sizeof(T);  // Yhat size

    cuda_check_error(cudaMalloc(&d_X, size_X), "cudaMalloc for tensor X");
    cuda_check_error(cudaMalloc(&d_Y, size_Y), "cudaMalloc for tensor Y");
    cuda_check_error(cudaMalloc(&d_M, size_M), "cudaMalloc for tensor M");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result tensor");
    cuda_check_error(cudaMalloc(&d_temp, size_temp), "cudaMalloc for temp tensor");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_X, X_ptr, size_X, cudaMemcpyHostToDevice), "cudaMemcpy X to device");
        cuda_check_error(cudaMemcpy(d_Y, Y_ptr, size_Y, cudaMemcpyHostToDevice), "cudaMemcpy Y to device");
        cuda_check_error(cudaMemcpy(d_M, M_ptr, size_M, cudaMemcpyHostToDevice), "cudaMemcpy M to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_X, d_Y, d_M, d_result, d_temp, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_X) cudaFree(d_X);
        if (d_Y) cudaFree(d_Y);
        if (d_M) cudaFree(d_M);
        if (d_result) cudaFree(d_result);
        if (d_temp) cudaFree(d_temp);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_X), "cudaFree X");
    cuda_check_error(cudaFree(d_Y), "cudaFree Y");
    cuda_check_error(cudaFree(d_M), "cudaFree M");
    cuda_check_error(cudaFree(d_result), "cudaFree result");
    cuda_check_error(cudaFree(d_temp), "cudaFree temp");

    return result;
}

// High-level dispatch functions
py::object glm_predict_naive_dispatch(py::array X, py::array M) {
    if (!X.dtype().is(M.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (X.dtype().is(py::dtype::of<float>())) {
        return glm_predict_naive_cuda_impl<float>(X.cast<py::array_t<float>>(), M.cast<py::array_t<float>>());
    } else if (X.dtype().is(py::dtype::of<double>())) {
        return glm_predict_naive_cuda_impl<double>(X.cast<py::array_t<double>>(), M.cast<py::array_t<double>>());
    } else if (X.dtype().is(py::dtype::of<std::int32_t>())) {
        return glm_predict_naive_cuda_impl<std::int32_t>(X.cast<py::array_t<std::int32_t>>(), M.cast<py::array_t<std::int32_t>>());
    } else if (X.dtype().is(py::dtype::of<std::int64_t>())) {
        return glm_predict_naive_cuda_impl<std::int64_t>(X.cast<py::array_t<std::int64_t>>(), M.cast<py::array_t<std::int64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for GLM predict: " + py::str(X.dtype()).cast<std::string>());
    }
}

py::object glm_gradient_naive_dispatch(py::array X, py::array Y, py::array M) {
    if (!X.dtype().is(Y.dtype()) || !X.dtype().is(M.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (X.dtype().is(py::dtype::of<float>())) {
        return glm_gradient_naive_cuda_impl<float>(X.cast<py::array_t<float>>(), Y.cast<py::array_t<float>>(), M.cast<py::array_t<float>>());
    } else if (X.dtype().is(py::dtype::of<double>())) {
        return glm_gradient_naive_cuda_impl<double>(X.cast<py::array_t<double>>(), Y.cast<py::array_t<double>>(), M.cast<py::array_t<double>>());
    } else if (X.dtype().is(py::dtype::of<std::int32_t>())) {
        return glm_gradient_naive_cuda_impl<std::int32_t>(X.cast<py::array_t<std::int32_t>>(), Y.cast<py::array_t<std::int32_t>>(), M.cast<py::array_t<std::int32_t>>());
    } else if (X.dtype().is(py::dtype::of<std::int64_t>())) {
        return glm_gradient_naive_cuda_impl<std::int64_t>(X.cast<py::array_t<std::int64_t>>(), Y.cast<py::array_t<std::int64_t>>(), M.cast<py::array_t<std::int64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for GLM gradient: " + py::str(X.dtype()).cast<std::string>());
    }
}

py::object glm_gradient_xyyhat_dispatch(py::array X, py::array Y, py::array M) {
    if (!X.dtype().is(Y.dtype()) || !X.dtype().is(M.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (X.dtype().is(py::dtype::of<float>())) {
        return glm_gradient_xyyhat_cuda_impl<float>(X.cast<py::array_t<float>>(), Y.cast<py::array_t<float>>(), M.cast<py::array_t<float>>());
    } else if (X.dtype().is(py::dtype::of<double>())) {
        return glm_gradient_xyyhat_cuda_impl<double>(X.cast<py::array_t<double>>(), Y.cast<py::array_t<double>>(), M.cast<py::array_t<double>>());
    } else if (X.dtype().is(py::dtype::of<std::int32_t>())) {
        return glm_gradient_xyyhat_cuda_impl<std::int32_t>(X.cast<py::array_t<std::int32_t>>(), Y.cast<py::array_t<std::int32_t>>(), M.cast<py::array_t<std::int32_t>>());
    } else if (X.dtype().is(py::dtype::of<std::int64_t>())) {
        return glm_gradient_xyyhat_cuda_impl<std::int64_t>(X.cast<py::array_t<std::int64_t>>(), Y.cast<py::array_t<std::int64_t>>(), M.cast<py::array_t<std::int64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for GLM gradient XYYhat: " + py::str(X.dtype()).cast<std::string>());
    }
}

// Python module definition
PYBIND11_MODULE(_glm_ops_cuda, m) {
    m.doc() = "CUDA GLM operations for py-gpu-algos";

    // Low-level type-specific functions for glm_predict_naive
    m.def("glm_predict_naive_float32", &glm_predict_naive_cuda_impl<float>,
          "GLM prediction (naive algorithm) for float32", py::arg("X"), py::arg("M"));
    m.def("glm_predict_naive_float64", &glm_predict_naive_cuda_impl<double>,
          "GLM prediction (naive algorithm) for float64", py::arg("X"), py::arg("M"));
    m.def("glm_predict_naive_int32", &glm_predict_naive_cuda_impl<std::int32_t>,
          "GLM prediction (naive algorithm) for int32", py::arg("X"), py::arg("M"));
    m.def("glm_predict_naive_int64", &glm_predict_naive_cuda_impl<std::int64_t>,
          "GLM prediction (naive algorithm) for int64", py::arg("X"), py::arg("M"));

    // Low-level type-specific functions for glm_gradient_naive
    m.def("glm_gradient_naive_float32", &glm_gradient_naive_cuda_impl<float>,
          "GLM gradient (naive algorithm) for float32", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_naive_float64", &glm_gradient_naive_cuda_impl<double>,
          "GLM gradient (naive algorithm) for float64", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_naive_int32", &glm_gradient_naive_cuda_impl<std::int32_t>,
          "GLM gradient (naive algorithm) for int32", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_naive_int64", &glm_gradient_naive_cuda_impl<std::int64_t>,
          "GLM gradient (naive algorithm) for int64", py::arg("X"), py::arg("Y"), py::arg("M"));

    // Low-level type-specific functions for glm_gradient_xyyhat
    m.def("glm_gradient_xyyhat_float32", &glm_gradient_xyyhat_cuda_impl<float>,
          "GLM gradient (XYYhat algorithm) for float32", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_xyyhat_float64", &glm_gradient_xyyhat_cuda_impl<double>,
          "GLM gradient (XYYhat algorithm) for float64", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_xyyhat_int32", &glm_gradient_xyyhat_cuda_impl<std::int32_t>,
          "GLM gradient (XYYhat algorithm) for int32", py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_xyyhat_int64", &glm_gradient_xyyhat_cuda_impl<std::int64_t>,
          "GLM gradient (XYYhat algorithm) for int64", py::arg("X"), py::arg("Y"), py::arg("M"));

    // High-level dispatch functions
    m.def("glm_predict_naive", &glm_predict_naive_dispatch,
          "GLM prediction (naive algorithm) with automatic type dispatch",
          py::arg("X"), py::arg("M"));
    m.def("glm_gradient_naive", &glm_gradient_naive_dispatch,
          "GLM gradient (naive algorithm) with automatic type dispatch",
          py::arg("X"), py::arg("Y"), py::arg("M"));
    m.def("glm_gradient_xyyhat", &glm_gradient_xyyhat_dispatch,
          "GLM gradient (XYYhat algorithm) with automatic type dispatch",
          py::arg("X"), py::arg("Y"), py::arg("M"));
}
