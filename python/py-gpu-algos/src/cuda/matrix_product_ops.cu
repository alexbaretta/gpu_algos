/*
 * CUDA Matrix Product Operations Bindings for py-gpu-algos
 *
 * This module provides Python bindings for matrix product operations using pybind11.
 * It instantiates matrix product kernels for all supported numeric types.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

// Include CUDA kernel headers
#include "cuda/kernels/matrix_product/matrix_product_naive.cuh"
#include "cuda/kernels/matrix_product/matrix_product_tiled.cuh"
#include "cuda/kernels/matrix_product/matrix_product_warp.cuh"
#include "cuda/kernels/matrix_product/matrix_product_cublas.cuh"
#include "cuda/kernels/matrix_product/matrix_product_cutlass.cuh"
#include "cuda/kernels/matrix_product/matrix_product_tensor.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/check_errors.cuh"

namespace py = pybind11;

// Helper function to launch matrix_product_naive kernels
template<typename T>
py::array_t<T> matrix_product_naive_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_naive_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n,
        16, 16    // Default block dimensions
    );

    // Create kernel instance
    Matrix_product_naive_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_product_tiled kernels
template<typename T>
py::array_t<T> matrix_product_tiled_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_tiled_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n
    );

    // Create kernel instance
    Matrix_product_tiled_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_product_warp kernels
template<typename T>
py::array_t<T> matrix_product_warp_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_warp_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n
    );

    // Create kernel instance
    Matrix_product_warp_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_product_cublas kernels
template<typename T>
py::array_t<T> matrix_product_cublas_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_cublas_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n
    );

    // Create kernel instance
    Matrix_product_cublas_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_product_cutlass kernels
template<typename T>
py::array_t<T> matrix_product_cutlass_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_cutlass_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n
    );

    // Create kernel instance
    Matrix_product_cutlass_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_product_tensor kernels
template<typename T>
py::array_t<T> matrix_product_tensor_cuda_impl(
    const py::array_t<T>& a,
    const py::array_t<T>& b
) {
    // Validate input arrays
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::invalid_argument("Input arrays must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array a must be C-contiguous");
    }

    if (!(b.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array b must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long k = a_buf.shape[1];  // cols of A, rows of B
    long n = b_buf.shape[1];  // cols of B

    if (b_buf.shape[0] != k) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    auto result = py::array_t<T>({m, n});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    const T* b_ptr = static_cast<const T*>(b_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_product_tensor_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, k, n
    );

    // Create kernel instance
    Matrix_product_tensor_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_result = m * n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_b, size_b), "cudaMalloc for matrix B");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
        cuda_check_error(cudaMemcpy(d_b, b_ptr, size_b, cudaMemcpyHostToDevice), "cudaMemcpy B to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_b, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_b), "cudaFree B");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// High-level dispatch functions for matrix product operations
py::object matrix_product_naive_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_naive_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_naive_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_product_naive_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>(), b.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_product_naive_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>(), b.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_product_naive_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>(), b.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_product_naive_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>(), b.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_product_naive_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>(), b.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_product_naive_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>(), b.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_product_naive_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>(), b.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_product_naive_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>(), b.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_product_tiled_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_tiled_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_tiled_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_product_tiled_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>(), b.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_product_tiled_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>(), b.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_product_tiled_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>(), b.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_product_tiled_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>(), b.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_product_tiled_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>(), b.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_product_tiled_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>(), b.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_product_tiled_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>(), b.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_product_tiled_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>(), b.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for tiled matrix product: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_product_warp_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_warp_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_warp_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_product_warp_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>(), b.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_product_warp_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>(), b.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_product_warp_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>(), b.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_product_warp_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>(), b.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_product_warp_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>(), b.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_product_warp_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>(), b.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_product_warp_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>(), b.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_product_warp_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>(), b.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for warp matrix product: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_product_cublas_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_cublas_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_cublas_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for cuBLAS matrix product: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_product_cutlass_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_cutlass_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_cutlass_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_product_cutlass_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>(), b.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_product_cutlass_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>(), b.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_product_cutlass_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>(), b.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_product_cutlass_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>(), b.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_product_cutlass_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>(), b.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_product_cutlass_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>(), b.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_product_cutlass_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>(), b.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_product_cutlass_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>(), b.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for CUTLASS matrix product: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_product_tensor_dispatch(py::array a, py::array b) {
    if (!a.dtype().is(b.dtype())) {
        throw std::invalid_argument("Input arrays must have the same dtype");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_product_tensor_cuda_impl<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_product_tensor_cuda_impl<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for tensor matrix product: " + py::str(a.dtype()).cast<std::string>());
    }
}

// Python module definition
PYBIND11_MODULE(_matrix_product_ops_cuda, m) {
    m.doc() = "CUDA matrix product operations for py-gpu-algos";

    // Low-level type-specific functions for naive
    m.def("matrix_product_naive_float32", &matrix_product_naive_cuda_impl<float>,
          "Matrix multiplication (naive algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_float64", &matrix_product_naive_cuda_impl<double>,
          "Matrix multiplication (naive algorithm) for float64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_int8", &matrix_product_naive_cuda_impl<std::int8_t>,
          "Matrix multiplication (naive algorithm) for int8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_int16", &matrix_product_naive_cuda_impl<std::int16_t>,
          "Matrix multiplication (naive algorithm) for int16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_int32", &matrix_product_naive_cuda_impl<std::int32_t>,
          "Matrix multiplication (naive algorithm) for int32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_int64", &matrix_product_naive_cuda_impl<std::int64_t>,
          "Matrix multiplication (naive algorithm) for int64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_uint8", &matrix_product_naive_cuda_impl<std::uint8_t>,
          "Matrix multiplication (naive algorithm) for uint8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_uint16", &matrix_product_naive_cuda_impl<std::uint16_t>,
          "Matrix multiplication (naive algorithm) for uint16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_uint32", &matrix_product_naive_cuda_impl<std::uint32_t>,
          "Matrix multiplication (naive algorithm) for uint32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_naive_uint64", &matrix_product_naive_cuda_impl<std::uint64_t>,
          "Matrix multiplication (naive algorithm) for uint64",
          py::arg("a"), py::arg("b"));

    // Low-level type-specific functions for tiled
    m.def("matrix_product_tiled_float32", &matrix_product_tiled_cuda_impl<float>,
          "Matrix multiplication (tiled algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_float64", &matrix_product_tiled_cuda_impl<double>,
          "Matrix multiplication (tiled algorithm) for float64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_int8", &matrix_product_tiled_cuda_impl<std::int8_t>,
          "Matrix multiplication (tiled algorithm) for int8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_int16", &matrix_product_tiled_cuda_impl<std::int16_t>,
          "Matrix multiplication (tiled algorithm) for int16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_int32", &matrix_product_tiled_cuda_impl<std::int32_t>,
          "Matrix multiplication (tiled algorithm) for int32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_int64", &matrix_product_tiled_cuda_impl<std::int64_t>,
          "Matrix multiplication (tiled algorithm) for int64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_uint8", &matrix_product_tiled_cuda_impl<std::uint8_t>,
          "Matrix multiplication (tiled algorithm) for uint8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_uint16", &matrix_product_tiled_cuda_impl<std::uint16_t>,
          "Matrix multiplication (tiled algorithm) for uint16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_uint32", &matrix_product_tiled_cuda_impl<std::uint32_t>,
          "Matrix multiplication (tiled algorithm) for uint32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled_uint64", &matrix_product_tiled_cuda_impl<std::uint64_t>,
          "Matrix multiplication (tiled algorithm) for uint64",
          py::arg("a"), py::arg("b"));

    // Low-level type-specific functions for warp
    m.def("matrix_product_warp_float32", &matrix_product_warp_cuda_impl<float>,
          "Matrix multiplication (warp algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_float64", &matrix_product_warp_cuda_impl<double>,
          "Matrix multiplication (warp algorithm) for float64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_int8", &matrix_product_warp_cuda_impl<std::int8_t>,
          "Matrix multiplication (warp algorithm) for int8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_int16", &matrix_product_warp_cuda_impl<std::int16_t>,
          "Matrix multiplication (warp algorithm) for int16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_int32", &matrix_product_warp_cuda_impl<std::int32_t>,
          "Matrix multiplication (warp algorithm) for int32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_int64", &matrix_product_warp_cuda_impl<std::int64_t>,
          "Matrix multiplication (warp algorithm) for int64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_uint8", &matrix_product_warp_cuda_impl<std::uint8_t>,
          "Matrix multiplication (warp algorithm) for uint8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_uint16", &matrix_product_warp_cuda_impl<std::uint16_t>,
          "Matrix multiplication (warp algorithm) for uint16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_uint32", &matrix_product_warp_cuda_impl<std::uint32_t>,
          "Matrix multiplication (warp algorithm) for uint32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp_uint64", &matrix_product_warp_cuda_impl<std::uint64_t>,
          "Matrix multiplication (warp algorithm) for uint64",
          py::arg("a"), py::arg("b"));

    // Low-level type-specific functions for cublas
    m.def("matrix_product_cublas_float32", &matrix_product_cublas_cuda_impl<float>,
          "Matrix multiplication (cuBLAS algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cublas_float64", &matrix_product_cublas_cuda_impl<double>,
          "Matrix multiplication (cuBLAS algorithm) for float64",
          py::arg("a"), py::arg("b"));

    // Low-level type-specific functions for cutlass
    m.def("matrix_product_cutlass_float32", &matrix_product_cutlass_cuda_impl<float>,
          "Matrix multiplication (CUTLASS algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_float64", &matrix_product_cutlass_cuda_impl<double>,
          "Matrix multiplication (CUTLASS algorithm) for float64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_int8", &matrix_product_cutlass_cuda_impl<std::int8_t>,
          "Matrix multiplication (CUTLASS algorithm) for int8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_int16", &matrix_product_cutlass_cuda_impl<std::int16_t>,
          "Matrix multiplication (CUTLASS algorithm) for int16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_int32", &matrix_product_cutlass_cuda_impl<std::int32_t>,
          "Matrix multiplication (CUTLASS algorithm) for int32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_int64", &matrix_product_cutlass_cuda_impl<std::int64_t>,
          "Matrix multiplication (CUTLASS algorithm) for int64",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_uint8", &matrix_product_cutlass_cuda_impl<std::uint8_t>,
          "Matrix multiplication (CUTLASS algorithm) for uint8",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_uint16", &matrix_product_cutlass_cuda_impl<std::uint16_t>,
          "Matrix multiplication (CUTLASS algorithm) for uint16",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_uint32", &matrix_product_cutlass_cuda_impl<std::uint32_t>,
          "Matrix multiplication (CUTLASS algorithm) for uint32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass_uint64", &matrix_product_cutlass_cuda_impl<std::uint64_t>,
          "Matrix multiplication (CUTLASS algorithm) for uint64",
          py::arg("a"), py::arg("b"));

    // Low-level type-specific functions for tensor
    m.def("matrix_product_tensor_float32", &matrix_product_tensor_cuda_impl<float>,
          "Matrix multiplication (tensor algorithm) for float32",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tensor_float64", &matrix_product_tensor_cuda_impl<double>,
          "Matrix multiplication (tensor algorithm) for float64",
          py::arg("a"), py::arg("b"));

    // High-level dispatch functions
    m.def("matrix_product_naive", &matrix_product_naive_dispatch,
          "Matrix multiplication (naive algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tiled", &matrix_product_tiled_dispatch,
          "Matrix multiplication (tiled algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_warp", &matrix_product_warp_dispatch,
          "Matrix multiplication (warp algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cublas", &matrix_product_cublas_dispatch,
          "Matrix multiplication (cuBLAS algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_cutlass", &matrix_product_cutlass_dispatch,
          "Matrix multiplication (CUTLASS algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
    m.def("matrix_product_tensor", &matrix_product_tensor_dispatch,
          "Matrix multiplication (tensor algorithm) with automatic type dispatch",
          py::arg("a"), py::arg("b"));
}
