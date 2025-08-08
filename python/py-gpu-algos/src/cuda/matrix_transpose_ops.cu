/*
 * CUDA Matrix Transpose Operations Bindings for py-gpu-algos
 *
 * This module provides Python bindings for matrix transpose operations using pybind11.
 * It instantiates matrix transpose kernels for all supported numeric types.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

// Include CUDA kernel headers
#include "cuda/kernels/matrix_transpose/matrix_transpose_naive.cuh"
#include "cuda/kernels/matrix_transpose/matrix_transpose_striped.cuh"
#include "cuda/kernels/matrix_transpose/matrix_transpose_tiled.cuh"
#include "cuda/kernels/matrix_transpose/matrix_transpose_cublas.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/check_errors.cuh"

namespace py = pybind11;

// Helper function to launch matrix_transpose_naive kernels
template<typename T>
py::array_t<T> matrix_transpose_naive_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 2) {
        throw std::invalid_argument("Input array must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long n = a_buf.shape[1];  // cols of A

    // Create output array (transposed dimensions)
    auto result = py::array_t<T>({n, m});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_transpose_naive_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, n
    );

    // Create kernel instance
    Matrix_transpose_naive_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * n * sizeof(T);
    size_t size_result = n * m * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_transpose_striped kernels
template<typename T>
py::array_t<T> matrix_transpose_striped_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 2) {
        throw std::invalid_argument("Input array must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long n = a_buf.shape[1];  // cols of A

    // Create output array (transposed dimensions)
    auto result = py::array_t<T>({n, m});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_transpose_striped_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, n,
        32        // Default block dimension
    );

    // Create kernel instance
    Matrix_transpose_striped_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * n * sizeof(T);
    size_t size_result = n * m * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_transpose_tiled kernels
template<typename T>
py::array_t<T> matrix_transpose_tiled_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 2) {
        throw std::invalid_argument("Input array must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long n = a_buf.shape[1];  // cols of A

    // Create output array (transposed dimensions)
    auto result = py::array_t<T>({n, m});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_transpose_tiled_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, n
    );

    // Create kernel instance
    Matrix_transpose_tiled_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * n * sizeof(T);
    size_t size_result = n * m * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// Helper function to launch matrix_transpose_cublas kernels
template<typename T>
py::array_t<T> matrix_transpose_cublas_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 2) {
        throw std::invalid_argument("Input array must be 2-dimensional");
    }

    if (!(a.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get matrix dimensions
    long m = a_buf.shape[0];  // rows of A
    long n = a_buf.shape[1];  // cols of A

    // Create output array (transposed dimensions)
    auto result = py::array_t<T>({n, m});
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Matrix_transpose_cublas_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        m, n
    );

    // Create kernel instance
    Matrix_transpose_cublas_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = m * n * sizeof(T);
    size_t size_result = n * m * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for matrix A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result matrix");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_a, a_ptr, size_a, cudaMemcpyHostToDevice), "cudaMemcpy A to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel
        kernel.run_device_kernel(d_a, d_result, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host
        cuda_check_error(cudaMemcpy(result_ptr, d_result, size_result, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_a) cudaFree(d_a);
        if (d_result) cudaFree(d_result);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_a), "cudaFree A");
    cuda_check_error(cudaFree(d_result), "cudaFree result");

    return result;
}

// High-level dispatch functions for matrix transpose operations
py::object matrix_transpose_naive_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_transpose_naive_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_transpose_naive_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_transpose_naive_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_transpose_naive_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_transpose_naive_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_transpose_naive_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_transpose_naive_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_transpose_naive_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_transpose_naive_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_transpose_naive_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_transpose_striped_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_transpose_striped_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_transpose_striped_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_transpose_striped_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_transpose_striped_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_transpose_striped_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_transpose_striped_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_transpose_striped_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_transpose_striped_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_transpose_striped_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_transpose_striped_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_transpose_tiled_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_transpose_tiled_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_transpose_tiled_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return matrix_transpose_tiled_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object matrix_transpose_cublas_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return matrix_transpose_cublas_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return matrix_transpose_cublas_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else {
        throw std::invalid_argument("Unsupported dtype for cuBLAS matrix transpose: " + py::str(a.dtype()).cast<std::string>());
    }
}

// Python module definition
PYBIND11_MODULE(_matrix_transpose_ops_cuda, m) {
    m.doc() = "CUDA matrix transpose operations for py-gpu-algos";

    // Low-level type-specific functions for naive
    m.def("matrix_transpose_naive_float32", &matrix_transpose_naive_cuda_impl<float>,
          "Matrix transpose (naive algorithm) for float32", py::arg("a"));
    m.def("matrix_transpose_naive_float64", &matrix_transpose_naive_cuda_impl<double>,
          "Matrix transpose (naive algorithm) for float64", py::arg("a"));
    m.def("matrix_transpose_naive_int8", &matrix_transpose_naive_cuda_impl<std::int8_t>,
          "Matrix transpose (naive algorithm) for int8", py::arg("a"));
    m.def("matrix_transpose_naive_int16", &matrix_transpose_naive_cuda_impl<std::int16_t>,
          "Matrix transpose (naive algorithm) for int16", py::arg("a"));
    m.def("matrix_transpose_naive_int32", &matrix_transpose_naive_cuda_impl<std::int32_t>,
          "Matrix transpose (naive algorithm) for int32", py::arg("a"));
    m.def("matrix_transpose_naive_int64", &matrix_transpose_naive_cuda_impl<std::int64_t>,
          "Matrix transpose (naive algorithm) for int64", py::arg("a"));
    m.def("matrix_transpose_naive_uint8", &matrix_transpose_naive_cuda_impl<std::uint8_t>,
          "Matrix transpose (naive algorithm) for uint8", py::arg("a"));
    m.def("matrix_transpose_naive_uint16", &matrix_transpose_naive_cuda_impl<std::uint16_t>,
          "Matrix transpose (naive algorithm) for uint16", py::arg("a"));
    m.def("matrix_transpose_naive_uint32", &matrix_transpose_naive_cuda_impl<std::uint32_t>,
          "Matrix transpose (naive algorithm) for uint32", py::arg("a"));
    m.def("matrix_transpose_naive_uint64", &matrix_transpose_naive_cuda_impl<std::uint64_t>,
          "Matrix transpose (naive algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for striped
    m.def("matrix_transpose_striped_float32", &matrix_transpose_striped_cuda_impl<float>,
          "Matrix transpose (striped algorithm) for float32", py::arg("a"));
    m.def("matrix_transpose_striped_float64", &matrix_transpose_striped_cuda_impl<double>,
          "Matrix transpose (striped algorithm) for float64", py::arg("a"));
    m.def("matrix_transpose_striped_int8", &matrix_transpose_striped_cuda_impl<std::int8_t>,
          "Matrix transpose (striped algorithm) for int8", py::arg("a"));
    m.def("matrix_transpose_striped_int16", &matrix_transpose_striped_cuda_impl<std::int16_t>,
          "Matrix transpose (striped algorithm) for int16", py::arg("a"));
    m.def("matrix_transpose_striped_int32", &matrix_transpose_striped_cuda_impl<std::int32_t>,
          "Matrix transpose (striped algorithm) for int32", py::arg("a"));
    m.def("matrix_transpose_striped_int64", &matrix_transpose_striped_cuda_impl<std::int64_t>,
          "Matrix transpose (striped algorithm) for int64", py::arg("a"));
    m.def("matrix_transpose_striped_uint8", &matrix_transpose_striped_cuda_impl<std::uint8_t>,
          "Matrix transpose (striped algorithm) for uint8", py::arg("a"));
    m.def("matrix_transpose_striped_uint16", &matrix_transpose_striped_cuda_impl<std::uint16_t>,
          "Matrix transpose (striped algorithm) for uint16", py::arg("a"));
    m.def("matrix_transpose_striped_uint32", &matrix_transpose_striped_cuda_impl<std::uint32_t>,
          "Matrix transpose (striped algorithm) for uint32", py::arg("a"));
    m.def("matrix_transpose_striped_uint64", &matrix_transpose_striped_cuda_impl<std::uint64_t>,
          "Matrix transpose (striped algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for tiled
    m.def("matrix_transpose_tiled_float32", &matrix_transpose_tiled_cuda_impl<float>,
          "Matrix transpose (tiled algorithm) for float32", py::arg("a"));
    m.def("matrix_transpose_tiled_float64", &matrix_transpose_tiled_cuda_impl<double>,
          "Matrix transpose (tiled algorithm) for float64", py::arg("a"));
    m.def("matrix_transpose_tiled_int8", &matrix_transpose_tiled_cuda_impl<std::int8_t>,
          "Matrix transpose (tiled algorithm) for int8", py::arg("a"));
    m.def("matrix_transpose_tiled_int16", &matrix_transpose_tiled_cuda_impl<std::int16_t>,
          "Matrix transpose (tiled algorithm) for int16", py::arg("a"));
    m.def("matrix_transpose_tiled_int32", &matrix_transpose_tiled_cuda_impl<std::int32_t>,
          "Matrix transpose (tiled algorithm) for int32", py::arg("a"));
    m.def("matrix_transpose_tiled_int64", &matrix_transpose_tiled_cuda_impl<std::int64_t>,
          "Matrix transpose (tiled algorithm) for int64", py::arg("a"));
    m.def("matrix_transpose_tiled_uint8", &matrix_transpose_tiled_cuda_impl<std::uint8_t>,
          "Matrix transpose (tiled algorithm) for uint8", py::arg("a"));
    m.def("matrix_transpose_tiled_uint16", &matrix_transpose_tiled_cuda_impl<std::uint16_t>,
          "Matrix transpose (tiled algorithm) for uint16", py::arg("a"));
    m.def("matrix_transpose_tiled_uint32", &matrix_transpose_tiled_cuda_impl<std::uint32_t>,
          "Matrix transpose (tiled algorithm) for uint32", py::arg("a"));
    m.def("matrix_transpose_tiled_uint64", &matrix_transpose_tiled_cuda_impl<std::uint64_t>,
          "Matrix transpose (tiled algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for cublas
    m.def("matrix_transpose_cublas_float32", &matrix_transpose_cublas_cuda_impl<float>,
          "Matrix transpose (cuBLAS algorithm) for float32", py::arg("a"));
    m.def("matrix_transpose_cublas_float64", &matrix_transpose_cublas_cuda_impl<double>,
          "Matrix transpose (cuBLAS algorithm) for float64", py::arg("a"));

    // High-level dispatch functions
    m.def("matrix_transpose_naive", &matrix_transpose_naive_dispatch,
          "Matrix transpose (naive algorithm) with automatic type dispatch", py::arg("a"));
    m.def("matrix_transpose_striped", &matrix_transpose_striped_dispatch,
          "Matrix transpose (striped algorithm) with automatic type dispatch", py::arg("a"));
    m.def("matrix_transpose_tiled", &matrix_transpose_tiled_dispatch,
          "Matrix transpose (tiled algorithm) with automatic type dispatch", py::arg("a"));
    m.def("matrix_transpose_cublas", &matrix_transpose_cublas_dispatch,
          "Matrix transpose (cuBLAS algorithm) with automatic type dispatch", py::arg("a"));
}
