/*
 * CUDA Vector Operations Bindings for py-gpu-algos
 *
 * This module provides Python bindings for vector operations using pybind11.
 * It instantiates vector kernels for all supported numeric types.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

// Include CUDA kernel headers
#include "cuda/kernels/vector_cumsum/vector_cumsum_serial.cuh"
#include "cuda/kernels/vector_cumsum/vector_cumsum_parallel.cuh"
#include "cuda/kernels/vector_cummax/vector_cummax_parallel.cuh"
#include "cuda/kernels/vector_scan_generic/vector_scan_parallel.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/check_errors.cuh"

namespace py = pybind11;

// Helper function to launch vector_cumsum_serial kernels
template<typename T>
py::array_t<T> vector_cumsum_serial_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 1) {
        throw std::invalid_argument("Input array must be 1-dimensional");
    }

    if (!a.flags() & py::array::c_style) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get vector size
    long n = a_buf.shape[0];

    // Create output array
    auto result = py::array_t<T>(n);
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Vector_cumsum_serial_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        n         // n
    );

    // Create kernel instance
    Vector_cumsum_serial_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = n * sizeof(T);
    size_t size_result = n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for vector A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result vector");

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

// Helper function to launch vector_cumsum_parallel kernels
template<typename T>
py::array_t<T> vector_cumsum_parallel_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 1) {
        throw std::invalid_argument("Input array must be 1-dimensional");
    }

    if (!a.flags() & py::array::c_style) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get vector size
    long n = a_buf.shape[0];

    // Create output array
    auto result = py::array_t<T>(n);
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Vector_cumsum_parallel_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        n, 1024   // n, block_dim
    );

    // Create kernel instance
    Vector_cumsum_parallel_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = n * sizeof(T);
    size_t size_result = n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for vector A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result vector");

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

// Helper function to launch vector_cummax_parallel kernels
template<typename T>
py::array_t<T> vector_cummax_parallel_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 1) {
        throw std::invalid_argument("Input array must be 1-dimensional");
    }

    if (!a.flags() & py::array::c_style) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get vector size
    long n = a_buf.shape[0];

    // Create output array
    auto result = py::array_t<T>(n);
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);

    // Create kernel specification
    Vector_cummax_parallel_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        n, 1024   // n, block_dim
    );

    // Create kernel instance
    Vector_cummax_parallel_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = n * sizeof(T);
    size_t size_result = n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for vector A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result vector");

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

// Helper function to launch vector_scan_parallel kernels
template<typename T, typename Op>
py::array_t<T> vector_scan_parallel_cuda_impl(const py::array_t<T>& a) {
    // Validate input array
    auto a_buf = a.request();

    if (a_buf.ndim != 1) {
        throw std::invalid_argument("Input array must be 1-dimensional");
    }

    if (!a.flags() & py::array::c_style) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get vector size
    long n = a_buf.shape[0];

    // Create output array
    auto result = py::array_t<T>(n);
    auto result_buf = result.request();

    // Get data pointers
    const T* a_ptr = static_cast<const T*>(a_buf.ptr);
    T* result_ptr = static_cast<T*>(result_buf.ptr);



    // Create kernel specification - operation string will be determined by Op type
    std::string operation;
    if constexpr (std::is_same_v<Op, cuda_max_op<T>>) {
        operation = "max";
    } else if constexpr (std::is_same_v<Op, cuda_min_op<T>>) {
        operation = "min";
    } else if constexpr (std::is_same_v<Op, cuda_sum_op<T>>) {
        operation = "sum";
    } else if constexpr (std::is_same_v<Op, cuda_prod_op<T>>) {
        operation = "prod";
    } else {
        operation = "unknown";
    }

    Vector_scan_parallel_spec spec(
        "float",   // Type string (will be overridden by template parameter)
        operation, // Operation
        n, 1024    // n, block_dim
    );

    // Create kernel instance
    Vector_scan_parallel_kernel<T, Op> kernel(spec);

    // Allocate GPU memory
    T* d_a = nullptr;
    T* d_result = nullptr;

    size_t size_a = n * sizeof(T);
    size_t size_result = n * sizeof(T);

    cuda_check_error(cudaMalloc(&d_a, size_a), "cudaMalloc for vector A");
    cuda_check_error(cudaMalloc(&d_result, size_result), "cudaMalloc for result vector");

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

// High-level dispatch functions
py::object vector_cumsum_serial_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return vector_cumsum_serial_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return vector_cumsum_serial_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return vector_cumsum_serial_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return vector_cumsum_serial_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return vector_cumsum_serial_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return vector_cumsum_serial_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return vector_cumsum_serial_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return vector_cumsum_serial_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return vector_cumsum_serial_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return vector_cumsum_serial_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object vector_cumsum_parallel_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return vector_cumsum_parallel_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return vector_cumsum_parallel_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return vector_cumsum_parallel_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object vector_cummax_parallel_dispatch(py::array a) {
    if (a.dtype().is(py::dtype::of<float>())) {
        return vector_cummax_parallel_cuda_impl<float>(a.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return vector_cummax_parallel_cuda_impl<double>(a.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<std::int8_t>())) {
        return vector_cummax_parallel_cuda_impl<std::int8_t>(a.cast<py::array_t<std::int8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int16_t>())) {
        return vector_cummax_parallel_cuda_impl<std::int16_t>(a.cast<py::array_t<std::int16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
        return vector_cummax_parallel_cuda_impl<std::int32_t>(a.cast<py::array_t<std::int32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
        return vector_cummax_parallel_cuda_impl<std::int64_t>(a.cast<py::array_t<std::int64_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint8_t>())) {
        return vector_cummax_parallel_cuda_impl<std::uint8_t>(a.cast<py::array_t<std::uint8_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint16_t>())) {
        return vector_cummax_parallel_cuda_impl<std::uint16_t>(a.cast<py::array_t<std::uint16_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint32_t>())) {
        return vector_cummax_parallel_cuda_impl<std::uint32_t>(a.cast<py::array_t<std::uint32_t>>());
    } else if (a.dtype().is(py::dtype::of<std::uint64_t>())) {
        return vector_cummax_parallel_cuda_impl<std::uint64_t>(a.cast<py::array_t<std::uint64_t>>());
    } else {
        throw std::invalid_argument("Unsupported dtype: " + py::str(a.dtype()).cast<std::string>());
    }
}

py::object vector_scan_parallel_dispatch(py::array a, const std::string& operation) {
    if (operation == "max") {
        if (a.dtype().is(py::dtype::of<float>())) {
            return vector_scan_parallel_cuda_impl<float, cuda_max_op<float>>(a.cast<py::array_t<float>>());
        } else if (a.dtype().is(py::dtype::of<double>())) {
            return vector_scan_parallel_cuda_impl<double, cuda_max_op<double>>(a.cast<py::array_t<double>>());
        } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
            return vector_scan_parallel_cuda_impl<std::int32_t, cuda_max_op<std::int32_t>>(a.cast<py::array_t<std::int32_t>>());
        } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
            return vector_scan_parallel_cuda_impl<std::int64_t, cuda_max_op<std::int64_t>>(a.cast<py::array_t<std::int64_t>>());
        } else {
            throw std::invalid_argument("Unsupported dtype for max operation: " + py::str(a.dtype()).cast<std::string>());
        }
    } else if (operation == "min") {
        if (a.dtype().is(py::dtype::of<float>())) {
            return vector_scan_parallel_cuda_impl<float, cuda_min_op<float>>(a.cast<py::array_t<float>>());
        } else if (a.dtype().is(py::dtype::of<double>())) {
            return vector_scan_parallel_cuda_impl<double, cuda_min_op<double>>(a.cast<py::array_t<double>>());
        } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
            return vector_scan_parallel_cuda_impl<std::int32_t, cuda_min_op<std::int32_t>>(a.cast<py::array_t<std::int32_t>>());
        } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
            return vector_scan_parallel_cuda_impl<std::int64_t, cuda_min_op<std::int64_t>>(a.cast<py::array_t<std::int64_t>>());
        } else {
            throw std::invalid_argument("Unsupported dtype for min operation: " + py::str(a.dtype()).cast<std::string>());
        }
    } else if (operation == "sum") {
        if (a.dtype().is(py::dtype::of<float>())) {
            return vector_scan_parallel_cuda_impl<float, cuda_sum_op<float>>(a.cast<py::array_t<float>>());
        } else if (a.dtype().is(py::dtype::of<double>())) {
            return vector_scan_parallel_cuda_impl<double, cuda_sum_op<double>>(a.cast<py::array_t<double>>());
        } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
            return vector_scan_parallel_cuda_impl<std::int32_t, cuda_sum_op<std::int32_t>>(a.cast<py::array_t<std::int32_t>>());
        } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
            return vector_scan_parallel_cuda_impl<std::int64_t, cuda_sum_op<std::int64_t>>(a.cast<py::array_t<std::int64_t>>());
        } else {
            throw std::invalid_argument("Unsupported dtype for sum operation: " + py::str(a.dtype()).cast<std::string>());
        }
    } else if (operation == "prod") {
        if (a.dtype().is(py::dtype::of<float>())) {
            return vector_scan_parallel_cuda_impl<float, cuda_prod_op<float>>(a.cast<py::array_t<float>>());
        } else if (a.dtype().is(py::dtype::of<double>())) {
            return vector_scan_parallel_cuda_impl<double, cuda_prod_op<double>>(a.cast<py::array_t<double>>());
        } else if (a.dtype().is(py::dtype::of<std::int32_t>())) {
            return vector_scan_parallel_cuda_impl<std::int32_t, cuda_prod_op<std::int32_t>>(a.cast<py::array_t<std::int32_t>>());
        } else if (a.dtype().is(py::dtype::of<std::int64_t>())) {
            return vector_scan_parallel_cuda_impl<std::int64_t, cuda_prod_op<std::int64_t>>(a.cast<py::array_t<std::int64_t>>());
        } else {
            throw std::invalid_argument("Unsupported dtype for prod operation: " + py::str(a.dtype()).cast<std::string>());
        }
    } else {
        throw std::invalid_argument("Unsupported operation: " + operation + ". Must be one of: max, min, sum, prod");
    }
}

// Python module definition
PYBIND11_MODULE(_vector_ops_cuda, m) {
    m.doc() = "CUDA vector operations for py-gpu-algos";

    // Low-level type-specific functions for cumsum_serial
    m.def("vector_cumsum_serial_float32", &vector_cumsum_serial_cuda_impl<float>,
          "Cumulative sum (serial algorithm) for float32", py::arg("a"));
    m.def("vector_cumsum_serial_float64", &vector_cumsum_serial_cuda_impl<double>,
          "Cumulative sum (serial algorithm) for float64", py::arg("a"));
    m.def("vector_cumsum_serial_int8", &vector_cumsum_serial_cuda_impl<std::int8_t>,
          "Cumulative sum (serial algorithm) for int8", py::arg("a"));
    m.def("vector_cumsum_serial_int16", &vector_cumsum_serial_cuda_impl<std::int16_t>,
          "Cumulative sum (serial algorithm) for int16", py::arg("a"));
    m.def("vector_cumsum_serial_int32", &vector_cumsum_serial_cuda_impl<std::int32_t>,
          "Cumulative sum (serial algorithm) for int32", py::arg("a"));
    m.def("vector_cumsum_serial_int64", &vector_cumsum_serial_cuda_impl<std::int64_t>,
          "Cumulative sum (serial algorithm) for int64", py::arg("a"));
    m.def("vector_cumsum_serial_uint8", &vector_cumsum_serial_cuda_impl<std::uint8_t>,
          "Cumulative sum (serial algorithm) for uint8", py::arg("a"));
    m.def("vector_cumsum_serial_uint16", &vector_cumsum_serial_cuda_impl<std::uint16_t>,
          "Cumulative sum (serial algorithm) for uint16", py::arg("a"));
    m.def("vector_cumsum_serial_uint32", &vector_cumsum_serial_cuda_impl<std::uint32_t>,
          "Cumulative sum (serial algorithm) for uint32", py::arg("a"));
    m.def("vector_cumsum_serial_uint64", &vector_cumsum_serial_cuda_impl<std::uint64_t>,
          "Cumulative sum (serial algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for cumsum_parallel
    m.def("vector_cumsum_parallel_float32", &vector_cumsum_parallel_cuda_impl<float>,
          "Cumulative sum (parallel algorithm) for float32", py::arg("a"));
    m.def("vector_cumsum_parallel_float64", &vector_cumsum_parallel_cuda_impl<double>,
          "Cumulative sum (parallel algorithm) for float64", py::arg("a"));
    m.def("vector_cumsum_parallel_int8", &vector_cumsum_parallel_cuda_impl<std::int8_t>,
          "Cumulative sum (parallel algorithm) for int8", py::arg("a"));
    m.def("vector_cumsum_parallel_int16", &vector_cumsum_parallel_cuda_impl<std::int16_t>,
          "Cumulative sum (parallel algorithm) for int16", py::arg("a"));
    m.def("vector_cumsum_parallel_int32", &vector_cumsum_parallel_cuda_impl<std::int32_t>,
          "Cumulative sum (parallel algorithm) for int32", py::arg("a"));
    m.def("vector_cumsum_parallel_int64", &vector_cumsum_parallel_cuda_impl<std::int64_t>,
          "Cumulative sum (parallel algorithm) for int64", py::arg("a"));
    m.def("vector_cumsum_parallel_uint8", &vector_cumsum_parallel_cuda_impl<std::uint8_t>,
          "Cumulative sum (parallel algorithm) for uint8", py::arg("a"));
    m.def("vector_cumsum_parallel_uint16", &vector_cumsum_parallel_cuda_impl<std::uint16_t>,
          "Cumulative sum (parallel algorithm) for uint16", py::arg("a"));
    m.def("vector_cumsum_parallel_uint32", &vector_cumsum_parallel_cuda_impl<std::uint32_t>,
          "Cumulative sum (parallel algorithm) for uint32", py::arg("a"));
    m.def("vector_cumsum_parallel_uint64", &vector_cumsum_parallel_cuda_impl<std::uint64_t>,
          "Cumulative sum (parallel algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for cummax_parallel
    m.def("vector_cummax_parallel_float32", &vector_cummax_parallel_cuda_impl<float>,
          "Cumulative maximum (parallel algorithm) for float32", py::arg("a"));
    m.def("vector_cummax_parallel_float64", &vector_cummax_parallel_cuda_impl<double>,
          "Cumulative maximum (parallel algorithm) for float64", py::arg("a"));
    m.def("vector_cummax_parallel_int8", &vector_cummax_parallel_cuda_impl<std::int8_t>,
          "Cumulative maximum (parallel algorithm) for int8", py::arg("a"));
    m.def("vector_cummax_parallel_int16", &vector_cummax_parallel_cuda_impl<std::int16_t>,
          "Cumulative maximum (parallel algorithm) for int16", py::arg("a"));
    m.def("vector_cummax_parallel_int32", &vector_cummax_parallel_cuda_impl<std::int32_t>,
          "Cumulative maximum (parallel algorithm) for int32", py::arg("a"));
    m.def("vector_cummax_parallel_int64", &vector_cummax_parallel_cuda_impl<std::int64_t>,
          "Cumulative maximum (parallel algorithm) for int64", py::arg("a"));
    m.def("vector_cummax_parallel_uint8", &vector_cummax_parallel_cuda_impl<std::uint8_t>,
          "Cumulative maximum (parallel algorithm) for uint8", py::arg("a"));
    m.def("vector_cummax_parallel_uint16", &vector_cummax_parallel_cuda_impl<std::uint16_t>,
          "Cumulative maximum (parallel algorithm) for uint16", py::arg("a"));
    m.def("vector_cummax_parallel_uint32", &vector_cummax_parallel_cuda_impl<std::uint32_t>,
          "Cumulative maximum (parallel algorithm) for uint32", py::arg("a"));
    m.def("vector_cummax_parallel_uint64", &vector_cummax_parallel_cuda_impl<std::uint64_t>,
          "Cumulative maximum (parallel algorithm) for uint64", py::arg("a"));

    // Low-level type-specific functions for scan_parallel with specific operations
    // Max operations
    m.def("vector_scan_parallel_max_float32", &vector_scan_parallel_cuda_impl<float, cuda_max_op<float>>,
          "Parallel scan with max operation for float32", py::arg("a"));
    m.def("vector_scan_parallel_max_float64", &vector_scan_parallel_cuda_impl<double, cuda_max_op<double>>,
          "Parallel scan with max operation for float64", py::arg("a"));
    m.def("vector_scan_parallel_max_int32", &vector_scan_parallel_cuda_impl<std::int32_t, cuda_max_op<std::int32_t>>,
          "Parallel scan with max operation for int32", py::arg("a"));
    m.def("vector_scan_parallel_max_int64", &vector_scan_parallel_cuda_impl<std::int64_t, cuda_max_op<std::int64_t>>,
          "Parallel scan with max operation for int64", py::arg("a"));

    // Min operations
    m.def("vector_scan_parallel_min_float32", &vector_scan_parallel_cuda_impl<float, cuda_min_op<float>>,
          "Parallel scan with min operation for float32", py::arg("a"));
    m.def("vector_scan_parallel_min_float64", &vector_scan_parallel_cuda_impl<double, cuda_min_op<double>>,
          "Parallel scan with min operation for float64", py::arg("a"));
    m.def("vector_scan_parallel_min_int32", &vector_scan_parallel_cuda_impl<std::int32_t, cuda_min_op<std::int32_t>>,
          "Parallel scan with min operation for int32", py::arg("a"));
    m.def("vector_scan_parallel_min_int64", &vector_scan_parallel_cuda_impl<std::int64_t, cuda_min_op<std::int64_t>>,
          "Parallel scan with min operation for int64", py::arg("a"));

    // Sum operations
    m.def("vector_scan_parallel_sum_float32", &vector_scan_parallel_cuda_impl<float, cuda_sum_op<float>>,
          "Parallel scan with sum operation for float32", py::arg("a"));
    m.def("vector_scan_parallel_sum_float64", &vector_scan_parallel_cuda_impl<double, cuda_sum_op<double>>,
          "Parallel scan with sum operation for float64", py::arg("a"));
    m.def("vector_scan_parallel_sum_int32", &vector_scan_parallel_cuda_impl<std::int32_t, cuda_sum_op<std::int32_t>>,
          "Parallel scan with sum operation for int32", py::arg("a"));
    m.def("vector_scan_parallel_sum_int64", &vector_scan_parallel_cuda_impl<std::int64_t, cuda_sum_op<std::int64_t>>,
          "Parallel scan with sum operation for int64", py::arg("a"));

    // Prod operations
    m.def("vector_scan_parallel_prod_float32", &vector_scan_parallel_cuda_impl<float, cuda_prod_op<float>>,
          "Parallel scan with prod operation for float32", py::arg("a"));
    m.def("vector_scan_parallel_prod_float64", &vector_scan_parallel_cuda_impl<double, cuda_prod_op<double>>,
          "Parallel scan with prod operation for float64", py::arg("a"));
    m.def("vector_scan_parallel_prod_int32", &vector_scan_parallel_cuda_impl<std::int32_t, cuda_prod_op<std::int32_t>>,
          "Parallel scan with prod operation for int32", py::arg("a"));
    m.def("vector_scan_parallel_prod_int64", &vector_scan_parallel_cuda_impl<std::int64_t, cuda_prod_op<std::int64_t>>,
          "Parallel scan with prod operation for int64", py::arg("a"));

    // High-level dispatch functions
    m.def("vector_cumsum_serial", &vector_cumsum_serial_dispatch,
          "Cumulative sum (serial algorithm) with automatic type dispatch", py::arg("a"));
    m.def("vector_cumsum_parallel", &vector_cumsum_parallel_dispatch,
          "Cumulative sum (parallel algorithm) with automatic type dispatch", py::arg("a"));
    m.def("vector_cummax_parallel", &vector_cummax_parallel_dispatch,
          "Cumulative maximum (parallel algorithm) with automatic type dispatch", py::arg("a"));
    m.def("vector_scan_parallel", &vector_scan_parallel_dispatch,
          "Parallel scan with automatic type dispatch", py::arg("a"), py::arg("operation"));
}
