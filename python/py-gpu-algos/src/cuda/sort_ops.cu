/*
 * CUDA Sort Operations Bindings for py-gpu-algos
 *
 * This module provides Python bindings for sort operations using pybind11.
 * It instantiates sort kernels for 3D tensor operations with all supported numeric types.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

// Include CUDA kernel headers
#include "cuda/kernels/sort/tensor_sort_bitonic.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/check_errors.cuh"
#include "common/types/tensor3d.hpp"

namespace py = pybind11;

// Helper function to launch tensor_sort_bitonic kernels (in-place operation)
template<typename T>
void tensor_sort_bitonic_cuda_impl(
    py::array_t<T>& tensor,      // Modified in-place
    const std::string& sort_dim  // "cols", "rows", or "sheets"
) {
    // Validate input array
    auto tensor_buf = tensor.request();

    if (tensor_buf.ndim != 3) {
        throw std::invalid_argument("Input array must be 3-dimensional");
    }

    if (!(tensor.flags() & py::array::c_style)) {
        throw std::invalid_argument("Array must be C-contiguous");
    }

    // Get tensor dimensions - numpy C-contiguous arrays have shape (sheet, row, col)
    // but Tensor3D expects (col, row, sheet) order
    const long sheets = tensor_buf.shape[0];  // First dimension is sheets
    const long rows = tensor_buf.shape[1];    // Second dimension is rows
    const long cols = tensor_buf.shape[2];    // Third dimension is cols

    // Validate sort dimension size is power of 2
    const long target_dim_size = sort_dim == "cols" ? cols : sort_dim == "rows" ? rows : sheets;

    // Get data pointer
    T* const tensor_ptr = static_cast<T*>(tensor_buf.ptr);

    // Create kernel specification - Tensor3d_sort_bitonic_spec expects (cols, rows, sheets) order
    Tensor3d_sort_bitonic_spec spec(
        "float",  // Type string (will be overridden by template parameter)
        sort_dim,  // Sort dimension
        cols, rows, sheets  // Tensor dimensions in col, row, sheet order
    );

    // Create kernel instance
    Tensor3d_sort_bitonic_kernel<T> kernel(spec);

    // Allocate GPU memory
    T* d_tensor = nullptr;

    const size_t size_tensor = cols * rows * sheets * sizeof(T);

    cuda_check_error(cudaMalloc(&d_tensor, size_tensor), "cudaMalloc for tensor");

    try {
        // Copy data to device
        cuda_check_error(cudaMemcpy(d_tensor, tensor_ptr, size_tensor, cudaMemcpyHostToDevice), "cudaMemcpy tensor to device");

        // Create CUDA stream
        cudaStream_t stream;
        cuda_check_error(cudaStreamCreate(&stream), "cudaStreamCreate");

        // Run kernel (in-place operation)
        kernel.run_device_kernel(d_tensor, nullptr, stream);

        // Wait for completion
        cuda_check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy result back to host (overwrites original data)
        cuda_check_error(cudaMemcpy(tensor_ptr, d_tensor, size_tensor, cudaMemcpyDeviceToHost), "cudaMemcpy result to host");

        // Cleanup stream
        cuda_check_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    } catch (...) {
        // Cleanup on error
        if (d_tensor) cudaFree(d_tensor);
        throw;
    }

    // Cleanup GPU memory
    cuda_check_error(cudaFree(d_tensor), "cudaFree tensor");
}

// High-level dispatch function
void tensor_sort_bitonic_dispatch(py::array tensor, const std::string& sort_dim) {
    if (tensor.dtype().is(py::dtype::of<float>())) {
        auto arr = tensor.cast<py::array_t<float>>();
        tensor_sort_bitonic_cuda_impl<float>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<double>())) {
        auto arr = tensor.cast<py::array_t<double>>();
        tensor_sort_bitonic_cuda_impl<double>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::int8_t>())) {
        auto arr = tensor.cast<py::array_t<std::int8_t>>();
        tensor_sort_bitonic_cuda_impl<std::int8_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::int16_t>())) {
        auto arr = tensor.cast<py::array_t<std::int16_t>>();
        tensor_sort_bitonic_cuda_impl<std::int16_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::int32_t>())) {
        auto arr = tensor.cast<py::array_t<std::int32_t>>();
        tensor_sort_bitonic_cuda_impl<std::int32_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::int64_t>())) {
        auto arr = tensor.cast<py::array_t<std::int64_t>>();
        tensor_sort_bitonic_cuda_impl<std::int64_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::uint8_t>())) {
        auto arr = tensor.cast<py::array_t<std::uint8_t>>();
        tensor_sort_bitonic_cuda_impl<std::uint8_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::uint16_t>())) {
        auto arr = tensor.cast<py::array_t<std::uint16_t>>();
        tensor_sort_bitonic_cuda_impl<std::uint16_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::uint32_t>())) {
        auto arr = tensor.cast<py::array_t<std::uint32_t>>();
        tensor_sort_bitonic_cuda_impl<std::uint32_t>(arr, sort_dim);
    } else if (tensor.dtype().is(py::dtype::of<std::uint64_t>())) {
        auto arr = tensor.cast<py::array_t<std::uint64_t>>();
        tensor_sort_bitonic_cuda_impl<std::uint64_t>(arr, sort_dim);
    } else {
        throw std::invalid_argument("Unsupported dtype for tensor sort: " + py::str(tensor.dtype()).cast<std::string>());
    }
}

// Python module definition
PYBIND11_MODULE(_sort_ops_cuda, m) {
    m.doc() = "CUDA sort operations for py-gpu-algos";

    // Low-level type-specific functions for tensor_sort_bitonic
    m.def("tensor_sort_bitonic_float32", &tensor_sort_bitonic_cuda_impl<float>,
          "3D tensor bitonic sort for float32", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_float64", &tensor_sort_bitonic_cuda_impl<double>,
          "3D tensor bitonic sort for float64", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_int8", &tensor_sort_bitonic_cuda_impl<std::int8_t>,
          "3D tensor bitonic sort for int8", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_int16", &tensor_sort_bitonic_cuda_impl<std::int16_t>,
          "3D tensor bitonic sort for int16", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_int32", &tensor_sort_bitonic_cuda_impl<std::int32_t>,
          "3D tensor bitonic sort for int32", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_int64", &tensor_sort_bitonic_cuda_impl<std::int64_t>,
          "3D tensor bitonic sort for int64", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_uint8", &tensor_sort_bitonic_cuda_impl<std::uint8_t>,
          "3D tensor bitonic sort for uint8", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_uint16", &tensor_sort_bitonic_cuda_impl<std::uint16_t>,
          "3D tensor bitonic sort for uint16", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_uint32", &tensor_sort_bitonic_cuda_impl<std::uint32_t>,
          "3D tensor bitonic sort for uint32", py::arg("tensor"), py::arg("sort_dim"));
    m.def("tensor_sort_bitonic_uint64", &tensor_sort_bitonic_cuda_impl<std::uint64_t>,
          "3D tensor bitonic sort for uint64", py::arg("tensor"), py::arg("sort_dim"));

    // High-level dispatch function
    m.def("tensor_sort_bitonic", &tensor_sort_bitonic_dispatch,
          "3D tensor bitonic sort with automatic type dispatch",
          py::arg("tensor"), py::arg("sort_dim"));
}
