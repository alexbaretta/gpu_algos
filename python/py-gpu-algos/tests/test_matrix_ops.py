"""
Tests for matrix operations in py-gpu-algos.

Tests all 8 matrix kernels:
- Matrix products: naive, tiled, warp, cublas, cutlass, tensor
- Matrix transpose: striped, tiled

Covers all 11 dtypes, error handling, and performance comparison.
"""

import itertools
import pytest
import numpy as np
import py_gpu_algos
from typing import List

from .conftest import ALL_DTYPES, FLOAT_DTYPES
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference_matrix_product,
    get_numpy_reference_matrix_transpose, ErrorCaseBuilder, print_performance_summary,
    create_non_contiguous_array
)

# Matrix product functions to test
CUDA_MATRIX_PRODUCT_FUNCTIONS = [
    'matrix_product_naive',
    'matrix_product_tiled',
    'matrix_product_warp',
]

TENSOR_MATRIX_PRODUCT_FUNCTIONS = [
    'matrix_product_tensor',
]

GEMM_MATRIX_PRODUCT_FUNCTIONS = [
    'matrix_product_cublas',
    'matrix_product_cutlass',
]

# Matrix transpose functions to test
CUDA_MATRIX_TRANSPOSE_FUNCTIONS = [
    'matrix_transpose_naive',
    'matrix_transpose_striped',
    'matrix_transpose_tiled'
]

GEMM_MATRIX_TRANSPOSE_FUNCTIONS = [
    'matrix_transpose_cublas',
]

def make_test_matrix_products(
        matrix_product_functions:List[str],
        dtypes:List[np.dtype],
        is_tensor_core:bool=False
    ):
    class MetaTestMatrixProducts:
        """Test all matrix product operations."""

        @pytest.mark.parametrize("func_name", matrix_product_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_incremental(self, func_name, dtype, test_input_matrix_incremental, test_shape_triplet):
            """Test basic matrix multiplication functionality."""
            func = getattr(py_gpu_algos, func_name)
            m, k, n = test_shape_triplet
            a = test_input_matrix_incremental(dtype, m, k)
            b = test_input_matrix_incremental(dtype, k, n) # In c++ we do not continue the sequence after the first matrix

            # Compute result
            result = func(a, b)

            # Validate basic properties
            expected_shape = (a.shape[0], b.shape[1])
            validate_basic_properties(result, expected_shape, dtype)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_matrix_product(a, b)
            assert_array_close(result, numpy_result, dtype, is_tensor_core=is_tensor_core)
            return

        @pytest.mark.parametrize("func_name", matrix_product_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_random(self, func_name, dtype, test_input_matrix_random, test_shape_triplet):
            """Test basic matrix multiplication functionality."""
            func = getattr(py_gpu_algos, func_name)
            m, k, n = test_shape_triplet
            a = test_input_matrix_random(dtype, m, k)
            b = test_input_matrix_random(dtype, k, n) # In c++ we do not continue the sequence after the first matrix

            # Compute result
            result = func(a, b)

            # Validate basic properties
            expected_shape = (a.shape[0], b.shape[1])
            validate_basic_properties(result, expected_shape, dtype)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_matrix_product(a, b)
            assert_array_close(result, numpy_result, dtype, is_tensor_core=is_tensor_core)
            return

        @pytest.mark.parametrize("func_name", matrix_product_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_low_level_functions(self, func_name, dtype, test_input_matrix_incremental, test_shape_triplet):
            """Test low-level type-specific functions."""
            dtype_name = dtype.__name__
            low_level_func_name = f"{func_name}_{dtype_name}"

            # Check if low-level function exists
            if not hasattr(py_gpu_algos, low_level_func_name):
                pytest.skip(f"Low-level function {low_level_func_name} not available")

            low_level_func = getattr(py_gpu_algos, low_level_func_name)
            high_level_func = getattr(py_gpu_algos, func_name)

            m, k, n = test_shape_triplet
            a = test_input_matrix_incremental(dtype, m, k)
            b = test_input_matrix_incremental(dtype, k, n) # In c++ we do not continue the sequence after the first matrix

            # Compare low-level and high-level results
            result_low = low_level_func(a, b)
            result_high = high_level_func(a, b)

            assert_array_close(result_low, result_high, dtype, is_tensor_core=is_tensor_core)
            return

        @pytest.mark.parametrize("func_name", matrix_product_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_error_handling(self, func_name, dtype, test_shape_triplet, test_input_matrix_random, test_input_tensor_3d_random):
            """Test error handling for invalid inputs."""
            func = getattr(py_gpu_algos, func_name)

            # Create test arrays
            m, k, n = test_shape_triplet
            a_float32 = test_input_matrix_random(dtype, 10, 5)
            b_float32 = test_input_matrix_random(dtype, 5, 8)
            a_float64 = test_input_matrix_random(dtype, 10, 5)
            b_wrong_dim = test_input_matrix_random(dtype, 8, 12)  # 5 != 8
            a_1d = test_input_matrix_random(dtype, 10, 1)
            a_3d = test_input_tensor_3d_random(dtype, 10, 5, 3)

            # Build error test cases
            error_cases = (ErrorCaseBuilder()
                .add_dimension_mismatch(func_name, a_float32, b_wrong_dim)
                .add_dtype_mismatch(func_name, a_float32, a_float64)
                .add_shape_error(func_name, a_1d, b_float32)
                .add_shape_error(func_name, a_float32, a_1d)
                .add_shape_error(func_name, a_3d, b_float32)
                .build())

            validate_function_error_cases(func, error_cases)
            return

        @pytest.mark.parametrize("func_name", matrix_product_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_non_contiguous_arrays(self, func_name, dtype, test_shape_triplet, test_input_matrix_incremental):
            """Test with non-contiguous array views."""
            func = getattr(py_gpu_algos, func_name)

            # Create large base arrays
            m, k, n = test_shape_triplet
            base_a = test_input_matrix_incremental(dtype, m, k)
            base_b = test_input_matrix_incremental(dtype, k, n) # In c++ we do not continue the sequence after the first matrix

            # Create non-contiguous views
            a_view = base_a[::2, :]  # Every other row
            b_view = base_b[:, ::2]  # Every other column

            # Function should handle non-contiguous arrays (convert to contiguous)
            result = func(a_view, b_view)
            numpy_result = get_numpy_reference_matrix_product(a_view, b_view)

            assert_array_close(result, numpy_result, dtype, is_tensor_core=is_tensor_core)
            return

    return MetaTestMatrixProducts

TestMatrixProductsCUDA = make_test_matrix_products(CUDA_MATRIX_PRODUCT_FUNCTIONS, ALL_DTYPES, is_tensor_core=False)
TestMatrixProductsTensor = make_test_matrix_products(TENSOR_MATRIX_PRODUCT_FUNCTIONS, FLOAT_DTYPES, is_tensor_core=True)
TestMatrixProductsGEMM = make_test_matrix_products(GEMM_MATRIX_PRODUCT_FUNCTIONS, FLOAT_DTYPES, is_tensor_core=True)

def make_test_matrix_transpose(
        matrix_transpose_functions:List[str],
        dtypes:List[np.dtype],
        is_tensor_core:bool=False
    ):

    class MetaTestMatrixTranspose:
        """Test all matrix transpose operations."""

        @pytest.mark.parametrize("func_name", matrix_transpose_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_basic_functionality(self, func_name, dtype, test_input_matrix_incremental, test_shape_triplet):
            """Test basic matrix transpose functionality."""
            func = getattr(py_gpu_algos, func_name)
            m, k, n = test_shape_triplet
            a = test_input_matrix_incremental(dtype, m, k)

            # Compute result
            result = func(a)

            # Validate basic properties
            expected_shape = (a.shape[1], a.shape[0])  # Transposed dimensions
            validate_basic_properties(result, expected_shape, dtype)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_matrix_transpose(a)
            assert_array_close(result, numpy_result, dtype)
            return

        @pytest.mark.parametrize("func_name", matrix_transpose_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_low_level_functions(self, func_name, dtype, test_input_matrix_incremental, test_shape_triplet):
            """Test low-level type-specific functions."""
            dtype_name = dtype.__name__
            low_level_func_name = f"{func_name}_{dtype_name}"

            # Check if low-level function exists
            if not hasattr(py_gpu_algos, low_level_func_name):
                pytest.skip(f"Low-level function {low_level_func_name} not available")

            low_level_func = getattr(py_gpu_algos, low_level_func_name)
            high_level_func = getattr(py_gpu_algos, func_name)

            m, k, n = test_shape_triplet
            a = test_input_matrix_incremental(dtype, m, k)

            # Compare low-level and high-level results
            result_low = low_level_func(a)
            result_high = high_level_func(a)

            assert_array_close(result_low, result_high, dtype)
            return

        @pytest.mark.parametrize("func_name", matrix_transpose_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_different_shapes(self, func_name, dtype, test_input_matrix_random, test_shape_triplet):
            """Test matrix transpose with different shapes."""
            func = getattr(py_gpu_algos, func_name)

            test_shapes = [
                (10, 8),     # Rectangular
                (64, 64),    # Square
                (100, 1),    # Very thin
                (1, 100),    # Very wide
                (32, 48),    # Multiple of common block sizes
            ]

            for m, n in test_shapes:
                a = test_input_matrix_random(dtype, m, n)

                result = func(a)
                numpy_result = get_numpy_reference_matrix_transpose(a)

                validate_basic_properties(result, (n, m), dtype)
                assert_array_close(result, numpy_result, dtype)
                pass
            return

        @pytest.mark.parametrize("func_name", matrix_transpose_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_square_matrices(self, func_name, dtype, test_input_matrix_random, test_shape_triplet):
            """Test transpose of square matrices."""
            func = getattr(py_gpu_algos, func_name)

            # Test various square sizes
            size, _, _ = test_shape_triplet

            a = test_input_matrix_random(dtype, size, size)

            result = func(a)
            numpy_result = get_numpy_reference_matrix_transpose(a)

            validate_basic_properties(result, (size, size), dtype)
            assert_array_close(result, numpy_result, dtype)
            return

        @pytest.mark.parametrize("func_name", matrix_transpose_functions)
        @pytest.mark.parametrize("dtype", dtypes)
        def test_error_handling(self, func_name, dtype, test_input_vector_random, test_input_tensor_3d_random, test_shape_triplet):
            """Test error handling for invalid inputs."""
            func = getattr(py_gpu_algos, func_name)

            # Create test arrays
            m, k, n = test_shape_triplet
            a_1d = test_input_vector_random(dtype, 10)
            a_3d = test_input_tensor_3d_random(dtype, 10, 5, 3)

            # Build error test cases
            error_cases = (ErrorCaseBuilder()
                .add_shape_error(func_name, a_1d)
                .add_shape_error(func_name, a_3d)
                .build())

            validate_function_error_cases(func, error_cases)
            return

    return MetaTestMatrixTranspose

TestMatrixTransposeCUDA = make_test_matrix_transpose(CUDA_MATRIX_TRANSPOSE_FUNCTIONS, ALL_DTYPES, is_tensor_core=False)
TestMatrixTransposeGEMM = make_test_matrix_transpose(GEMM_MATRIX_TRANSPOSE_FUNCTIONS, FLOAT_DTYPES, is_tensor_core=False)
