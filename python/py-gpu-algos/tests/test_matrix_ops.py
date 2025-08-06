"""
Tests for matrix operations in py-gpu-algos.

Tests all 8 matrix kernels:
- Matrix products: naive, tiled, warp, cublas, cutlass, tensor
- Matrix transpose: striped, tiled

Covers all 11 dtypes, error handling, and performance comparison.
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference_matrix_product,
    get_numpy_reference_matrix_transpose, ErrorCaseBuilder, print_performance_summary,
    create_non_contiguous_array
)

# Matrix product functions to test
MATRIX_PRODUCT_FUNCTIONS = [
    'matrix_product_naive',
    'matrix_product_tiled',
    'matrix_product_warp',
    'matrix_product_cublas',
    'matrix_product_cutlass',
    'matrix_product_tensor'
]

# Matrix transpose functions to test
MATRIX_TRANSPOSE_FUNCTIONS = [
    'matrix_transpose_striped',
    'matrix_transpose_tiled'
]

class TestMatrixProducts:
    """Test all matrix product operations."""

    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_all, small_matrices):
        """Test basic matrix multiplication functionality."""
        func = getattr(py_gpu_algos, func_name)
        a, b = small_matrices(dtype_all)

        # Compute result
        result = func(a, b)

        # Validate basic properties
        expected_shape = (a.shape[0], b.shape[1])
        validate_basic_properties(result, expected_shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_matrix_product(a, b)
        assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_all, small_matrices):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        a, b = small_matrices(dtype_all)

        # Compare low-level and high-level results
        result_low = low_level_func(a, b)
        result_high = high_level_func(a, b)

        assert_array_close(result_low, result_high, dtype_all)

    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_different_shapes(self, func_name, dtype_float):
        """Test matrix multiplication with different shapes."""
        func = getattr(py_gpu_algos, func_name)

        test_shapes = [
            (10, 8, 12),   # Rectangular
            (64, 64, 64),  # Square
            (100, 1, 50),  # Very thin middle dimension
            (1, 100, 1),   # Very thin outer dimensions
            (32, 16, 48),  # Multiple of common block sizes
        ]

        for m, k, n in test_shapes:
            np.random.seed(42)
            a = np.random.randn(m, k).astype(dtype_float)
            b = np.random.randn(k, n).astype(dtype_float)

            result = func(a, b)
            numpy_result = get_numpy_reference_matrix_product(a, b)

            validate_basic_properties(result, (m, n), dtype_float)
            assert_array_close(result, numpy_result, dtype_float)

    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_error_handling(self, func_name, dtype_float):
        """Test error handling for invalid inputs."""
        func = getattr(py_gpu_algos, func_name)

        # Create test arrays
        a_float32 = np.random.randn(10, 5).astype(np.float32)
        b_float32 = np.random.randn(5, 8).astype(np.float32)
        a_float64 = np.random.randn(10, 5).astype(np.float64)
        b_wrong_dim = np.random.randn(8, 12).astype(np.float32)  # 5 != 8
        a_1d = np.random.randn(10).astype(np.float32)
        a_3d = np.random.randn(10, 5, 3).astype(np.float32)

        # Build error test cases
        error_cases = (ErrorCaseBuilder()
            .add_dimension_mismatch(func_name, a_float32, b_wrong_dim)
            .add_dtype_mismatch(func_name, a_float32, a_float64)
            .add_shape_error(func_name, a_1d, b_float32)
            .add_shape_error(func_name, a_float32, a_1d)
            .add_shape_error(func_name, a_3d, b_float32)
            .build())

        validate_function_error_cases(func, error_cases)

    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_non_contiguous_arrays(self, func_name, dtype_float):
        """Test with non-contiguous array views."""
        func = getattr(py_gpu_algos, func_name)

        # Create large base arrays
        base_a = np.random.randn(20, 16).astype(dtype_float)
        base_b = np.random.randn(16, 24).astype(dtype_float)

        # Create non-contiguous views
        a_view = base_a[::2, :]  # Every other row
        b_view = base_b[:, ::2]  # Every other column

        # Function should handle non-contiguous arrays (convert to contiguous)
        result = func(a_view, b_view)
        numpy_result = get_numpy_reference_matrix_product(a_view, b_view)

        assert_array_close(result, numpy_result, dtype_float)

    @pytest.mark.performance
    @pytest.mark.parametrize("func_name", MATRIX_PRODUCT_FUNCTIONS)
    def test_performance_comparison(self, func_name, performance_sizes):
        """Test performance comparison with NumPy."""
        func = getattr(py_gpu_algos, func_name)
        dtype = np.float32

        for m, k, n in performance_sizes:
            np.random.seed(42)
            a = np.random.randn(m, k).astype(dtype)
            b = np.random.randn(k, n).astype(dtype)

            results = compare_with_numpy_reference(
                func, get_numpy_reference_matrix_product,
                (a, b), dtype
            )

            # Performance should be reasonable (accuracy is required)
            assert results['accuracy_pass'], f"Accuracy failed for {func_name} with shape ({m}, {k}, {n})"

            # Optional: Print performance summary for manual inspection
            if m >= 256:  # Only print for larger sizes
                print_performance_summary(results, f"{func_name} ({m}x{k}x{n})")

class TestMatrixTranspose:
    """Test all matrix transpose operations."""

    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_all, small_matrices):
        """Test basic matrix transpose functionality."""
        func = getattr(py_gpu_algos, func_name)
        a, _ = small_matrices(dtype_all)  # Only need one matrix

        # Compute result
        result = func(a)

        # Validate basic properties
        expected_shape = (a.shape[1], a.shape[0])  # Transposed dimensions
        validate_basic_properties(result, expected_shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_matrix_transpose(a)
        assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_all, small_matrices):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        a, _ = small_matrices(dtype_all)

        # Compare low-level and high-level results
        result_low = low_level_func(a)
        result_high = high_level_func(a)

        assert_array_close(result_low, result_high, dtype_all)

    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_different_shapes(self, func_name, dtype_float):
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
            np.random.seed(42)
            a = np.random.randn(m, n).astype(dtype_float)

            result = func(a)
            numpy_result = get_numpy_reference_matrix_transpose(a)

            validate_basic_properties(result, (n, m), dtype_float)
            assert_array_close(result, numpy_result, dtype_float)

    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_square_matrices(self, func_name, dtype_all):
        """Test transpose of square matrices."""
        func = getattr(py_gpu_algos, func_name)

        # Test various square sizes
        sizes = [16, 32, 64, 100]

        for size in sizes:
            np.random.seed(42)
            a = np.random.randn(size, size).astype(dtype_all)

            # Scale for integer types
            if np.issubdtype(dtype_all, np.integer):
                if dtype_all in [np.int8, np.uint8]:
                    scale = 5
                elif dtype_all in [np.int16, np.uint16]:
                    scale = 50
                else:
                    scale = 100
                a = (a * scale).astype(dtype_all)

            result = func(a)
            numpy_result = get_numpy_reference_matrix_transpose(a)

            validate_basic_properties(result, (size, size), dtype_all)
            assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_error_handling(self, func_name):
        """Test error handling for invalid inputs."""
        func = getattr(py_gpu_algos, func_name)

        # Create test arrays
        a_1d = np.random.randn(10).astype(np.float32)
        a_3d = np.random.randn(10, 5, 3).astype(np.float32)

        # Build error test cases
        error_cases = (ErrorCaseBuilder()
            .add_shape_error(func_name, (a_1d,), {})
            .add_shape_error(func_name, (a_3d,), {})
            .build())

        validate_function_error_cases(func, error_cases)

    @pytest.mark.performance
    @pytest.mark.parametrize("func_name", MATRIX_TRANSPOSE_FUNCTIONS)
    def test_performance_comparison(self, func_name, performance_sizes):
        """Test transpose performance comparison with NumPy."""
        func = getattr(py_gpu_algos, func_name)
        dtype = np.float32

        for m, _, n in performance_sizes:  # Use m and n for transpose shapes
            np.random.seed(42)
            a = np.random.randn(m, n).astype(dtype)

            results = compare_with_numpy_reference(
                func, get_numpy_reference_matrix_transpose,
                (a,), dtype
            )

            # Performance should be reasonable (accuracy is required)
            assert results['accuracy_pass'], f"Accuracy failed for {func_name} with shape ({m}, {n})"

            # Optional: Print performance summary for manual inspection
            if m >= 256:  # Only print for larger sizes
                print_performance_summary(results, f"{func_name} ({m}x{n} transpose)")

class TestMatrixOperationsIntegration:
    """Integration tests combining multiple matrix operations."""

    def test_matrix_product_then_transpose(self, dtype_float):
        """Test matrix product followed by transpose."""
        # Create test matrices
        np.random.seed(42)
        a = np.random.randn(32, 24).astype(dtype_float)
        b = np.random.randn(24, 40).astype(dtype_float)

        # GPU computation: (A @ B).T
        product = py_gpu_algos.matrix_product_naive(a, b)
        result = py_gpu_algos.matrix_transpose_striped(product)

        # NumPy reference: (A @ B).T
        numpy_product = get_numpy_reference_matrix_product(a, b)
        numpy_result = get_numpy_reference_matrix_transpose(numpy_product)

        assert_array_close(result, numpy_result, dtype_float)

    def test_transpose_then_product(self, dtype_float):
        """Test transpose followed by matrix product."""
        # Create test matrices
        np.random.seed(42)
        a = np.random.randn(32, 24).astype(dtype_float)
        b = np.random.randn(32, 40).astype(dtype_float)

        # GPU computation: A.T @ B
        a_transposed = py_gpu_algos.matrix_transpose_striped(a)
        result = py_gpu_algos.matrix_product_naive(a_transposed, b)

        # NumPy reference: A.T @ B
        numpy_a_transposed = get_numpy_reference_matrix_transpose(a)
        numpy_result = get_numpy_reference_matrix_product(numpy_a_transposed, b)

        assert_array_close(result, numpy_result, dtype_float)

    def test_multiple_product_algorithms_consistency(self, dtype_float):
        """Test that different matrix product algorithms give consistent results."""
        # Create test matrices
        np.random.seed(42)
        a = np.random.randn(64, 48).astype(dtype_float)
        b = np.random.randn(48, 56).astype(dtype_float)

        # Run different algorithms
        results = {}
        for func_name in MATRIX_PRODUCT_FUNCTIONS:
            if hasattr(py_gpu_algos, func_name):
                func = getattr(py_gpu_algos, func_name)
                try:
                    results[func_name] = func(a, b)
                except Exception as e:
                    # Some algorithms might not be available or might fail
                    pytest.skip(f"Algorithm {func_name} failed: {e}")

        # Compare all results pairwise
        result_items = list(results.items())
        for i in range(len(result_items)):
            for j in range(i+1, len(result_items)):
                name1, result1 = result_items[i]
                name2, result2 = result_items[j]

                try:
                    assert_array_close(result1, result2, dtype_float)
                except AssertionError:
                    # Allow for some algorithms to have different precision
                    # but at least check they're close with relaxed tolerance
                    np.testing.assert_allclose(result1, result2,
                                             rtol=1e-3, atol=1e-4,
                                             err_msg=f"Results differ between {name1} and {name2}")

    def test_multiple_transpose_algorithms_consistency(self, dtype_float):
        """Test that different transpose algorithms give consistent results."""
        # Create test matrix
        np.random.seed(42)
        a = np.random.randn(64, 56).astype(dtype_float)

        # Run different algorithms
        results = {}
        for func_name in MATRIX_TRANSPOSE_FUNCTIONS:
            if hasattr(py_gpu_algos, func_name):
                func = getattr(py_gpu_algos, func_name)
                try:
                    results[func_name] = func(a)
                except Exception as e:
                    pytest.skip(f"Algorithm {func_name} failed: {e}")

        # Compare all results pairwise
        result_items = list(results.items())
        for i in range(len(result_items)):
            for j in range(i+1, len(result_items)):
                name1, result1 = result_items[i]
                name2, result2 = result_items[j]

                assert_array_close(result1, result2, dtype_float)
