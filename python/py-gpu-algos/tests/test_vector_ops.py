"""
Tests for vector operations in py-gpu-algos.

Tests all 4 vector kernels:
- vector_cumsum_serial: Serial cumulative sum
- vector_cumsum_parallel: Parallel cumulative sum
- vector_cummax_parallel: Parallel cumulative maximum
- vector_scan_parallel: Generic scan with operations (sum, max, min, prod)

Covers all 11 dtypes, error handling, and performance comparison.
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference_cumsum_serial,
    get_numpy_reference_cumsum_parallel, get_numpy_reference_cummax_parallel,
    get_numpy_reference_scan_parallel, ErrorCaseBuilder, print_performance_summary
)

# Vector function names to test
CUMSUM_FUNCTIONS = ['vector_cumsum_serial', 'vector_cumsum_parallel']
CUMMAX_FUNCTIONS = ['vector_cummax_parallel']
SCAN_FUNCTIONS = ['vector_scan_parallel']

# All vector functions
ALL_VECTOR_FUNCTIONS = CUMSUM_FUNCTIONS + CUMMAX_FUNCTIONS + SCAN_FUNCTIONS

class TestVectorCumsum:
    """Test cumulative sum operations."""

    @pytest.mark.parametrize("func_name", CUMSUM_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_all, small_vectors):
        """Test basic cumulative sum functionality."""
        func = getattr(py_gpu_algos, func_name)
        vec = small_vectors(dtype_all)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        if func_name == 'vector_cumsum_serial':
            numpy_result = get_numpy_reference_cumsum_serial(vec)
        else:
            numpy_result = get_numpy_reference_cumsum_parallel(vec)

        assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("func_name", CUMSUM_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_all, small_vectors):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = small_vectors(dtype_all)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec)

        assert_array_close(result_low, result_high, dtype_all)

    @pytest.mark.parametrize("func_name", CUMSUM_FUNCTIONS)
    def test_different_sizes(self, func_name, dtype_float):
        """Test cumulative sum with different vector sizes."""
        func = getattr(py_gpu_algos, func_name)

        test_sizes = [10, 100, 1000, 4096, 10000]

        for size in test_sizes:
            np.random.seed(42)
            vec = np.random.randn(size).astype(dtype_float)

            result = func(vec)
            numpy_result = np.cumsum(vec)

            validate_basic_properties(result, (size,), dtype_float)
            assert_array_close(result, numpy_result, dtype_float)

    @pytest.mark.parametrize("func_name", CUMSUM_FUNCTIONS)
    def test_edge_cases(self, func_name, dtype_all):
        """Test edge cases for cumulative sum."""
        func = getattr(py_gpu_algos, func_name)

        # Single element vector
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single)
        numpy_result_single = np.cumsum(vec_single)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # Vector with zeros
        vec_zeros = np.zeros(100).astype(dtype_all)
        result_zeros = func(vec_zeros)
        numpy_result_zeros = np.cumsum(vec_zeros)
        assert_array_close(result_zeros, numpy_result_zeros, dtype_all)

        # Vector with negative values (for signed types)
        if not np.issubdtype(dtype_all, np.unsignedinteger):
            vec_negative = np.array([-1, -2, -3, -4, -5]).astype(dtype_all)
            result_negative = func(vec_negative)
            numpy_result_negative = np.cumsum(vec_negative)
            assert_array_close(result_negative, numpy_result_negative, dtype_all)

    def test_serial_vs_parallel_consistency(self, dtype_all, small_vectors):
        """Test that serial and parallel cumsum give same results."""
        vec = small_vectors(dtype_all)

        result_serial = py_gpu_algos.vector_cumsum_serial(vec)
        result_parallel = py_gpu_algos.vector_cumsum_parallel(vec)

        assert_array_close(result_serial, result_parallel, dtype_all)

class TestVectorCummax:
    """Test cumulative maximum operations."""

    @pytest.mark.parametrize("func_name", CUMMAX_FUNCTIONS)
    def test_basic_functionality(self, func_name, dtype_all, small_vectors):
        """Test basic cumulative maximum functionality."""
        func = getattr(py_gpu_algos, func_name)
        vec = small_vectors(dtype_all)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_cummax_parallel(vec)
        assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("func_name", CUMMAX_FUNCTIONS)
    def test_low_level_functions(self, func_name, dtype_all, small_vectors):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = small_vectors(dtype_all)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec)

        assert_array_close(result_low, result_high, dtype_all)

    @pytest.mark.parametrize("func_name", CUMMAX_FUNCTIONS)
    def test_monotonic_sequences(self, func_name, dtype_all):
        """Test cummax with monotonic sequences."""
        func = getattr(py_gpu_algos, func_name)

        # Increasing sequence
        if np.issubdtype(dtype_all, np.integer):
            vec_inc = np.arange(1, 101, dtype=dtype_all)
        else:
            vec_inc = np.linspace(0.1, 10.0, 100).astype(dtype_all)

        result_inc = func(vec_inc)
        numpy_result_inc = np.maximum.accumulate(vec_inc)
        assert_array_close(result_inc, numpy_result_inc, dtype_all)

        # Decreasing sequence (cummax should plateau)
        vec_dec = vec_inc[::-1]
        result_dec = func(vec_dec)
        numpy_result_dec = np.maximum.accumulate(vec_dec)
        assert_array_close(result_dec, numpy_result_dec, dtype_all)

    @pytest.mark.parametrize("func_name", CUMMAX_FUNCTIONS)
    def test_edge_cases(self, func_name, dtype_all):
        """Test edge cases for cumulative maximum."""
        func = getattr(py_gpu_algos, func_name)

        # Single element
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single)
        numpy_result_single = np.maximum.accumulate(vec_single)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # All same values
        vec_same = np.full(100, 5).astype(dtype_all)
        result_same = func(vec_same)
        numpy_result_same = np.maximum.accumulate(vec_same)
        assert_array_close(result_same, numpy_result_same, dtype_all)

class TestVectorScan:
    """Test generic scan operations."""

    @pytest.mark.parametrize("operation", ["sum", "max", "min", "prod"])
    def test_basic_functionality(self, operation, dtype_all, small_vectors):
        """Test basic scan functionality for all operations."""
        func = py_gpu_algos.vector_scan_parallel
        vec = small_vectors(dtype_all)

        # Compute result
        result = func(vec, operation)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_scan_parallel(vec, operation)
        assert_array_close(result, numpy_result, dtype_all)

    @pytest.mark.parametrize("operation", ["sum", "max", "min", "prod"])
    def test_low_level_functions(self, operation, dtype_all, small_vectors):
        """Test low-level type-specific scan functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"vector_scan_parallel_{operation}_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = py_gpu_algos.vector_scan_parallel

        vec = small_vectors(dtype_all)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec, operation)

        assert_array_close(result_low, result_high, dtype_all)

    def test_scan_sum_vs_cumsum(self, dtype_all, small_vectors):
        """Test that scan with sum operation matches cumsum."""
        vec = small_vectors(dtype_all)

        scan_result = py_gpu_algos.vector_scan_parallel(vec, "sum")
        cumsum_result = py_gpu_algos.vector_cumsum_parallel(vec)

        assert_array_close(scan_result, cumsum_result, dtype_all)

    def test_scan_max_vs_cummax(self, dtype_all, small_vectors):
        """Test that scan with max operation matches cummax."""
        vec = small_vectors(dtype_all)

        scan_result = py_gpu_algos.vector_scan_parallel(vec, "max")
        cummax_result = py_gpu_algos.vector_cummax_parallel(vec)

        assert_array_close(scan_result, cummax_result, dtype_all)

    @pytest.mark.parametrize("operation", ["sum", "max", "min", "prod"])
    def test_scan_different_sizes(self, operation, dtype_float):
        """Test scan operations with different vector sizes."""
        func = py_gpu_algos.vector_scan_parallel

        test_sizes = [10, 100, 1000, 4096]

        for size in test_sizes:
            np.random.seed(42)
            # For prod operation, use smaller values to avoid overflow
            if operation == "prod":
                vec = (np.random.randn(size) * 0.1 + 1.0).astype(dtype_float)
            else:
                vec = np.random.randn(size).astype(dtype_float)

            result = func(vec, operation)
            numpy_result = get_numpy_reference_scan_parallel(vec, operation)

            validate_basic_properties(result, (size,), dtype_float)

            # For prod operation, allow larger tolerance due to accumulation errors
            if operation == "prod":
                assert_array_close(result, numpy_result, dtype_float, rtol=1e-3, atol=1e-4)
            else:
                assert_array_close(result, numpy_result, dtype_float)

    def test_scan_edge_cases(self, dtype_all):
        """Test scan operations with edge cases."""
        # Single element
        vec_single = np.array([42]).astype(dtype_all)

        for operation in ["sum", "max", "min", "prod"]:
            result = py_gpu_algos.vector_scan_parallel(vec_single, operation)
            numpy_result = get_numpy_reference_scan_parallel(vec_single, operation)
            assert_array_close(result, numpy_result, dtype_all)

    def test_invalid_operation(self, dtype_float, small_vectors):
        """Test that invalid operation names raise errors."""
        vec = small_vectors(dtype_float)

        with pytest.raises(ValueError):
            py_gpu_algos.vector_scan_parallel(vec, "invalid_operation")

class TestVectorOperationsErrorHandling:
    """Test error handling for all vector operations."""

    @pytest.mark.parametrize("func_name", ALL_VECTOR_FUNCTIONS)
    def test_non_vector_inputs(self, func_name):
        """Test error handling for non-1D inputs."""
        if func_name == 'vector_scan_parallel':
            func = lambda x: getattr(py_gpu_algos, func_name)(x, "sum")
        else:
            func = getattr(py_gpu_algos, func_name)

        # 2D array
        arr_2d = np.random.randn(10, 5).astype(np.float32)
        # 3D array
        arr_3d = np.random.randn(10, 5, 3).astype(np.float32)
        # 0D array (scalar)
        arr_0d = np.array(42.0).astype(np.float32)

        error_cases = (ErrorCaseBuilder()
            .add_shape_error(func_name, (arr_2d,), {})
            .add_shape_error(func_name, (arr_3d,), {})
            .add_shape_error(func_name, (arr_0d,), {})
            .build())

        validate_function_error_cases(func, error_cases)

    def test_empty_vector(self):
        """Test handling of empty vectors."""
        empty_vec = np.array([]).astype(np.float32)

        # Most operations should handle empty vectors gracefully
        for func_name in CUMSUM_FUNCTIONS + CUMMAX_FUNCTIONS:
            func = getattr(py_gpu_algos, func_name)
            result = func(empty_vec)
            assert result.shape == (0,)
            assert result.dtype == np.float32

    def test_scan_empty_vector(self):
        """Test scan operations with empty vectors."""
        empty_vec = np.array([]).astype(np.float32)

        for operation in ["sum", "max", "min", "prod"]:
            result = py_gpu_algos.vector_scan_parallel(empty_vec, operation)
            assert result.shape == (0,)
            assert result.dtype == np.float32

class TestVectorOperationsPerformance:
    """Performance tests for vector operations."""

    @pytest.mark.performance
    @pytest.mark.parametrize("func_name", CUMSUM_FUNCTIONS)
    def test_cumsum_performance(self, func_name, performance_sizes):
        """Test cumsum performance compared to NumPy."""
        func = getattr(py_gpu_algos, func_name)
        dtype = np.float32

        # Use first dimension of performance_sizes as vector size
        for size, _, _ in performance_sizes:
            size = size * 1000  # Make vectors larger for meaningful performance test

            np.random.seed(42)
            vec = np.random.randn(size).astype(dtype)

            results = compare_with_numpy_reference(
                func, np.cumsum, (vec,), dtype
            )

            assert results['accuracy_pass'], f"Accuracy failed for {func_name} with size {size}"

            if size >= 64000:  # Only print for larger sizes
                print_performance_summary(results, f"{func_name} (size {size})")

    @pytest.mark.performance
    @pytest.mark.parametrize("operation", ["sum", "max", "min"])  # Skip prod to avoid overflow
    def test_scan_performance(self, operation, performance_sizes):
        """Test scan performance compared to NumPy."""
        func = py_gpu_algos.vector_scan_parallel
        dtype = np.float32

        # Use first dimension as vector size
        for size, _, _ in performance_sizes:
            size = size * 1000  # Make vectors larger

            np.random.seed(42)
            vec = np.random.randn(size).astype(dtype)

            # Create wrapper function for comparison
            gpu_func = lambda v: func(v, operation)
            numpy_func = lambda v: get_numpy_reference_scan_parallel(v, operation)

            results = compare_with_numpy_reference(
                gpu_func, numpy_func, (vec,), dtype
            )

            assert results['accuracy_pass'], f"Accuracy failed for scan_{operation} with size {size}"

            if size >= 64000:
                print_performance_summary(results, f"scan_{operation} (size {size})")

class TestVectorOperationsIntegration:
    """Integration tests combining multiple vector operations."""

    def test_cumsum_consistency_across_algorithms(self, dtype_float, small_vectors):
        """Test that all cumsum algorithms produce consistent results."""
        vec = small_vectors(dtype_float)

        # Compare all cumsum-like operations
        cumsum_serial = py_gpu_algos.vector_cumsum_serial(vec)
        cumsum_parallel = py_gpu_algos.vector_cumsum_parallel(vec)
        scan_sum = py_gpu_algos.vector_scan_parallel(vec, "sum")

        # All should be identical
        assert_array_close(cumsum_serial, cumsum_parallel, dtype_float)
        assert_array_close(cumsum_serial, scan_sum, dtype_float)
        assert_array_close(cumsum_parallel, scan_sum, dtype_float)

    def test_cummax_consistency(self, dtype_float, small_vectors):
        """Test that cummax and scan max produce consistent results."""
        vec = small_vectors(dtype_float)

        cummax_result = py_gpu_algos.vector_cummax_parallel(vec)
        scan_max_result = py_gpu_algos.vector_scan_parallel(vec, "max")

        assert_array_close(cummax_result, scan_max_result, dtype_float)

    def test_multiple_operations_on_same_vector(self, dtype_float):
        """Test multiple operations on the same vector."""
        np.random.seed(42)
        vec = np.random.randn(1000).astype(dtype_float)

        # Run all operations
        cumsum_result = py_gpu_algos.vector_cumsum_parallel(vec)
        cummax_result = py_gpu_algos.vector_cummax_parallel(vec)
        scan_min_result = py_gpu_algos.vector_scan_parallel(vec, "min")

        # Verify properties
        # Last element of cumsum should equal sum of vector
        assert abs(cumsum_result[-1] - np.sum(vec)) < 1e-5

        # Last element of cummax should equal max of vector
        assert abs(cummax_result[-1] - np.max(vec)) < 1e-6

        # Last element of cummin should equal min of vector
        assert abs(scan_min_result[-1] - np.min(vec)) < 1e-6

        # Cumsum should be monotonically increasing for positive vectors
        vec_positive = np.abs(vec)
        cumsum_positive = py_gpu_algos.vector_cumsum_parallel(vec_positive)
        assert np.all(np.diff(cumsum_positive) >= 0), "Cumsum of positive vector should be monotonic"

    def test_operation_chaining(self, dtype_float):
        """Test chaining multiple vector operations."""
        np.random.seed(42)
        vec = np.abs(np.random.randn(500)).astype(dtype_float) + 0.1  # Positive values

        # Chain: vec -> cumsum -> cummax
        cumsum_result = py_gpu_algos.vector_cumsum_parallel(vec)
        final_result = py_gpu_algos.vector_cummax_parallel(cumsum_result)

        # Compare with NumPy reference
        numpy_cumsum = np.cumsum(vec)
        numpy_final = np.maximum.accumulate(numpy_cumsum)

        assert_array_close(final_result, numpy_final, dtype_float)

        # Since cumsum of positive values is monotonic, cummax should be identity
        assert_array_close(final_result, cumsum_result, dtype_float)
