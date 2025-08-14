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
from tests.conftest import test_input_vector_incremental
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference, get_numpy_reference_scan,
    get_numpy_reference_reduction, ErrorCaseBuilder, print_performance_summary
)

# Vector function names to test
# CUMSUM_FUNCTIONS = ['vector_cumsum_serial', 'vector_cumsum_parallel']
CUMOP_FUNCTIONS = ['vector_cumsum_serial', 'vector_cumsum_parallel', 'vector_cummax_parallel']
REDUCTION_FUNCTIONS = ['vector_reduction_recursive']
AGGOP_FUNCTIONS = ['vector_sum_atomic']
SCAN_FUNCTIONS = ['vector_scan_parallel']
OPERATIONS = ["sum", "max", "min", "prod"]

TEST_SIZES = [10, 100, 1000, 4096, 10000, 1000000]

np.seterr(under='warn', over='warn')

class TestVectorCumOp:
    """Test cumulative operations."""

    @pytest.mark.parametrize("func_name", CUMOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_incremental(self, func_name, dtype_all, test_input_vector_incremental, test_size):
        """Test cumulative operations with incremental input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", CUMOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_random(self, func_name, dtype_all, test_input_vector_random, test_size):
        """Test cumulative operations with random input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_random(dtype_all, test_size)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", CUMOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", [1000])
    def test_low_level_functions(self, func_name, dtype_all, test_input_vector_incremental, test_size):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check that low-level function exists
        assert hasattr(py_gpu_algos, low_level_func_name), \
            f"Low-level function {low_level_func_name} not available"

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec)

        np.testing.assert_array_equal(
            result_low, result_high,
            err_msg=f"Low-level function {low_level_func_name} does not match high-level function {func_name}"
        )
        return

    @pytest.mark.parametrize("func_name", CUMOP_FUNCTIONS)
    def test_edge_cases(self, func_name, dtype_all):
        """Test edge cases for cumulative sum."""
        func = getattr(py_gpu_algos, func_name)

        # Single element vector
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single)
        numpy_result_single = get_numpy_reference(func_name, vec_single)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # Vector with zeros
        vec_zeros = np.zeros(100).astype(dtype_all)
        result_zeros = func(vec_zeros)
        numpy_result_zeros = get_numpy_reference(func_name, vec_zeros)
        assert_array_close(result_zeros, numpy_result_zeros, dtype_all)

        # Vector with negative values (for signed types)
        if not np.issubdtype(dtype_all, np.unsignedinteger):
            vec_negative = np.array([-1, -2, -3, -4, -5]).astype(dtype_all)
            result_negative = func(vec_negative)
            numpy_result_negative = get_numpy_reference(func_name, vec_negative)
            assert_array_close(result_negative, numpy_result_negative, dtype_all)
            pass
        return


class TestVectorScan:
    """Test generic scan operations."""

    @pytest.mark.parametrize("func_name", SCAN_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_incremental(self, func_name, operation, dtype_all, test_input_vector_incremental, test_size):
        """Test cumulative operations with incremental input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compute result
        result = func(vec, operation)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec, operation)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", SCAN_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_random(self, func_name, operation, dtype_all, test_input_vector_random, test_size):
        """Test cumulative operations with random input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_random(dtype_all, test_size)

        # Compute result
        result = func(vec, operation)

        # Validate basic properties
        validate_basic_properties(result, vec.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec, operation)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", SCAN_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", [1000])
    def test_low_level_functions(self, func_name, operation, dtype_all, test_input_vector_incremental, test_size):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{operation}_{dtype_name}"

        # Check that low-level function exists
        assert hasattr(py_gpu_algos, low_level_func_name), \
            f"Low-level function {low_level_func_name} not available"

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec, operation)

        np.testing.assert_array_equal(
            result_low, result_high,
            err_msg=f"Low-level function {low_level_func_name} does not match high-level function {func_name}"
        )
        return

    @pytest.mark.parametrize("func_name", SCAN_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    def test_edge_cases(self, func_name, operation, dtype_all):
        """Test edge cases for cumulative sum."""
        func = getattr(py_gpu_algos, func_name)

        # Single element vector
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single, operation)
        numpy_result_single = get_numpy_reference(func_name, vec_single, operation)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # Vector with zeros
        vec_zeros = np.zeros(100).astype(dtype_all)
        result_zeros = func(vec_zeros, operation)
        numpy_result_zeros = get_numpy_reference(func_name, vec_zeros, operation)
        assert_array_close(result_zeros, numpy_result_zeros, dtype_all)

        # Vector with negative values (for signed types)
        if not np.issubdtype(dtype_all, np.unsignedinteger):
            vec_negative = np.array([-1, -2, -3, -4, -5]).astype(dtype_all)
            result_negative = func(vec_negative, operation)
            numpy_result_negative = get_numpy_reference(func_name, vec_negative, operation)
            assert_array_close(result_negative, numpy_result_negative, dtype_all)
            pass
        return



class TestVectorAggOp:
    """Test vector aggregation operations."""

    @pytest.mark.parametrize("func_name", AGGOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_incremental(self, func_name, dtype_all, test_input_vector_incremental, test_size):
        """Test aggregation operations with incremental input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, (1,), dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", AGGOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_random(self, func_name, dtype_all, test_input_vector_random, test_size):
        """Test aggregation operations with random input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_random(dtype_all, test_size)

        # Compute result
        result = func(vec)

        # Validate basic properties
        validate_basic_properties(result, (1,), dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", AGGOP_FUNCTIONS)
    @pytest.mark.parametrize("test_size", [1000])
    def test_low_level_functions(self, func_name, dtype_all, test_input_vector_incremental, test_size):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{dtype_name}"

        # Check that low-level function exists
        assert hasattr(py_gpu_algos, low_level_func_name), \
            f"Low-level function {low_level_func_name} not available"

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec)

        np.testing.assert_array_equal(
            result_low, result_high,
            err_msg=f"Low-level function {low_level_func_name} does not match high-level function {func_name}"
        )
        return

    @pytest.mark.parametrize("func_name", AGGOP_FUNCTIONS)
    def test_edge_cases(self, func_name, dtype_all):
        """Test edge cases for aggregation operations."""
        func = getattr(py_gpu_algos, func_name)

        # Single element vector
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single)
        numpy_result_single = get_numpy_reference(func_name, vec_single)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # Vector with zeros
        vec_zeros = np.zeros(100).astype(dtype_all)
        result_zeros = func(vec_zeros)
        numpy_result_zeros = get_numpy_reference(func_name, vec_zeros)
        assert_array_close(result_zeros, numpy_result_zeros, dtype_all)

        # Vector with negative values (for signed types)
        if not np.issubdtype(dtype_all, np.unsignedinteger):
            vec_negative = np.array([-1, -2, -3, -4, -5]).astype(dtype_all)
            result_negative = func(vec_negative)
            numpy_result_negative = get_numpy_reference(func_name, vec_negative)
            assert_array_close(result_negative, numpy_result_negative, dtype_all)
            pass
        return


class TestVectorReduction:
    """Test vector reduction operations."""

    @pytest.mark.parametrize("func_name", REDUCTION_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_incremental(self, func_name, operation, dtype_all, test_input_vector_incremental, test_size):
        """Test reduction operations with incremental input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compute result
        result = func(vec, operation)

        # Validate basic properties
        validate_basic_properties(result, (1,), dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec, operation)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", REDUCTION_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_random(self, func_name, operation, dtype_all, test_input_vector_random, test_size):
        """Test reduction operations with random input."""
        func = getattr(py_gpu_algos, func_name)
        vec = test_input_vector_random(dtype_all, test_size)

        # Compute result
        result = func(vec, operation)

        # Validate basic properties
        validate_basic_properties(result, (1,), dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference(func_name, vec, operation)
        assert_array_close(result, numpy_result, dtype_all)
        return

    @pytest.mark.parametrize("func_name", REDUCTION_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    @pytest.mark.parametrize("test_size", [1000])
    def test_low_level_functions(self, func_name, operation, dtype_all, test_input_vector_incremental, test_size):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"{func_name}_{operation}_{dtype_name}"

        # Check that low-level function exists
        assert hasattr(py_gpu_algos, low_level_func_name), \
            f"Low-level function {low_level_func_name} not available"

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = getattr(py_gpu_algos, func_name)

        vec = test_input_vector_incremental(dtype_all, test_size)

        # Compare low-level and high-level results
        result_low = low_level_func(vec)
        result_high = high_level_func(vec, operation)

        np.testing.assert_array_equal(
            result_low, result_high,
            err_msg=f"Low-level function {low_level_func_name} does not match high-level function {func_name}"
        )
        return

    @pytest.mark.parametrize("func_name", REDUCTION_FUNCTIONS)
    @pytest.mark.parametrize("operation", OPERATIONS)
    def test_edge_cases(self, func_name, operation, dtype_all):
        """Test edge cases for reduction operations."""
        func = getattr(py_gpu_algos, func_name)

        # Single element vector
        vec_single = np.array([42]).astype(dtype_all)
        result_single = func(vec_single, operation)
        numpy_result_single = get_numpy_reference(func_name, vec_single, operation)
        assert_array_close(result_single, numpy_result_single, dtype_all)

        # Vector with zeros
        vec_zeros = np.zeros(100).astype(dtype_all)
        result_zeros = func(vec_zeros, operation)
        numpy_result_zeros = get_numpy_reference(func_name, vec_zeros, operation)
        assert_array_close(result_zeros, numpy_result_zeros, dtype_all)

        # Vector with negative values (for signed types)
        if not np.issubdtype(dtype_all, np.unsignedinteger):
            vec_negative = np.array([-1, -2, -3, -4, -5]).astype(dtype_all)
            result_negative = func(vec_negative, operation)
            numpy_result_negative = get_numpy_reference(func_name, vec_negative, operation)
            assert_array_close(result_negative, numpy_result_negative, dtype_all)
            pass
        return
