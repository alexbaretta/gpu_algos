"""
Comprehensive error handling tests for py-gpu-algos.

Tests error conditions, edge cases, and invalid inputs across all modules:
- Matrix operations error handling
- Vector operations error handling
- GLM operations error handling
- Sort operations error handling
- Cross-module error consistency
- Memory safety and resource handling
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    ErrorCaseBuilder
)

class TestMatrixOperationsErrorHandling:
    """Test error handling for matrix operations."""

    def test_dimension_mismatch_errors(self):
        """Test dimension mismatch errors for matrix operations."""
        # Test all matrix product functions
        matrix_product_functions = [
            'matrix_product_naive', 'matrix_product_tiled', 'matrix_product_warp',
            'matrix_product_cublas', 'matrix_product_cutlass', 'matrix_product_tensor'
        ]

        for func_name in matrix_product_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            # Create matrices with incompatible dimensions
            a = np.random.randn(10, 8).astype(np.float32)
            b_wrong = np.random.randn(5, 12).astype(np.float32)  # 8 != 5

            with pytest.raises(ValueError, match="dimension|shape|incompatible"):
                func(a, b_wrong)

    def test_dtype_mismatch_errors(self):
        """Test dtype mismatch errors for matrix operations."""
        matrix_functions = [
            'matrix_product_naive', 'matrix_transpose_striped'
        ]

        for func_name in matrix_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            a_float32 = np.random.randn(10, 8).astype(np.float32)

            if 'product' in func_name:
                b_float64 = np.random.randn(8, 12).astype(np.float64)
                with pytest.raises(ValueError, match="dtype|type"):
                    func(a_float32, b_float64)
            else:
                # Transpose function - just test with wrong input type
                a_complex = np.random.randn(10, 8).astype(np.complex64)
                with pytest.raises((ValueError, TypeError)):
                    func(a_complex)

    def test_invalid_array_dimensions(self):
        """Test invalid array dimensions for matrix operations."""
        matrix_functions = [
            'matrix_product_naive', 'matrix_transpose_striped'
        ]

        for func_name in matrix_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            # 1D arrays
            arr_1d = np.random.randn(10).astype(np.float32)
            # 3D arrays
            arr_3d = np.random.randn(5, 6, 7).astype(np.float32)

            if 'product' in func_name:
                with pytest.raises(ValueError, match="2D|dimension"):
                    func(arr_1d, arr_1d)
                with pytest.raises(ValueError, match="2D|dimension"):
                    func(arr_3d, arr_3d)
            else:
                with pytest.raises(ValueError, match="2D|dimension"):
                    func(arr_1d)
                with pytest.raises(ValueError, match="2D|dimension"):
                    func(arr_3d)

class TestVectorOperationsErrorHandling:
    """Test error handling for vector operations."""

    def test_non_vector_inputs(self):
        """Test error handling for non-1D inputs."""
        vector_functions = [
            'vector_cumsum_serial', 'vector_cumsum_parallel',
            'vector_cummax_parallel'
        ]

        for func_name in vector_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            # 2D array
            arr_2d = np.random.randn(10, 5).astype(np.float32)
            # 3D array
            arr_3d = np.random.randn(5, 6, 7).astype(np.float32)
            # 0D array (scalar)
            arr_0d = np.array(42.0).astype(np.float32)

            with pytest.raises(ValueError, match="1D|vector|dimension"):
                func(arr_2d)
            with pytest.raises(ValueError, match="1D|vector|dimension"):
                func(arr_3d)
            with pytest.raises(ValueError, match="1D|vector|dimension"):
                func(arr_0d)

    def test_scan_invalid_operations(self):
        """Test invalid operation names for scan functions."""
        if not hasattr(py_gpu_algos, 'vector_scan_parallel'):
            pytest.skip("vector_scan_parallel not available")

        vec = np.random.randn(100).astype(np.float32)

        invalid_operations = [
            "invalid", "multiply", "divide", "mean", "median",
            "sum_squared", "", "SUM", "MAX"  # Case sensitive
        ]

        for invalid_op in invalid_operations:
            with pytest.raises(ValueError, match="operation|invalid"):
                py_gpu_algos.vector_scan_parallel(vec, invalid_op)

    def test_vector_dtype_errors(self):
        """Test unsupported dtypes for vector operations."""
        # Complex dtypes should fail
        vec_complex = np.random.randn(100).astype(np.complex64)

        vector_functions = ['vector_cumsum_serial', 'vector_cumsum_parallel']

        for func_name in vector_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            with pytest.raises((ValueError, TypeError)):
                func(vec_complex)

class TestGLMOperationsErrorHandling:
    """Test error handling for GLM operations."""

    def test_tensor_dimension_errors(self):
        """Test tensor dimension errors for GLM operations."""
        glm_functions = [
            'glm_predict_naive', 'glm_gradient_naive', 'glm_gradient_xyyhat'
        ]

        for func_name in glm_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            # Wrong number of dimensions
            X_2d = np.random.randn(10, 100).astype(np.float32)
            M_2d = np.random.randn(10, 3).astype(np.float32)
            Y_2d = np.random.randn(3, 100).astype(np.float32)

            X_4d = np.random.randn(10, 4, 100, 5).astype(np.float32)

            if 'predict' in func_name:
                with pytest.raises(ValueError, match="3D|dimension"):
                    func(X_2d, M_2d)
                with pytest.raises(ValueError, match="3D|dimension"):
                    func(X_4d, M_2d)
            else:  # gradient functions
                with pytest.raises(ValueError, match="3D|dimension"):
                    func(X_2d, Y_2d, M_2d)
                with pytest.raises(ValueError, match="3D|dimension"):
                    func(X_4d, Y_2d, M_2d)

    def test_glm_shape_compatibility_errors(self):
        """Test shape compatibility errors for GLM operations."""
        # Create tensors with incompatible shapes
        X = np.random.randn(10, 4, 100).astype(np.float32)

        # Wrong number of features
        M_wrong_features = np.random.randn(8, 3, 4).astype(np.float32)  # 8 != 10
        # Wrong number of tasks
        M_wrong_tasks = np.random.randn(10, 3, 5).astype(np.float32)     # 5 != 4

        glm_functions = ['glm_predict_naive', 'glm_gradient_naive']

        for func_name in glm_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            if 'predict' in func_name:
                with pytest.raises(ValueError, match="shape|dimension|compatibility"):
                    func(X, M_wrong_features)
                with pytest.raises(ValueError, match="shape|dimension|compatibility"):
                    func(X, M_wrong_tasks)
            else:  # gradient functions
                Y = np.random.randn(3, 4, 100).astype(np.float32)
                M = np.random.randn(10, 3, 4).astype(np.float32)

                with pytest.raises(ValueError, match="shape|dimension|compatibility"):
                    func(X, Y, M_wrong_features)

    def test_glm_unsupported_dtypes(self):
        """Test unsupported dtypes for GLM operations."""
        # GLM operations only support specific dtypes
        X_float16 = np.random.randn(5, 2, 50).astype(np.float16)
        M_float16 = np.random.randn(5, 3, 2).astype(np.float16)
        Y_float16 = np.random.randn(3, 2, 50).astype(np.float16)

        X_int8 = np.random.randint(-10, 10, (5, 2, 50), dtype=np.int8)
        M_int8 = np.random.randint(-10, 10, (5, 3, 2), dtype=np.int8)

        glm_functions = ['glm_predict_naive', 'glm_gradient_naive']

        for func_name in glm_functions:
            if not hasattr(py_gpu_algos, func_name):
                continue

            func = getattr(py_gpu_algos, func_name)

            if 'predict' in func_name:
                # float16 and int8 should fail
                with pytest.raises((ValueError, TypeError)):
                    func(X_float16, M_float16)
                with pytest.raises((ValueError, TypeError)):
                    func(X_int8, M_int8)
            else:  # gradient functions
                with pytest.raises((ValueError, TypeError)):
                    func(X_float16, Y_float16, M_float16)

class TestSortOperationsErrorHandling:
    """Test error handling for sort operations."""


    def test_sort_invalid_axes(self):
        """Test invalid axis names for sort operations."""
        if not hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            pytest.skip("tensor_sort_bitonic not available")

        func = py_gpu_algos.tensor_sort_bitonic
        tensor = np.random.randn(8, 4, 16).astype(np.float32)

        invalid_axes = [
            "x", "y", "z", "width", "height", "length",
            0, 1, 2,  # Numeric axes should fail
            "ROWS", "COLS", "DEPTH",  # Case sensitive
            "", None, "axis_0"
        ]

        for invalid_axis in invalid_axes:
            with pytest.raises(ValueError, match=r"sort_dim must be one of.*rows.*cols.*sheets"):
                func(tensor, invalid_axis)

    def test_sort_dimension_errors(self):
        """Test dimension errors for sort operations."""
        if not hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            pytest.skip("tensor_sort_bitonic not available")

        func = py_gpu_algos.tensor_sort_bitonic

        # Wrong number of dimensions
        tensor_1d = np.random.randn(16).astype(np.float32)
        tensor_2d = np.random.randn(8, 16).astype(np.float32)
        tensor_4d = np.random.randn(4, 4, 4, 4).astype(np.float32)

        with pytest.raises(ValueError, match="3D|dimension"):
            func(tensor_1d, "rows")
        with pytest.raises(ValueError, match="3D|dimension"):
            func(tensor_2d, "rows")
        with pytest.raises(ValueError, match="3D|dimension"):
            func(tensor_4d, "rows")

class TestCrossModuleErrorConsistency:
    """Test error consistency across different modules."""

    def test_dtype_error_consistency(self):
        """Test that dtype errors are consistent across modules."""
        # Test complex dtypes across different modules
        complex_array_1d = np.random.randn(100).astype(np.complex64)
        complex_array_2d = np.random.randn(10, 10).astype(np.complex64)
        complex_array_3d = np.random.randn(4, 4, 4).astype(np.complex64)

        # All modules should reject complex dtypes
        functions_to_test = [
            ('vector_cumsum_serial', (complex_array_1d,)),
            ('matrix_product_naive', (complex_array_2d, complex_array_2d)),
            ('matrix_transpose_striped', (complex_array_2d,)),
        ]

        if hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            functions_to_test.append(('tensor_sort_bitonic', (complex_array_3d, "rows")))

        for func_name, args in functions_to_test:
            if hasattr(py_gpu_algos, func_name):
                func = getattr(py_gpu_algos, func_name)
                with pytest.raises((ValueError, TypeError)):
                    func(*args)

    def test_empty_array_handling(self):
        """Test handling of empty arrays across modules."""
        empty_1d = np.array([]).astype(np.float32)
        empty_2d = np.array([]).reshape(0, 0).astype(np.float32)

        # Vector operations should handle empty arrays gracefully
        vector_functions = ['vector_cumsum_serial', 'vector_cumsum_parallel']

        for func_name in vector_functions:
            if hasattr(py_gpu_algos, func_name):
                func = getattr(py_gpu_algos, func_name)
                result = func(empty_1d)
                assert result.shape == (0,)
                assert result.dtype == np.float32

    def test_contiguity_handling(self):
        """Test that all functions handle non-contiguous arrays properly."""
        # Create non-contiguous arrays
        base_2d = np.random.randn(20, 20).astype(np.float32)
        non_contiguous_2d = base_2d[::2, ::2]  # Non-contiguous view

        base_1d = np.random.randn(200).astype(np.float32)
        non_contiguous_1d = base_1d[::2]  # Non-contiguous view

        # Functions should either work with non-contiguous arrays or make them contiguous
        functions_to_test = [
            ('vector_cumsum_serial', (non_contiguous_1d,)),
            ('matrix_transpose_striped', (non_contiguous_2d,)),
        ]

        # These should not raise contiguity errors
        for func_name, args in functions_to_test:
            if hasattr(py_gpu_algos, func_name):
                func = getattr(py_gpu_algos, func_name)
                try:
                    result = func(*args)
                    # Result should be contiguous
                    assert result.flags.c_contiguous or result.flags.f_contiguous
                except Exception as e:
                    # If it fails, it should not be due to contiguity
                    assert "contiguous" not in str(e).lower()

class TestMemorySafetyAndEdgeCases:
    """Test memory safety and edge cases."""

    def test_very_large_dimensions(self):
        """Test behavior with very large dimensions (but not too large to cause OOM)."""
        # Test with moderately large arrays that might stress the system
        large_size = 10000

        # Vector operations
        if hasattr(py_gpu_algos, 'vector_cumsum_serial'):
            large_vector = np.random.randn(large_size).astype(np.float32)
            result = py_gpu_algos.vector_cumsum_serial(large_vector)
            assert result.shape == (large_size,)
            assert result.dtype == np.float32

    def test_zero_sized_valid_dimensions(self):
        """Test operations with zero-sized but validly shaped arrays."""
        # Matrix with zero rows
        matrix_zero_rows = np.random.randn(0, 5).astype(np.float32)
        matrix_compatible = np.random.randn(5, 3).astype(np.float32)

        # Should either work or fail gracefully
        if hasattr(py_gpu_algos, 'matrix_product_naive'):
            try:
                result = py_gpu_algos.matrix_product_naive(matrix_zero_rows, matrix_compatible)
                assert result.shape == (0, 3)
            except ValueError:
                # Acceptable to reject zero-sized arrays
                pass

    def test_numerical_edge_values(self):
        """Test with numerical edge values (inf, nan, very large/small numbers)."""
        # Test with infinity and NaN values
        special_values = [np.inf, -np.inf, np.nan, 1e30, -1e30, 1e-30, -1e-30]

        for value in special_values:
            vec = np.full(100, value, dtype=np.float32)

            # Operations should not crash, but results may be inf/nan
            if hasattr(py_gpu_algos, 'vector_cumsum_serial'):
                try:
                    result = py_gpu_algos.vector_cumsum_serial(vec)
                    assert result.shape == vec.shape
                    assert result.dtype == vec.dtype
                except Exception:
                    # Some operations might reject special values
                    pass

    def test_input_modification_safety(self):
        """Test that input arrays are not modified by operations (except in-place ones)."""
        # Vector operations (should not modify input)
        vec = np.random.randn(100).astype(np.float32)
        vec_original = vec.copy()

        if hasattr(py_gpu_algos, 'vector_cumsum_serial'):
            py_gpu_algos.vector_cumsum_serial(vec)
            np.testing.assert_array_equal(vec, vec_original)

        # Matrix operations (should not modify input)
        matrix_a = np.random.randn(10, 8).astype(np.float32)
        matrix_b = np.random.randn(8, 12).astype(np.float32)
        matrix_a_original = matrix_a.copy()
        matrix_b_original = matrix_b.copy()

        if hasattr(py_gpu_algos, 'matrix_product_naive'):
            py_gpu_algos.matrix_product_naive(matrix_a, matrix_b)
            np.testing.assert_array_equal(matrix_a, matrix_a_original)
            np.testing.assert_array_equal(matrix_b, matrix_b_original)

        # Sort operations ARE in-place, so input should be modified
        if hasattr(py_gpu_algos, 'tensor_sort_bitonic'):
            tensor = np.random.randn(8, 4, 16).astype(np.float32)
            tensor_original = tensor.copy()

            py_gpu_algos.tensor_sort_bitonic(tensor, "rows")
            # Tensor should be modified (sorted)
            assert not np.array_equal(tensor, tensor_original)

class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_multiple_operations_sequence(self):
        """Test that multiple operations in sequence don't cause resource leaks."""
        # Run many operations in sequence to test resource management
        for i in range(10):
            if hasattr(py_gpu_algos, 'vector_cumsum_serial'):
                vec = np.random.randn(1000).astype(np.float32)
                result = py_gpu_algos.vector_cumsum_serial(vec)
                assert result.shape == vec.shape

            if hasattr(py_gpu_algos, 'matrix_product_naive'):
                a = np.random.randn(50, 40).astype(np.float32)
                b = np.random.randn(40, 60).astype(np.float32)
                result = py_gpu_algos.matrix_product_naive(a, b)
                assert result.shape == (50, 60)

    def test_operation_with_exception_cleanup(self):
        """Test that resources are cleaned up properly when operations raise exceptions."""
        # Try operations that should fail and ensure no resource leaks
        if hasattr(py_gpu_algos, 'matrix_product_naive'):
            a = np.random.randn(10, 8).astype(np.float32)
            b_wrong = np.random.randn(5, 12).astype(np.float32)

            for _ in range(5):
                with pytest.raises(ValueError):
                    py_gpu_algos.matrix_product_naive(a, b_wrong)

                # Immediately try a valid operation to ensure resources are still available
                b_correct = np.random.randn(8, 12).astype(np.float32)
                result = py_gpu_algos.matrix_product_naive(a, b_correct)
                assert result.shape == (10, 12)
