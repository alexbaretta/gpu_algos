"""
Tests for sort operations in py-gpu-algos.

Tests the single sort kernel:
- tensor_sort_bitonic: In-place bitonic sort for 3D tensors

This operation requires power-of-2 dimensions and operates in-place.
Supports all 11 dtypes and sorts along specified axis ("rows", "cols", "depth").
Tensor shape: (dim0, dim1, dim2) where all dimensions must be powers of 2.
"""

import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_cases,
    compare_with_numpy_reference, get_numpy_reference_tensor_sort,
    ErrorCaseBuilder, print_performance_summary, is_power_of_2,
    generate_power_of_2_shape
)

# Sort function to test
SORT_FUNCTIONS = ['tensor_sort_bitonic']

# Sort axis options
SORT_AXES = ["rows", "cols", "depth"]

class TestTensorSortBitonic:
    """Test 3D tensor bitonic sort operations."""

    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_basic_functionality(self, axis, dtype_all, power_of_2_tensors):
        """Test basic bitonic sort functionality."""
        func = py_gpu_algos.tensor_sort_bitonic
        tensor = power_of_2_tensors(dtype_all)

        # Make a copy since sort is in-place
        tensor_copy = tensor.copy()

        # Compute result (in-place operation)
        func(tensor_copy, axis)

        # Validate basic properties
        validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

        # Compare with NumPy reference
        numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
        assert_array_close(tensor_copy, numpy_result, dtype_all)

    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_low_level_functions(self, axis, dtype_all, power_of_2_tensors):
        """Test low-level type-specific functions."""
        dtype_name = dtype_all.__name__
        low_level_func_name = f"tensor_sort_bitonic_{dtype_name}"

        # Check if low-level function exists
        if not hasattr(py_gpu_algos, low_level_func_name):
            pytest.skip(f"Low-level function {low_level_func_name} not available")

        low_level_func = getattr(py_gpu_algos, low_level_func_name)
        high_level_func = py_gpu_algos.tensor_sort_bitonic

        tensor = power_of_2_tensors(dtype_all)

        # Test both functions (make copies since operation is in-place)
        tensor_low = tensor.copy()
        tensor_high = tensor.copy()

        low_level_func(tensor_low, axis)
        high_level_func(tensor_high, axis)

        assert_array_close(tensor_low, tensor_high, dtype_all)

    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_different_shapes(self, axis, dtype_all):
        """Test bitonic sort with different power-of-2 shapes."""
        func = py_gpu_algos.tensor_sort_bitonic

        # Power-of-2 shapes to test
        test_shapes = [
            (2, 2, 2),      # Minimal
            (4, 4, 4),      # Small
            (8, 4, 16),     # Medium rectangular
            (16, 8, 8),     # Medium square-ish
            (32, 16, 4),    # Larger
        ]

        for shape in test_shapes:
            np.random.seed(42)

            # Create test tensor
            if np.issubdtype(dtype_all, np.integer):
                if dtype_all in [np.int8, np.uint8]:
                    tensor = np.random.randint(-10, 10, shape, dtype=dtype_all)
                elif dtype_all in [np.int16, np.uint16]:
                    tensor = np.random.randint(-100, 100, shape, dtype=dtype_all)
                else:
                    tensor = np.random.randint(-1000, 1000, shape, dtype=dtype_all)
            else:
                tensor = (np.random.randn(*shape) * 100).astype(dtype_all)

            # Test sorting
            tensor_copy = tensor.copy()
            func(tensor_copy, axis)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
            validate_basic_properties(tensor_copy, shape, dtype_all)
            assert_array_close(tensor_copy, numpy_result, dtype_all)

    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_already_sorted_arrays(self, axis, dtype_all):
        """Test sorting arrays that are already sorted."""
        func = py_gpu_algos.tensor_sort_bitonic

        # Create sorted arrays
        shape = (8, 4, 16)

        if np.issubdtype(dtype_all, np.integer):
            if dtype_all in [np.int8, np.uint8]:
                base_values = np.arange(-10, 10, dtype=dtype_all)
            elif dtype_all in [np.int16, np.uint16]:
                base_values = np.arange(-50, 50, dtype=dtype_all)
            else:
                base_values = np.arange(-100, 100, dtype=dtype_all)
        else:
            base_values = np.linspace(-10, 10, 100).astype(dtype_all)

        # Create tensor with sorted slices along the specified axis
        tensor = np.random.choice(base_values, shape).astype(dtype_all)

        # Sort with NumPy first to get a properly sorted tensor
        if axis == "rows":
            tensor = np.sort(tensor, axis=0)
        elif axis == "cols":
            tensor = np.sort(tensor, axis=1)
        else:  # depth
            tensor = np.sort(tensor, axis=2)

        # Now test our sort function
        tensor_copy = tensor.copy()
        func(tensor_copy, axis)

        # Result should be identical to input (already sorted)
        assert_array_close(tensor_copy, tensor, dtype_all)

    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_reverse_sorted_arrays(self, axis, dtype_all):
        """Test sorting arrays that are reverse sorted."""
        func = py_gpu_algos.tensor_sort_bitonic

        shape = (8, 4, 16)

        if np.issubdtype(dtype_all, np.integer):
            if dtype_all in [np.int8, np.uint8]:
                base_values = np.arange(-10, 10, dtype=dtype_all)
            elif dtype_all in [np.int16, np.uint16]:
                base_values = np.arange(-50, 50, dtype=dtype_all)
            else:
                base_values = np.arange(-100, 100, dtype=dtype_all)
        else:
            base_values = np.linspace(-10, 10, 100).astype(dtype_all)

        # Create tensor with reverse sorted slices
        tensor = np.random.choice(base_values, shape).astype(dtype_all)

        # Sort in reverse order first
        if axis == "rows":
            tensor = -np.sort(-tensor, axis=0)  # Reverse sort
        elif axis == "cols":
            tensor = -np.sort(-tensor, axis=1)
        else:  # depth
            tensor = -np.sort(-tensor, axis=2)

        # Now test our sort function (should sort in ascending order)
        tensor_copy = tensor.copy()
        func(tensor_copy, axis)

        # Compare with NumPy reference (ascending sort)
        numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
        assert_array_close(tensor_copy, numpy_result, dtype_all)

    def test_all_same_values(self, dtype_all):
        """Test sorting when all values are the same."""
        func = py_gpu_algos.tensor_sort_bitonic

        shape = (8, 4, 16)

        # Create tensor with all same values
        if np.issubdtype(dtype_all, np.integer):
            value = 42
        else:
            value = 3.14159

        tensor = np.full(shape, value, dtype=dtype_all)

        # Sort along each axis
        for axis in SORT_AXES:
            tensor_copy = tensor.copy()
            func(tensor_copy, axis)

            # Result should be identical to input
            assert_array_close(tensor_copy, tensor, dtype_all)

    def test_mathematical_properties(self, dtype_all):
        """Test mathematical properties of sorting."""
        func = py_gpu_algos.tensor_sort_bitonic

        shape = (8, 4, 16)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            if dtype_all in [np.int8, np.uint8]:
                tensor = np.random.randint(-20, 20, shape, dtype=dtype_all)
            else:
                tensor = np.random.randint(-100, 100, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 50).astype(dtype_all)

        for axis in SORT_AXES:
            tensor_copy = tensor.copy()
            func(tensor_copy, axis)

            # Check that values are sorted along the specified axis
            if axis == "rows":
                axis_num = 0
            elif axis == "cols":
                axis_num = 1
            else:  # depth
                axis_num = 2

            # For each slice along the other axes, values should be sorted
            for i in range(shape[(axis_num + 1) % 3]):
                for j in range(shape[(axis_num + 2) % 3]):
                    if axis_num == 0:
                        slice_vals = tensor_copy[:, i, j]
                    elif axis_num == 1:
                        slice_vals = tensor_copy[i, :, j]
                    else:  # axis_num == 2
                        slice_vals = tensor_copy[i, j, :]

                    # Check that slice is sorted
                    sorted_slice = np.sort(slice_vals)
                    assert_array_close(slice_vals, sorted_slice, dtype_all)

            # Check that all original values are preserved (permutation property)
            assert np.sort(tensor.flatten()).tolist() == np.sort(tensor_copy.flatten()).tolist()

    def test_error_handling_power_of_2(self, dtype_all):
        """Test error handling for non-power-of-2 dimensions."""
        func = py_gpu_algos.tensor_sort_bitonic

        # Non-power-of-2 shapes
        invalid_shapes = [
            (7, 4, 8),      # 7 is not power of 2
            (8, 6, 4),      # 6 is not power of 2
            (4, 4, 12),     # 12 is not power of 2
            (3, 5, 7),      # None are powers of 2
        ]

        for shape in invalid_shapes:
            tensor = np.random.randn(*shape).astype(dtype_all)

            with pytest.raises(ValueError, match="power of 2|bitonic"):
                func(tensor, "rows")

    def test_error_handling_general(self, dtype_all):
        """Test general error handling."""
        func = py_gpu_algos.tensor_sort_bitonic

        # Valid power-of-2 tensor for other tests
        valid_tensor = np.random.randn(8, 4, 16).astype(dtype_all)

        # Wrong number of dimensions
        tensor_1d = np.random.randn(16).astype(dtype_all)
        tensor_2d = np.random.randn(8, 16).astype(dtype_all)
        tensor_4d = np.random.randn(4, 4, 4, 4).astype(dtype_all)

        # Invalid axis names
        invalid_axes = ["x", "y", "z", "width", "height", "invalid", 0, 1, 2]

        # Build error test cases
        error_cases = []

        # Dimension errors
        error_cases.extend([
            ((tensor_1d, "rows"), {}, ValueError, "1D tensor"),
            ((tensor_2d, "rows"), {}, ValueError, "2D tensor"),
            ((tensor_4d, "rows"), {}, ValueError, "4D tensor"),
        ])

        # Invalid axis errors
        for invalid_axis in invalid_axes:
            error_cases.append((
                (valid_tensor, invalid_axis), {}, ValueError, f"invalid axis {invalid_axis}"
            ))

        validate_function_error_cases(func, error_cases)

class TestTensorSortIntegration:
    """Integration tests for tensor sort operations."""

    def test_sort_along_different_axes_consistency(self, dtype_all):
        """Test that sorting along different axes produces correct results."""
        shape = (8, 4, 16)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            tensor = np.random.randint(-50, 50, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 30).astype(dtype_all)

        # Sort along each axis and verify with NumPy
        for axis in SORT_AXES:
            tensor_copy = tensor.copy()
            py_gpu_algos.tensor_sort_bitonic(tensor_copy, axis)

            numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)

    def test_multiple_sorts_idempotent(self, dtype_all):
        """Test that sorting multiple times gives the same result (idempotent)."""
        shape = (8, 4, 16)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            tensor = np.random.randint(-30, 30, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 20).astype(dtype_all)

        for axis in SORT_AXES:
            tensor_copy = tensor.copy()

            # Sort once
            py_gpu_algos.tensor_sort_bitonic(tensor_copy, axis)
            first_sort = tensor_copy.copy()

            # Sort again
            py_gpu_algos.tensor_sort_bitonic(tensor_copy, axis)
            second_sort = tensor_copy.copy()

            # Results should be identical
            assert_array_close(first_sort, second_sort, dtype_all)

    def test_sorting_different_axes_sequence(self, dtype_all):
        """Test sorting along different axes in sequence."""
        shape = (8, 4, 16)
        np.random.seed(42)

        if np.issubdtype(dtype_all, np.integer):
            tensor = np.random.randint(-40, 40, shape, dtype=dtype_all)
        else:
            tensor = (np.random.randn(*shape) * 25).astype(dtype_all)

        # Sort along rows, then cols, then depth
        tensor_copy = tensor.copy()

        py_gpu_algos.tensor_sort_bitonic(tensor_copy, "rows")
        py_gpu_algos.tensor_sort_bitonic(tensor_copy, "cols")
        py_gpu_algos.tensor_sort_bitonic(tensor_copy, "depth")

        # Result should be sorted along depth (last sort)
        numpy_result = get_numpy_reference_tensor_sort(tensor_copy, "depth")
        # Note: We compare against sorting the already twice-sorted tensor
        # because each sort operation changes the tensor

        # Just verify that final result is properly sorted along depth axis
        for i in range(shape[0]):
            for j in range(shape[1]):
                depth_slice = tensor_copy[i, j, :]
                sorted_slice = np.sort(depth_slice)
                assert_array_close(depth_slice, sorted_slice, dtype_all)

class TestTensorSortPerformance:
    """Performance tests for tensor sort operations."""

    @pytest.mark.performance
    @pytest.mark.parametrize("axis", SORT_AXES)
    def test_sort_performance(self, axis, performance_sizes):
        """Test sort performance compared to NumPy."""
        func = py_gpu_algos.tensor_sort_bitonic
        dtype = np.float32

        for base_size, _, _ in performance_sizes:
            # Generate power-of-2 dimensions for bitonic sort
            if base_size <= 64:
                shape = (base_size, base_size // 2, base_size)
            elif base_size <= 128:
                shape = (64, 32, 64)
            elif base_size <= 256:
                shape = (128, 64, 64)
            else:
                shape = (256, 64, 32)

            # Ensure all dimensions are powers of 2
            shape = tuple(2**int(np.log2(max(2, dim))) for dim in shape)

            np.random.seed(42)
            tensor = (np.random.randn(*shape) * 100).astype(dtype)

            # Create wrapper that makes copy for fair comparison
            def gpu_sort_wrapper(t):
                t_copy = t.copy()
                func(t_copy, axis)
                return t_copy

            def numpy_sort_wrapper(t):
                return get_numpy_reference_tensor_sort(t, axis)

            results = compare_with_numpy_reference(
                gpu_sort_wrapper, numpy_sort_wrapper,
                (tensor,), dtype
            )

            assert results['accuracy_pass'], f"Accuracy failed for sort along {axis}"

            if shape[0] >= 64:
                print_performance_summary(results,
                    f"tensor_sort_bitonic_{axis} ({shape[0]}x{shape[1]}x{shape[2]})")

class TestTensorSortEdgeCases:
    """Test edge cases for tensor sort operations."""

    def test_minimal_tensor(self, dtype_all):
        """Test sorting minimal 2x2x2 tensor."""
        func = py_gpu_algos.tensor_sort_bitonic

        # Create minimal tensor
        if np.issubdtype(dtype_all, np.integer):
            tensor = np.array([[[4, 1], [3, 2]], [[8, 5], [7, 6]]]).astype(dtype_all)
        else:
            tensor = np.array([[[4.0, 1.0], [3.0, 2.0]], [[8.0, 5.0], [7.0, 6.0]]]).astype(dtype_all)

        for axis in SORT_AXES:
            tensor_copy = tensor.copy()
            func(tensor_copy, axis)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)

    def test_single_element_slices(self, dtype_all):
        """Test sorting when some dimensions are 1 (but still power of 2)."""
        func = py_gpu_algos.tensor_sort_bitonic

        # 1 is a power of 2 (2^0)
        shapes_with_ones = [
            (1, 4, 8),
            (4, 1, 8),
            (4, 8, 1),
            (1, 1, 8),
            (1, 8, 1),
            (8, 1, 1),
        ]

        for shape in shapes_with_ones:
            if np.issubdtype(dtype_all, np.integer):
                tensor = np.random.randint(-10, 10, shape, dtype=dtype_all)
            else:
                tensor = (np.random.randn(*shape) * 10).astype(dtype_all)

            for axis in SORT_AXES:
                tensor_copy = tensor.copy()
                func(tensor_copy, axis)

                # Compare with NumPy reference
                numpy_result = get_numpy_reference_tensor_sort(tensor, axis)
                assert_array_close(tensor_copy, numpy_result, dtype_all)

    def test_extreme_values(self, dtype_all):
        """Test sorting with extreme values for each dtype."""
        func = py_gpu_algos.tensor_sort_bitonic
        shape = (4, 4, 8)

        if np.issubdtype(dtype_all, np.integer):
            # Use full range of integer type
            info = np.iinfo(dtype_all)
            # Use a subset to avoid overflow in calculations
            min_val = max(info.min, -1000)
            max_val = min(info.max, 1000)
            tensor = np.random.randint(min_val, max_val, shape, dtype=dtype_all)
        else:
            # Use large floating point values
            if dtype_all == np.float16:
                tensor = (np.random.randn(*shape) * 100).astype(dtype_all)
            else:
                tensor = (np.random.randn(*shape) * 1e6).astype(dtype_all)

        for axis in SORT_AXES:
            tensor_copy = tensor.copy()
            func(tensor_copy, axis)

            # Just verify sorting property (NumPy comparison may have precision issues)
            if axis == "rows":
                axis_num = 0
            elif axis == "cols":
                axis_num = 1
            else:
                axis_num = 2

            # Check that each slice is sorted
            for i in range(shape[(axis_num + 1) % 3]):
                for j in range(shape[(axis_num + 2) % 3]):
                    if axis_num == 0:
                        slice_vals = tensor_copy[:, i, j]
                    elif axis_num == 1:
                        slice_vals = tensor_copy[i, :, j]
                    else:
                        slice_vals = tensor_copy[i, j, :]

                    # Verify slice is sorted
                    assert np.all(slice_vals[:-1] <= slice_vals[1:]), f"Slice not sorted for {dtype_all}"
