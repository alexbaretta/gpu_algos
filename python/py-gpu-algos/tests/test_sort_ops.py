"""
Tests for sort operations in py-gpu-algos.

Tests the single sort kernel:
- tensor_sort_bitonic: In-place bitonic sort for 3D tensors

This operation requires power-of-2 dimensions and operates in-place.
Supports all 11 dtypes and sorts along specified sort_axis ("rows", "cols", "sheets").
Tensor shape: (dim0, dim1, dim2) where all dimensions must be powers of 2.
"""

import itertools
import pytest
import numpy as np
import py_gpu_algos
from .utils import (
    assert_array_close, validate_basic_properties, validate_function_error_case,
    validate_function_error_cases, compare_with_numpy_reference, get_numpy_reference_tensor_sort,
    ErrorCaseBuilder, print_performance_summary, is_power_of_2,
)
from .conftest import TEST_PROBLEM_SIZES

P2_SORT_FUNCTIONS = ['tensor_sort_bitonic']
SORT_AXES = ["cols", "rows", "sheets"]

def pick_problem_p2shapes(sort_axis:str,i_start:int, stride:int) -> tuple[int, int, int]:
    """Pick problem sizes from TEST_SIZES"""
    l_standard = len(TEST_PROBLEM_SIZES)
    l_p2 = len(TEST_PROBLEM_P2SIZES)
    i_col = i_start % l_standard if sort_axis != 'cols' else i_start % l_p2
    i_row = (i_col + stride) % l_standard if sort_axis != 'rows' else (i_col + stride) % l_p2
    i_sheet = (i_row + stride) % l_standard if sort_axis != 'sheets' else (i_row + stride) % l_p2
    return (
        TEST_PROBLEM_P2SIZES[i_sheet] if sort_axis == 'sheets' else TEST_PROBLEM_SIZES[i_sheet],
        TEST_PROBLEM_P2SIZES[i_row] if sort_axis == 'rows' else TEST_PROBLEM_SIZES[i_row],
        TEST_PROBLEM_P2SIZES[i_col] if sort_axis == 'cols' else TEST_PROBLEM_SIZES[i_col],
    )

TEST_PROBLEM_P2SIZES = [ 2**power for power in [ 3,5,7,9 ] ]

def make_test_p2_tensor_sort(
        sort_axis:str,
    ):

    p2_shapes = [(4,4,4)] + sorted(list(
        [pick_problem_p2shapes(sort_axis, i, 1) for i in range(len(TEST_PROBLEM_P2SIZES))]
        + [pick_problem_p2shapes(sort_axis, i, 2) for i in range(len(TEST_PROBLEM_P2SIZES))]
        + [pick_problem_p2shapes(sort_axis, i, 3) for i in range(len(TEST_PROBLEM_P2SIZES))]
    ))

    class MetaTestP2TensorSort:
        """Test 3D tensor bitonic sort operations."""

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_incremental(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_incremental):
            """Test bitonic sort with incremental input."""
            func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            tensor = test_input_tensor_3d_incremental(dtype_all, *p2_shape)

            # Make a copy since sort is in-place
            tensor_copy = tensor.copy()

            # Compute result (in-place operation)
            func(tensor_copy, sort_axis)

            # Validate basic properties
            validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, sort_axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_random(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_random):
            """Test bitonic sort with random input."""
            func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            tensor = test_input_tensor_3d_random(dtype_all, *p2_shape)

            # Make a copy since sort is in-place
            tensor_copy = tensor.copy()

            # Compute result (in-place operation)
            func(tensor_copy, sort_axis)

            # Validate basic properties
            validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, sort_axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_presorted(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_random):
            """Test bitonic sort with random input."""
            func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            random_tensor = test_input_tensor_3d_random(dtype_all, *p2_shape)
            tensor = get_numpy_reference_tensor_sort(random_tensor, sort_axis)

            # Make a copy since sort is in-place
            tensor_copy = tensor.copy()

            # Compute result (in-place operation)
            func(tensor_copy, sort_axis)

            # Validate basic properties
            validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, sort_axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_revsorted(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_random):
            """Test bitonic sort with random input."""
            func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            random_tensor = test_input_tensor_3d_random(dtype_all, *p2_shape)
            if sort_axis == "cols":
                tensor = get_numpy_reference_tensor_sort(random_tensor, sort_axis)[:,:,::-1]
            elif sort_axis == "rows":
                tensor = get_numpy_reference_tensor_sort(random_tensor, sort_axis)[:,::-1,:]
            else:  # sheets
                tensor = get_numpy_reference_tensor_sort(random_tensor, sort_axis)[::-1,:,:]
                pass

            # Make a copy since sort is in-place
            tensor_copy = tensor.copy()

            # Compute result (in-place operation)
            func(tensor_copy, sort_axis)

            # Validate basic properties
            validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, sort_axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_constant(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_constant):
            """Test bitonic sort with random input."""
            func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            tensor = test_input_tensor_3d_constant(dtype_all, *p2_shape, value=42)

            # Make a copy since sort is in-place
            tensor_copy = tensor.copy()

            # Compute result (in-place operation)
            func(tensor_copy, sort_axis)

            # Validate basic properties
            validate_basic_properties(tensor_copy, tensor.shape, dtype_all)

            # Compare with NumPy reference
            numpy_result = get_numpy_reference_tensor_sort(tensor, sort_axis)
            assert_array_close(tensor_copy, numpy_result, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("p2_shape", p2_shapes)
        def test_low_level_functions(self, func_name, p2_shape, dtype_all, test_input_tensor_3d_incremental):
            """Test low-level type-specific functions."""
            dtype_name = dtype_all.__name__
            low_level_func_name = f"{func_name}_{dtype_name}"

            # Check if low-level function exists
            if not hasattr(py_gpu_algos, low_level_func_name):
                pytest.skip(f"Low-level function {low_level_func_name} not available")

            low_level_func = getattr(py_gpu_algos, low_level_func_name)
            high_level_func = getattr(py_gpu_algos, func_name)

            # Generate power-of-2 shape for bitonic sort
            tensor = test_input_tensor_3d_incremental(dtype_all, *p2_shape)

            # Test both functions (make copies since operation is in-place)
            tensor_low = tensor.copy()
            tensor_high = tensor.copy()

            low_level_func(tensor_low, sort_axis)
            high_level_func(tensor_high, sort_axis)

            assert_array_close(tensor_low, tensor_high, dtype_all)
            return

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("wrong_axis", [axis for axis in SORT_AXES if axis != sort_axis])
        def test_error_handling_wrong_axis(self, func_name, dtype_all, wrong_axis, test_input_tensor_3d_random):
            """Test error handling for invalid inputs: wrong sort axis"""
            func = getattr(py_gpu_algos, func_name)
            shape = pick_problem_p2shapes(wrong_axis, 0, 1)
            error_case = ((test_input_tensor_3d_random(dtype_all, *shape), sort_axis), {}, ValueError, "power of 2|bitonic")
            validate_function_error_case(func, error_case)

        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("invalid_axis", ["x", "y", "z", "width", "height", "invalid", 0, 1, 2])
        def test_error_handling_invalid_axis(self, func_name, dtype_all, invalid_axis, test_input_tensor_3d_random):
            """Test error handling for invalid inputs: invalid axis"""
            func = getattr(py_gpu_algos, func_name)
            shape = pick_problem_p2shapes(invalid_axis, 0, 1)
            error_case = ((test_input_tensor_3d_random(dtype_all, *shape), sort_axis), {}, ValueError, f"invalid sort_axis {invalid_axis}")
            validate_function_error_case(func, error_case)


        @pytest.mark.parametrize("func_name", P2_SORT_FUNCTIONS)
        @pytest.mark.parametrize("wrong_axis", [axis for axis in SORT_AXES if axis != sort_axis])
        def test_error_handling_bad_tensor3d(self, func_name, dtype_all, wrong_axis, test_input_tensor_3d_random):
            """Test error handling for invalid inputs: invalid tensor3d"""
            func = getattr(py_gpu_algos, func_name)

            # Wrong number of dimensions
            tensor_1d = np.zeros(16, dtype=dtype_all)
            tensor_2d = np.zeros((8, 16), dtype=dtype_all)
            tensor_4d = np.zeros((4, 4, 4, 4), dtype=dtype_all)

            # Build error test cases
            error_cases = []

            # Dimension errors
            error_cases.extend([
                ((tensor_1d, "rows"), {}, ValueError, "1D tensor"),
                ((tensor_2d, "rows"), {}, ValueError, "2D tensor"),
                ((tensor_4d, "rows"), {}, ValueError, "4D tensor"),
            ])

            validate_function_error_cases(func, error_cases)
            return
        pass
    return MetaTestP2TensorSort

TestP2TensorSortCols = make_test_p2_tensor_sort(sort_axis="cols")
TestP2TensorSortRows = make_test_p2_tensor_sort(sort_axis="rows")
TestP2TensorSortSheets = make_test_p2_tensor_sort(sort_axis="sheets")
