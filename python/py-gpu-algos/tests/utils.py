"""
Test utilities for py-gpu-algos test suite.

Common functions for testing, validation, and benchmarking.
"""

import numpy as np
import time
import sys
from typing import Callable, Dict, Any, Tuple

def assert_array_close(actual, expected, dtype, tol_bits=6):
    """
    Assert that two arrays are close with appropriate tolerances for the dtype.
    """
    assert actual.shape == expected.shape, f"Shapes do not match: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"Dtypes do not match: {actual.dtype} != {expected.dtype}"
    if np.issubdtype(dtype, np.integer):
        # For integers, check exact equality
        success = np.array_equal(actual, expected)
        if not success:
            assert False, f"Integer arrays not exactly equal for dtype {dtype}"
    else:
        if dtype == np.float16:
            significand_bits = 11
        elif dtype == np.float32:
            significand_bits = 24
        elif dtype == np.float64:
            significand_bits = 53
        else:
            assert False, f"This should not be possible: {dtype}"
            pass
        # For floats, check within tolerance
        rtol = 0.5**(significand_bits - tol_bits)
        success = np.allclose(actual, expected, rtol=rtol)
        if not success:
            assert False, f"Float arrays not close for dtype {dtype}"
            # np.testing.assert_allclose(actual, expected, rtol=rtol,
            #                          err_msg=f"Float arrays not close for dtype {dtype}")

def validate_basic_properties(result, expected_shape, expected_dtype):
    """
    Validate basic properties of a result array.
    """
    assert isinstance(result, np.ndarray), f"Result should be numpy array, got {type(result)}"
    assert result.shape == expected_shape, f"Wrong shape: expected {expected_shape}, got {result.shape}"
    assert result.dtype == expected_dtype, f"Wrong dtype: expected {expected_dtype}, got {result.dtype}"

def validate_function_error_cases(func: Callable, test_cases: list):
    """
    Test that a function properly raises errors for invalid inputs.

    Args:
        func: The function to test
        test_cases: List of (args, kwargs, expected_exception_type, description)
    """
    for args, kwargs, expected_exception, description in test_cases:
        try:
            result = func(*args, **kwargs)
            raise AssertionError(f"Expected {expected_exception.__name__} for {description}, but function succeeded")
        except expected_exception:
            pass  # Expected behavior
        except Exception as e:
            raise AssertionError(f"Expected {expected_exception.__name__} for {description}, got {type(e).__name__}: {e}")

def benchmark_function(func: Callable, args: tuple, kwargs: dict = None, warmup_runs: int = 3, timing_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple timing runs.

    Returns:
        Dictionary with timing statistics
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)

    # Timing runs
    times = []
    for _ in range(timing_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result': result
    }

def compare_with_numpy_reference(gpu_func: Callable, numpy_func: Callable,
                                args: tuple, dtype, **comparison_kwargs) -> Dict[str, Any]:
    """
    Compare GPU function result with NumPy reference implementation.

    Returns:
        Dictionary with comparison results and timing info
    """
    # Run GPU function
    gpu_stats = benchmark_function(gpu_func, args)
    gpu_result = gpu_stats['result']

    # Run NumPy reference
    numpy_stats = benchmark_function(numpy_func, args)
    numpy_result = numpy_stats['result']

    # Compare results
    try:
        assert_array_close(gpu_result, numpy_result, dtype, **comparison_kwargs)
        accuracy_pass = True
        max_error = np.max(np.abs(gpu_result.astype(np.float64) - numpy_result.astype(np.float64)))
    except AssertionError as e:
        accuracy_pass = False
        max_error = np.max(np.abs(gpu_result.astype(np.float64) - numpy_result.astype(np.float64)))

    # Calculate speedup
    speedup = numpy_stats['mean_time'] / gpu_stats['mean_time'] if gpu_stats['mean_time'] > 0 else float('inf')

    return {
        'accuracy_pass': accuracy_pass,
        'max_error': max_error,
        'gpu_time': gpu_stats['mean_time'],
        'numpy_time': numpy_stats['mean_time'],
        'speedup': speedup,
        'gpu_result': gpu_result,
        'numpy_result': numpy_result
    }

def get_numpy_reference_cumsum_serial(vec):
    """NumPy reference for serial cumulative sum."""
    return np.cumsum(vec, dtype=vec.dtype)

def get_numpy_reference_cumsum_parallel(vec):
    """NumPy reference for parallel cumulative sum."""
    return np.cumsum(vec, dtype=vec.dtype)

def get_numpy_reference_cummax_parallel(vec):
    """NumPy reference for parallel cumulative maximum."""
    return np.maximum.accumulate(vec, dtype=vec.dtype)

def get_numpy_reference_scan_parallel(vec, operation):
    """NumPy reference for parallel scan operations."""
    if operation == "sum":
        return np.cumsum(vec, dtype=vec.dtype)
    elif operation == "max":
        return np.maximum.accumulate(vec, dtype=vec.dtype)
    elif operation == "min":
        return np.minimum.accumulate(vec, dtype=vec.dtype)
    elif operation == "prod":
        return np.cumprod(vec, dtype=vec.dtype)
    else:
        raise ValueError(f"Unknown scan operation: {operation}")

def get_numpy_reference_matrix_product(a, b):
    """NumPy reference for matrix multiplication."""
    return np.dot(a, b)

def get_numpy_reference_matrix_transpose(a):
    """NumPy reference for matrix transpose."""
    return a.T

def get_numpy_reference_glm_predict(X, M):
    """
    NumPy reference for GLM prediction.

    X: (nfeatures, ntasks, nobs)
    M: (nfeatures, ntargets, ntasks)
    Returns: (ntargets, ntasks, nobs)
    """
    nfeatures, ntasks, nobs = X.shape
    nfeatures2, ntargets, ntasks2 = M.shape
    assert nfeatures == nfeatures2 and ntasks == ntasks2

    # Result shape: (ntargets, ntasks, nobs)
    result = np.zeros((ntargets, ntasks, nobs), dtype=X.dtype)

    for task in range(ntasks):
        # X[:, task, :] is (nfeatures, nobs)
        # M[:, :, task] is (nfeatures, ntargets)
        # Result should be (ntargets, nobs)
        result[:, task, :] = M[:, :, task].T @ X[:, task, :]

    return result

def get_numpy_reference_glm_gradient(X, Y, M):
    """
    NumPy reference for GLM gradient computation.

    X: (nfeatures, ntasks, nobs)
    Y: (ntargets, ntasks, nobs)
    M: (nfeatures, ntargets, ntasks)
    Returns: (nfeatures, ntargets, ntasks)
    """
    nfeatures, ntasks, nobs = X.shape
    ntargets, ntasks2, nobs2 = Y.shape
    nfeatures2, ntargets2, ntasks3 = M.shape

    assert ntasks == ntasks2 == ntasks3
    assert nobs == nobs2
    assert nfeatures == nfeatures2
    assert ntargets == ntargets2

    # Compute prediction first
    Y_pred = get_numpy_reference_glm_predict(X, M)

    # Gradient is d/dM ||Y - Y_pred||^2 = -2 * X * (Y - Y_pred)^T
    gradient = np.zeros_like(M)

    for task in range(ntasks):
        # X[:, task, :] is (nfeatures, nobs)
        # (Y - Y_pred)[:, task, :] is (ntargets, nobs)
        # Result should be (nfeatures, ntargets)
        residual = Y[:, task, :] - Y_pred[:, task, :]  # (ntargets, nobs)
        gradient[:, :, task] = X[:, task, :] @ residual.T  # (nfeatures, nobs) @ (nobs, ntargets)

    return gradient

def get_numpy_reference_tensor_sort(tensor, axis_name):
    """
    NumPy reference for 3D tensor sorting.

    Args:
        tensor: 3D array to sort
        axis_name: "rows", "cols", or "depth"

    Returns:
        Sorted tensor (in-place operation simulated)
    """
    result = tensor.copy()

    if axis_name == "rows":
        axis = 0
    elif axis_name == "cols":
        axis = 1
    elif axis_name == "depth":
        axis = 2
    else:
        raise ValueError(f"Unknown axis: {axis_name}")

    return np.sort(result, axis=axis)

def create_non_contiguous_array(base_array, axis=0):
    """
    Create a non-contiguous view of an array for testing contiguity handling.
    """
    if axis == 0:
        return base_array[::2]  # Every other row
    elif axis == 1:
        return base_array[:, ::2]  # Every other column
    else:
        raise ValueError("Only axis 0 and 1 supported")

def is_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def generate_power_of_2_shape(max_dim=64):
    """Generate a 3D shape where all dimensions are powers of 2."""
    powers = [2**i for i in range(1, 7) if 2**i <= max_dim]  # [2, 4, 8, 16, 32, 64]
    return tuple(np.random.choice(powers, size=3))

def print_performance_summary(results: Dict[str, Any], test_name: str):
    """Print a formatted performance summary."""
    print(f"\n{test_name} Performance Summary:")
    print(f"  GPU Time:    {results['gpu_time']*1000:.2f} ms")
    print(f"  NumPy Time:  {results['numpy_time']*1000:.2f} ms")
    print(f"  Speedup:     {results['speedup']:.2f}x")
    print(f"  Max Error:   {results['max_error']:.2e}")
    print(f"  Accuracy:    {'PASS' if results['accuracy_pass'] else 'FAIL'}")

class ErrorCaseBuilder:
    """Helper class to build error test cases systematically."""

    def __init__(self):
        self.cases = []

    def add_dimension_mismatch(self, func_name: str, *arrays_with_wrong_dims):
        """Add test case for dimension mismatch."""
        self.cases.append((
            arrays_with_wrong_dims,
            {},
            ValueError,
            f"{func_name} dimension mismatch"
        ))
        return self

    def add_dtype_mismatch(self, func_name: str, *arrays_with_different_dtypes):
        """Add test case for dtype mismatch."""
        self.cases.append((
            arrays_with_different_dtypes,
            {},
            ValueError,
            f"{func_name} dtype mismatch"
        ))
        return self

    def add_shape_error(self, func_name: str, *arrays_with_wrong_shapes):
        """Add test case for shape errors."""
        self.cases.append((
            arrays_with_wrong_shapes,
            {},
            ValueError,
            f"{func_name} shape error"
        ))
        return self

    def add_invalid_parameter(self, func_name: str, arrays, invalid_param_dict):
        """Add test case for invalid parameter values."""
        self.cases.append((
            arrays,
            invalid_param_dict,
            ValueError,
            f"{func_name} invalid parameter"
        ))
        return self

    def build(self):
        """Return the list of test cases."""
        return self.cases
