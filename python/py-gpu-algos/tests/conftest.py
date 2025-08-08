"""
Pytest configuration and fixtures for py-gpu-algos test suite.

This module provides common fixtures, test data, and configuration
for testing all GPU kernel bindings.
"""

import itertools
import pytest
import numpy as np
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if CUDA is available
try:
    import py_gpu_algos
    CUDA_AVAILABLE = True  # Set to True if we can import the package
except ImportError:
    CUDA_AVAILABLE = False

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

# All supported dtypes for testing
ALL_DTYPES = [
    np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
]

# Common floating point dtypes
FLOAT_DTYPES = [np.float32, np.float64] # Eventually will include np.float16
GEMM_DTYPES = [np.float32, np.float64]

# Common integer dtypes
INT_DTYPES = [np.int8, np.int16, np.int32, np.int64]

# Common unsigned integer dtypes
UINT_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]

# GLM-supported dtypes (restricted set)
GLM_DTYPES = [np.float32, np.float64, np.int32, np.int64]

# Scan operation types
SCAN_OPERATIONS = ["sum", "max", "min", "prod"]

# Sort axis options for 3D tensors
SORT_AXES = ["rows", "cols", "depth"]

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA backend is available."""
    return CUDA_AVAILABLE

@pytest.fixture(params=ALL_DTYPES, ids=lambda x: x.__name__)
def dtype_all(request):
    """Fixture providing all supported dtypes."""
    return request.param

@pytest.fixture(params=FLOAT_DTYPES, ids=lambda x: x.__name__)
def dtype_float(request):
    """Fixture providing floating point dtypes."""
    return request.param

@pytest.fixture(params=INT_DTYPES + UINT_DTYPES, ids=lambda x: x.__name__)
def dtype_int(request):
    """Fixture providing integer dtypes."""
    return request.param

@pytest.fixture(params=GLM_DTYPES, ids=lambda x: x.__name__)
def dtype_glm(request):
    """Fixture providing GLM-supported dtypes."""
    return request.param

@pytest.fixture(params=SCAN_OPERATIONS)
def scan_operation(request):
    """Fixture providing scan operation types."""
    return request.param

@pytest.fixture(params=SORT_AXES)
def sort_axis(request):
    """Fixture providing sort axis options."""
    return request.param


@pytest.fixture
def test_input_vector_incremental():
    """Generate incremental vector"""
    def _generate(dtype, n):
        vec = np.arange(n, dtype=dtype)

        return vec
    return _generate

@pytest.fixture
def test_input_vector_random():
    """Generate random vector"""
    def _generate(dtype, n):
        generator = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            vec = generator.integers(0, 100, n).astype(dtype)
        else:
            vec = generator.uniform(low=0, high=1, size=n).astype(dtype)
            pass

        return vec
    return _generate




TEST_PROBLEM_SIZES = [
    10, 33, 100, 256,
    # 1000, 10000
]

def pick_problem_sizes(i_start:int, stride:int) -> tuple[int, int, int]:
    """Pick problem sizes from TEST_SIZES"""
    l = len(TEST_PROBLEM_SIZES)
    i_m = i_start % l
    i_k = (i_m + stride) % l
    i_n = (i_k + stride) % l
    return (TEST_PROBLEM_SIZES[i_m], TEST_PROBLEM_SIZES[i_k], TEST_PROBLEM_SIZES[i_n])

TEST_M_K_N = list(itertools.chain(
    [pick_problem_sizes(i, 1) for i in range(len(TEST_PROBLEM_SIZES))],
    [pick_problem_sizes(i, 2) for i in range(len(TEST_PROBLEM_SIZES))],
    [pick_problem_sizes(i, 3) for i in range(len(TEST_PROBLEM_SIZES))],
))

@pytest.fixture(params=TEST_M_K_N)
def test_shape_triplet(request):
    """Fixture that provides each triplet of test shapes from TEST_SHAPES."""
    return request.param

@pytest.fixture
def test_input_matrix_incremental():
    """Generate incremental matrix"""
    def _generate(dtype, nrows, ncols, start=0):
        mat = np.arange(start, start + nrows * ncols)
        result = mat.astype(dtype).reshape(nrows, ncols)
        return result
    return _generate

@pytest.fixture
def test_input_matrix_random():
    """Generate random matrix"""
    def _generate(dtype, nrows, ncols):
        generator = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            vec = generator.integers(0, 100, (nrows, ncols), dtype=dtype)
        else:
            vec = generator.uniform(low=0, high=1, size=(nrows, ncols)).astype(dtype)
            pass

        return vec
    return _generate

@pytest.fixture
def test_input_tensor_3d_incremental():
    """Generate incremental tensor_3d"""
    def _generate(dtype, m=100, k=100, n=100):
        mat = np.arange(m * k * n, dtype=dtype).reshape(m, k, n)

        return mat
    return _generate

@pytest.fixture
def test_input_tensor_3d_random():
    """Generate random tensor_3d"""
    def _generate(dtype, m=100, k=100, n=100):
        generator = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            vec = generator.integers(0, 100, m * k * n).astype(dtype).reshape(m, k, n)
        else:
            vec = generator.uniform(low=0, high=1, size=m * k * n).astype(dtype).reshape(m, k, n)
            pass

        return vec
    return _generate


@pytest.fixture
def performance_sizes():
    """Generate different sizes for performance testing."""
    return [ 2**(3*i) for i in range(2, 7) ]

@pytest.fixture
def glm_test_data():
    """Generate test data for GLM operations."""
    def _generate(dtype, nfeatures=10, ntargets=3, ntasks=4, nobs=100):
        generator = np.random.default_rng(42)

        if np.issubdtype(dtype, np.floating):
            X = generator.normal(1, 1, (nfeatures, ntasks, nobs)).astype(dtype)
            Y = generator.normal(1, 1, (ntargets, ntasks, nobs)).astype(dtype)
            M = generator.normal(1, 1, (nfeatures, ntargets, ntasks)).astype(dtype)
        else:
            X = generator.integers(-100, 100, (nfeatures, ntasks, nobs)).astype(dtype)
            Y = generator.integers(-100, 100, (ntargets, ntasks, nobs)).astype(dtype)
            M = generator.integers(-100, 100, (nfeatures, ntargets, ntasks)).astype(dtype)
            pass

        return X, Y, M
    return _generate

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark large parameter tests as slow
        if hasattr(item, 'callspec') and item.callspec:
            params = item.callspec.params
            if any(isinstance(p, type) and p in ALL_DTYPES for p in params.values()):
                # Tests with all dtypes are slower
                if len([p for p in params.values() if isinstance(p, type) and p in ALL_DTYPES]) > 0:
                    item.add_marker(pytest.mark.slow)
