"""
Pytest configuration and fixtures for py-gpu-algos test suite.

This module provides common fixtures, test data, and configuration
for testing all GPU kernel bindings.
"""

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
FLOAT_DTYPES = [np.float32, np.float64]

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
def small_matrices():
    """Generate small matrices for basic testing."""
    def _generate(dtype, m=16, n=12, k=20):
        np.random.seed(42)  # Reproducible results
        a = np.random.randn(m, n).astype(dtype)
        b = np.random.randn(n, k).astype(dtype)

        # Scale for integer types to avoid overflow
        if np.issubdtype(dtype, np.integer):
            if dtype in [np.int8, np.uint8]:
                scale = 5
            elif dtype in [np.int16, np.uint16]:
                scale = 50
            else:
                scale = 100
            a = (a * scale).astype(dtype)
            b = (b * scale).astype(dtype)

        return a, b
    return _generate

@pytest.fixture
def small_vectors():
    """Generate small vectors for basic testing."""
    def _generate(dtype, n=1000):
        np.random.seed(42)
        vec = np.random.randn(n).astype(dtype)

        # Scale for integer types
        if np.issubdtype(dtype, np.integer):
            if dtype in [np.int8, np.uint8]:
                scale = 5
            elif dtype in [np.int16, np.uint16]:
                scale = 50
            else:
                scale = 100
            vec = (vec * scale).astype(dtype)

        return vec
    return _generate

@pytest.fixture
def small_tensors_3d():
    """Generate small 3D tensors for GLM and sort testing."""
    def _generate(dtype, shape=(8, 4, 16)):
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(dtype)

        # Scale for integer types
        if np.issubdtype(dtype, np.integer):
            if dtype in [np.int8, np.uint8]:
                scale = 5
            elif dtype in [np.int16, np.uint16]:
                scale = 50
            else:
                scale = 100
            tensor = (tensor * scale).astype(dtype)

        return tensor
    return _generate

@pytest.fixture
def power_of_2_tensors():
    """Generate 3D tensors with power-of-2 dimensions for bitonic sort."""
    def _generate(dtype, shape=(8, 4, 16)):  # All dimensions are powers of 2
        np.random.seed(42)
        # For sort testing, use integers in a reasonable range
        if np.issubdtype(dtype, np.integer):
            if dtype in [np.int8, np.uint8]:
                tensor = np.random.randint(-10, 10, shape, dtype=dtype)
            elif dtype in [np.int16, np.uint16]:
                tensor = np.random.randint(-100, 100, shape, dtype=dtype)
            else:
                tensor = np.random.randint(-1000, 1000, shape, dtype=dtype)
        else:
            tensor = (np.random.randn(*shape) * 100).astype(dtype)

        return tensor
    return _generate

@pytest.fixture
def tolerance():
    """Get appropriate tolerance for floating point comparisons."""
    def _get_tolerance(dtype):
        if dtype == np.float16:
            return {"rtol": 1e-3, "atol": 1e-4}
        elif dtype == np.float32:
            return {"rtol": 1e-5, "atol": 1e-6}
        elif dtype == np.float64:
            return {"rtol": 1e-12, "atol": 1e-14}
        else:
            # Integer types should be exact
            return {"rtol": 0, "atol": 0}
    return _get_tolerance

@pytest.fixture
def performance_sizes():
    """Generate different sizes for performance testing."""
    return [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512)
    ]

@pytest.fixture
def glm_test_data():
    """Generate test data for GLM operations."""
    def _generate(dtype, nfeatures=10, ntargets=3, ntasks=4, nobs=100):
        np.random.seed(42)

        X = np.random.randn(nfeatures, ntasks, nobs).astype(dtype)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(dtype)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(dtype)

        # Scale for integer types
        if np.issubdtype(dtype, np.integer):
            scale = 10
            X = (X * scale).astype(dtype)
            Y = (Y * scale).astype(dtype)
            M = (M * scale).astype(dtype)

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
