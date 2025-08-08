# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/matrix_ops.py

"""
Matrix operations module for py-gpu-algos

This module provides high-level Python interfaces for matrix operations,
automatically dispatching to the appropriate CUDA backend based on array types.
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, overload
import warnings

try:
    from ._module_loader import get_cuda_module

    # Load the CUDA module from build directory
    _matrix_ops_cuda = get_cuda_module('_matrix_ops_cuda')

    # Import functions from the CUDA module
    _matrix_product_naive_cuda = _matrix_ops_cuda.matrix_product_naive
    # _matrix_product_naive_float16_cuda = _matrix_ops_cuda.matrix_product_naive_float16  # TODO: Handle float16 properly
    _matrix_product_naive_float32_cuda = _matrix_ops_cuda.matrix_product_naive_float32
    _matrix_product_naive_float64_cuda = _matrix_ops_cuda.matrix_product_naive_float64
    _matrix_product_naive_int8_cuda = _matrix_ops_cuda.matrix_product_naive_int8
    _matrix_product_naive_int16_cuda = _matrix_ops_cuda.matrix_product_naive_int16
    _matrix_product_naive_int32_cuda = _matrix_ops_cuda.matrix_product_naive_int32
    _matrix_product_naive_int64_cuda = _matrix_ops_cuda.matrix_product_naive_int64
    _matrix_product_naive_uint8_cuda = _matrix_ops_cuda.matrix_product_naive_uint8
    _matrix_product_naive_uint16_cuda = _matrix_ops_cuda.matrix_product_naive_uint16
    _matrix_product_naive_uint32_cuda = _matrix_ops_cuda.matrix_product_naive_uint32
    _matrix_product_naive_uint64_cuda = _matrix_ops_cuda.matrix_product_naive_uint64

    # Additional matrix operations
    _matrix_product_tiled_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled', None)
    _matrix_product_tiled_float32_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_float32', None)
    _matrix_product_tiled_float64_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_float64', None)
    _matrix_product_tiled_int8_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_int8', None)
    _matrix_product_tiled_int16_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_int16', None)
    _matrix_product_tiled_int32_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_int32', None)
    _matrix_product_tiled_int64_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_int64', None)
    _matrix_product_tiled_uint8_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_uint8', None)
    _matrix_product_tiled_uint16_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_uint16', None)
    _matrix_product_tiled_uint32_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_uint32', None)
    _matrix_product_tiled_uint64_cuda = getattr(_matrix_ops_cuda, 'matrix_product_tiled_uint64', None)

    _matrix_transpose_striped_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped', None)
    _matrix_transpose_striped_float32_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_float32', None)
    _matrix_transpose_striped_float64_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_float64', None)
    _matrix_transpose_striped_int8_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_int8', None)
    _matrix_transpose_striped_int16_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_int16', None)
    _matrix_transpose_striped_int32_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_int32', None)
    _matrix_transpose_striped_int64_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_int64', None)
    _matrix_transpose_striped_uint8_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_uint8', None)
    _matrix_transpose_striped_uint16_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_uint16', None)
    _matrix_transpose_striped_uint32_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_uint32', None)
    _matrix_transpose_striped_uint64_cuda = getattr(_matrix_ops_cuda, 'matrix_transpose_striped_uint64', None)
except Exception as e:
    raise e


T = TypeVar('T', bound=np.generic)

# Type mapping for low-level function dispatch
_TYPE_DISPATCH_MAP = {
    # np.dtype(np.float16): 'float16',  # TODO: Handle float16 properly
    np.dtype(np.float32): 'float32',
    np.dtype(np.float64): 'float64',
    np.dtype(np.int8): 'int8',
    np.dtype(np.int16): 'int16',
    np.dtype(np.int32): 'int32',
    np.dtype(np.int64): 'int64',
    np.dtype(np.uint8): 'uint8',
    np.dtype(np.uint16): 'uint16',
    np.dtype(np.uint32): 'uint32',
    np.dtype(np.uint64): 'uint64',
}

def _validate_matrix_inputs(a: np.ndarray, b: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix operations."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{operation_name}: Input arrays must be 2-dimensional")

    if a.dtype != b.dtype:
        raise ValueError(f"{operation_name}: Input arrays must have the same dtype")

    if a.dtype not in _TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"{operation_name}: Matrix dimensions incompatible for multiplication: "
                        f"({a.shape[0]}, {a.shape[1]}) @ ({b.shape[0]}, {b.shape[1]})")

def _validate_matrix_transpose_inputs(a: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix transpose operations."""
    if a.ndim != 2:
        raise ValueError(f"{operation_name}: Input array must be 2-dimensional")

    if a.dtype not in _TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions

# TODO: Handle float16 properly - commented out until we figure out the right approach
# def matrix_product_naive_float16(a: NDArray[np.float16], b: NDArray[np.float16]) -> NDArray[np.float16]:
#     """Matrix multiplication using naive algorithm for float16 arrays.
#
#     Args:
#         a: Input matrix A of shape (m, n)
#         b: Input matrix B of shape (n, k)
#
#     Returns:
#         Result matrix C of shape (m, k) where C = A @ B
#
#     Raises:
#         RuntimeError: If CUDA backend is not available
#         ValueError: If input validation fails
#     """
#     if not _CUDA_AVAILABLE:
#         raise RuntimeError("CUDA backend not available")
#
#     _validate_matrix_inputs(a, b, "matrix_product_naive_float16")
#     a_contig = _ensure_contiguous(a)
#     b_contig = _ensure_contiguous(b)
#
#     return _matrix_product_naive_float16_cuda(a_contig, b_contig)

def matrix_product_naive_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using naive algorithm for float32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_float32_cuda(a_contig, b_contig)

def matrix_product_naive_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using naive algorithm for float64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_float64_cuda(a_contig, b_contig)

def matrix_product_naive_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix multiplication using naive algorithm for int8 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_int8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int8_cuda(a_contig, b_contig)

def matrix_product_naive_int16(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix multiplication using naive algorithm for int16 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_int16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int16_cuda(a_contig, b_contig)

def matrix_product_naive_int32(a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix multiplication using naive algorithm for int32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_int32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int32_cuda(a_contig, b_contig)

def matrix_product_naive_int64(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix multiplication using naive algorithm for int64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_int64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int64_cuda(a_contig, b_contig)

def matrix_product_naive_uint8(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix multiplication using naive algorithm for uint8 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_uint8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint8_cuda(a_contig, b_contig)

def matrix_product_naive_uint16(a: NDArray[np.uint16], b: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix multiplication using naive algorithm for uint16 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_uint16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint16_cuda(a_contig, b_contig)

def matrix_product_naive_uint32(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix multiplication using naive algorithm for uint32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_uint32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint32_cuda(a_contig, b_contig)

def matrix_product_naive_uint64(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix multiplication using naive algorithm for uint64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_naive_uint64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint64_cuda(a_contig, b_contig)

# High-level dispatch function

@overload
def matrix_product_naive(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_naive(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using naive algorithm with automatic type dispatch.

    This function automatically selects the appropriate low-level implementation
    based on the dtype of the input arrays.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B

    Raises:
        RuntimeError: If CUDA backend is not available
        ValueError: If input validation fails or unsupported dtype

    Examples:
        >>> import numpy as np
        >>> a = np.random.randn(100, 50).astype(np.float32)
        >>> b = np.random.randn(50, 80).astype(np.float32)
        >>> c = matrix_product_naive(a, b)
        >>> c.shape
        (100, 80)
    """
    _validate_matrix_inputs(a, b, "matrix_product_naive")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_cuda(a_contig, b_contig)

# Additional matrix operations

# Low-level type-specific functions for matrix_product_tiled

def matrix_product_tiled_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using tiled algorithm for float32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_float32_cuda(a_contig, b_contig)

def matrix_product_tiled_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using tiled algorithm for float64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_float64_cuda(a_contig, b_contig)

def matrix_product_tiled_int32(a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix multiplication using tiled algorithm for int32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_int32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int32_cuda(a_contig, b_contig)

def matrix_product_tiled_int64(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix multiplication using tiled algorithm for int64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_int64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int64_cuda(a_contig, b_contig)

def matrix_product_tiled_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix multiplication using tiled algorithm for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")
    if _matrix_product_tiled_int8_cuda is None:
        raise NotImplementedError("matrix_product_tiled for int8 is not implemented in the CUDA backend")

    _validate_matrix_inputs(a, b, "matrix_product_tiled_int8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int8_cuda(a_contig, b_contig)

def matrix_product_tiled_int16(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix multiplication using tiled algorithm for int16 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_int16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int16_cuda(a_contig, b_contig)

def matrix_product_tiled_uint8(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix multiplication using tiled algorithm for uint8 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_uint8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint8_cuda(a_contig, b_contig)

def matrix_product_tiled_uint16(a: NDArray[np.uint16], b: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix multiplication using tiled algorithm for uint16 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_uint16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint16_cuda(a_contig, b_contig)

def matrix_product_tiled_uint32(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix multiplication using tiled algorithm for uint32 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_uint32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint32_cuda(a_contig, b_contig)

def matrix_product_tiled_uint64(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix multiplication using tiled algorithm for uint64 arrays."""
    _validate_matrix_inputs(a, b, "matrix_product_tiled_uint64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint64_cuda(a_contig, b_contig)

# High-level dispatch function for tiled matrix product

@overload
def matrix_product_tiled(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_tiled(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using tiled algorithm with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B

    Note:
        This function supports all numeric types: float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64
    """
    _validate_matrix_inputs(a, b, "matrix_product_tiled")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_transpose_striped

def matrix_transpose_striped_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix transpose using striped algorithm for float32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_float32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_float32_cuda(a_contig)

def matrix_transpose_striped_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix transpose using striped algorithm for float64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_float64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_float64_cuda(a_contig)

def matrix_transpose_striped_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix transpose using striped algorithm for int8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_int8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_int8_cuda(a_contig)

def matrix_transpose_striped_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix transpose using striped algorithm for int16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_int16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_int16_cuda(a_contig)

def matrix_transpose_striped_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix transpose using striped algorithm for int32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_int32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_int32_cuda(a_contig)

def matrix_transpose_striped_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix transpose using striped algorithm for int64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_int64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_int64_cuda(a_contig)

def matrix_transpose_striped_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix transpose using striped algorithm for uint8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_uint8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_uint8_cuda(a_contig)

def matrix_transpose_striped_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix transpose using striped algorithm for uint16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_uint16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_uint16_cuda(a_contig)

def matrix_transpose_striped_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix transpose using striped algorithm for uint32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_uint32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_uint32_cuda(a_contig)

def matrix_transpose_striped_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix transpose using striped algorithm for uint64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped_uint64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_uint64_cuda(a_contig)

# High-level dispatch function for striped matrix transpose

@overload
def matrix_transpose_striped(a: NDArray[T]) -> NDArray[T]: ...

def matrix_transpose_striped(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix transpose using striped algorithm with automatic type dispatch.

    Args:
        a: Input matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    _validate_matrix_transpose_inputs(a, "matrix_transpose_striped")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_striped_cuda(a_contig)

# Export all functions
__all__ = [
    # Matrix product naive
    'matrix_product_naive',
    # 'matrix_product_naive_float16',  # TODO: Handle float16 properly
    'matrix_product_naive_float32',
    'matrix_product_naive_float64',
    'matrix_product_naive_int8',
    'matrix_product_naive_int16',
    'matrix_product_naive_int32',
    'matrix_product_naive_int64',
    'matrix_product_naive_uint8',
    'matrix_product_naive_uint16',
    'matrix_product_naive_uint32',
    'matrix_product_naive_uint64',

    # Matrix product tiled
    'matrix_product_tiled',
    'matrix_product_tiled_float32',
    'matrix_product_tiled_float64',
    'matrix_product_tiled_int32',
    'matrix_product_tiled_int64',

    # Matrix transpose striped
    'matrix_transpose_striped',
    'matrix_transpose_striped_float32',
    'matrix_transpose_striped_float64',
    'matrix_transpose_striped_int8',
    'matrix_transpose_striped_int16',
    'matrix_transpose_striped_int32',
    'matrix_transpose_striped_int64',
    'matrix_transpose_striped_uint8',
    'matrix_transpose_striped_uint16',
    'matrix_transpose_striped_uint32',
    'matrix_transpose_striped_uint64',
]
