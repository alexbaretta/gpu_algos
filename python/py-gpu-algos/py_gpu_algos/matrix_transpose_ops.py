# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/matrix_transpose_ops.py

"""
Matrix transpose operations module for py-gpu-algos

This module provides high-level Python interfaces for matrix transpose operations,
automatically dispatching to the appropriate CUDA backend based on array types.
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, overload
import warnings

from ._module_loader import get_cuda_module

# Load the CUDA module from build directory
_matrix_transpose_ops_cuda = get_cuda_module('_matrix_transpose_ops_cuda')

# Import functions from the CUDA module
_matrix_transpose_naive_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive
_matrix_transpose_striped_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped
_matrix_transpose_tiled_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled
_matrix_transpose_cublas_cuda = _matrix_transpose_ops_cuda.matrix_transpose_cublas

# Import low-level type-specific functions for matrix_transpose_naive
_matrix_transpose_naive_float32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_float32
_matrix_transpose_naive_float64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_float64
_matrix_transpose_naive_int8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_int8
_matrix_transpose_naive_int16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_int16
_matrix_transpose_naive_int32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_int32
_matrix_transpose_naive_int64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_int64
_matrix_transpose_naive_uint8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_uint8
_matrix_transpose_naive_uint16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_uint16
_matrix_transpose_naive_uint32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_uint32
_matrix_transpose_naive_uint64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_naive_uint64

# Import low-level type-specific functions for matrix_transpose_striped
_matrix_transpose_striped_float32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_float32
_matrix_transpose_striped_float64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_float64
_matrix_transpose_striped_int8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_int8
_matrix_transpose_striped_int16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_int16
_matrix_transpose_striped_int32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_int32
_matrix_transpose_striped_int64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_int64
_matrix_transpose_striped_uint8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_uint8
_matrix_transpose_striped_uint16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_uint16
_matrix_transpose_striped_uint32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_uint32
_matrix_transpose_striped_uint64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_striped_uint64

# Import low-level type-specific functions for matrix_transpose_tiled
_matrix_transpose_tiled_float32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_float32
_matrix_transpose_tiled_float64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_float64
_matrix_transpose_tiled_int8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_int8
_matrix_transpose_tiled_int16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_int16
_matrix_transpose_tiled_int32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_int32
_matrix_transpose_tiled_int64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_int64
_matrix_transpose_tiled_uint8_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_uint8
_matrix_transpose_tiled_uint16_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_uint16
_matrix_transpose_tiled_uint32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_uint32
_matrix_transpose_tiled_uint64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_tiled_uint64

# Import low-level type-specific functions for matrix_transpose_cublas (float/double only)
_matrix_transpose_cublas_float32_cuda = _matrix_transpose_ops_cuda.matrix_transpose_cublas_float32
_matrix_transpose_cublas_float64_cuda = _matrix_transpose_ops_cuda.matrix_transpose_cublas_float64

T = TypeVar('T', bound=np.generic)

# Type mapping for matrix transpose operations (all supported types)
_MATRIX_TRANSPOSE_TYPE_DISPATCH_MAP = {
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

# Type mapping for cublas operations (float/double only)
_CUBLAS_TYPE_DISPATCH_MAP = {
    np.dtype(np.float32): 'float32',
    np.dtype(np.float64): 'float64',
}

def _validate_matrix_transpose_inputs(a: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix transpose operations."""
    if a.ndim != 2:
        raise ValueError(f"{operation_name}: Input array must be 2-dimensional")

    if a.dtype not in _MATRIX_TRANSPOSE_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}")

def _validate_matrix_transpose_cublas_inputs(a: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix transpose operations with cublas (float/double only)."""
    if a.ndim != 2:
        raise ValueError(f"{operation_name}: Input array must be 2-dimensional")

    if a.dtype not in _CUBLAS_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}. cublas operations support: float32, float64")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions for matrix_transpose_naive

def matrix_transpose_naive_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix transpose using naive algorithm for float32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_float32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_float32_cuda(a_contig)

def matrix_transpose_naive_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix transpose using naive algorithm for float64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_float64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_float64_cuda(a_contig)

def matrix_transpose_naive_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix transpose using naive algorithm for int8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_int8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_int8_cuda(a_contig)

def matrix_transpose_naive_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix transpose using naive algorithm for int16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_int16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_int16_cuda(a_contig)

def matrix_transpose_naive_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix transpose using naive algorithm for int32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_int32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_int32_cuda(a_contig)

def matrix_transpose_naive_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix transpose using naive algorithm for int64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_int64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_int64_cuda(a_contig)

def matrix_transpose_naive_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix transpose using naive algorithm for uint8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_uint8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_uint8_cuda(a_contig)

def matrix_transpose_naive_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix transpose using naive algorithm for uint16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_uint16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_uint16_cuda(a_contig)

def matrix_transpose_naive_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix transpose using naive algorithm for uint32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_uint32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_uint32_cuda(a_contig)

def matrix_transpose_naive_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix transpose using naive algorithm for uint64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive_uint64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_uint64_cuda(a_contig)

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

# Low-level type-specific functions for matrix_transpose_tiled

def matrix_transpose_tiled_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix transpose using tiled algorithm for float32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_float32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_float32_cuda(a_contig)

def matrix_transpose_tiled_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix transpose using tiled algorithm for float64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_float64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_float64_cuda(a_contig)

def matrix_transpose_tiled_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix transpose using tiled algorithm for int8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_int8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_int8_cuda(a_contig)

def matrix_transpose_tiled_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix transpose using tiled algorithm for int16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_int16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_int16_cuda(a_contig)

def matrix_transpose_tiled_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix transpose using tiled algorithm for int32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_int32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_int32_cuda(a_contig)

def matrix_transpose_tiled_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix transpose using tiled algorithm for int64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_int64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_int64_cuda(a_contig)

def matrix_transpose_tiled_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix transpose using tiled algorithm for uint8 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_uint8")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_uint8_cuda(a_contig)

def matrix_transpose_tiled_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix transpose using tiled algorithm for uint16 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_uint16")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_uint16_cuda(a_contig)

def matrix_transpose_tiled_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix transpose using tiled algorithm for uint32 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_uint32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_uint32_cuda(a_contig)

def matrix_transpose_tiled_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix transpose using tiled algorithm for uint64 arrays."""
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled_uint64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_uint64_cuda(a_contig)

# Low-level type-specific functions for matrix_transpose_cublas (float/double only)

def matrix_transpose_cublas_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix transpose using cuBLAS for float32 arrays."""
    _validate_matrix_transpose_cublas_inputs(a, "matrix_transpose_cublas_float32")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_cublas_float32_cuda(a_contig)

def matrix_transpose_cublas_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix transpose using cuBLAS for float64 arrays."""
    _validate_matrix_transpose_cublas_inputs(a, "matrix_transpose_cublas_float64")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_cublas_float64_cuda(a_contig)

# High-level dispatch functions

@overload
def matrix_transpose_naive(a: NDArray[T]) -> NDArray[T]: ...

def matrix_transpose_naive(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix transpose using naive algorithm with automatic type dispatch.

    This function automatically selects the appropriate low-level implementation
    based on the dtype of the input array.

    Args:
        a: Input matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)

    Raises:
        RuntimeError: If CUDA backend is not available
        ValueError: If input validation fails or unsupported dtype

    Examples:
        >>> import numpy as np
        >>> a = np.random.randn(100, 50).astype(np.float32)
        >>> b = matrix_transpose_naive(a)
        >>> b.shape
        (50, 100)
    """
    _validate_matrix_transpose_inputs(a, "matrix_transpose_naive")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_naive_cuda(a_contig)

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

@overload
def matrix_transpose_tiled(a: NDArray[T]) -> NDArray[T]: ...

def matrix_transpose_tiled(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix transpose using tiled algorithm with automatic type dispatch.

    Args:
        a: Input matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    _validate_matrix_transpose_inputs(a, "matrix_transpose_tiled")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_tiled_cuda(a_contig)

@overload
def matrix_transpose_cublas(a: NDArray[T]) -> NDArray[T]: ...

def matrix_transpose_cublas(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix transpose using cuBLAS with automatic type dispatch.

    Args:
        a: Input matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)

    Note:
        This function supports only float32 and float64 types.
    """
    _validate_matrix_transpose_cublas_inputs(a, "matrix_transpose_cublas")
    a_contig = _ensure_contiguous(a)

    return _matrix_transpose_cublas_cuda(a_contig)

# Export all functions
__all__ = [
    # High-level dispatch functions
    'matrix_transpose_naive',
    'matrix_transpose_striped',
    'matrix_transpose_tiled',
    'matrix_transpose_cublas',

    # Low-level type-specific functions for matrix_transpose_naive
    'matrix_transpose_naive_float32',
    'matrix_transpose_naive_float64',
    'matrix_transpose_naive_int8',
    'matrix_transpose_naive_int16',
    'matrix_transpose_naive_int32',
    'matrix_transpose_naive_int64',
    'matrix_transpose_naive_uint8',
    'matrix_transpose_naive_uint16',
    'matrix_transpose_naive_uint32',
    'matrix_transpose_naive_uint64',

    # Low-level type-specific functions for matrix_transpose_striped
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

    # Low-level type-specific functions for matrix_transpose_tiled
    'matrix_transpose_tiled_float32',
    'matrix_transpose_tiled_float64',
    'matrix_transpose_tiled_int8',
    'matrix_transpose_tiled_int16',
    'matrix_transpose_tiled_int32',
    'matrix_transpose_tiled_int64',
    'matrix_transpose_tiled_uint8',
    'matrix_transpose_tiled_uint16',
    'matrix_transpose_tiled_uint32',
    'matrix_transpose_tiled_uint64',

    # Low-level type-specific functions for matrix_transpose_cublas
    'matrix_transpose_cublas_float32',
    'matrix_transpose_cublas_float64',
]
