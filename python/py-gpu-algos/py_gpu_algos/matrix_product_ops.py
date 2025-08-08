# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/matrix_product_ops.py

"""
Matrix product operations module for py-gpu-algos

This module provides high-level Python interfaces for matrix product operations,
automatically dispatching to the appropriate CUDA backend based on array types.
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, overload
import warnings

from ._module_loader import get_cuda_module

# Load the CUDA module from build directory
_matrix_product_ops_cuda = get_cuda_module('_matrix_product_ops_cuda')

# Import functions from the CUDA module
_matrix_product_naive_cuda = _matrix_product_ops_cuda.matrix_product_naive
_matrix_product_tiled_cuda = _matrix_product_ops_cuda.matrix_product_tiled
_matrix_product_warp_cuda = _matrix_product_ops_cuda.matrix_product_warp
_matrix_product_cublas_cuda = _matrix_product_ops_cuda.matrix_product_cublas
_matrix_product_cutlass_cuda = _matrix_product_ops_cuda.matrix_product_cutlass
_matrix_product_tensor_cuda = _matrix_product_ops_cuda.matrix_product_tensor

# Import low-level type-specific functions for matrix_product_naive
_matrix_product_naive_float32_cuda = _matrix_product_ops_cuda.matrix_product_naive_float32
_matrix_product_naive_float64_cuda = _matrix_product_ops_cuda.matrix_product_naive_float64
_matrix_product_naive_int8_cuda = _matrix_product_ops_cuda.matrix_product_naive_int8
_matrix_product_naive_int16_cuda = _matrix_product_ops_cuda.matrix_product_naive_int16
_matrix_product_naive_int32_cuda = _matrix_product_ops_cuda.matrix_product_naive_int32
_matrix_product_naive_int64_cuda = _matrix_product_ops_cuda.matrix_product_naive_int64
_matrix_product_naive_uint8_cuda = _matrix_product_ops_cuda.matrix_product_naive_uint8
_matrix_product_naive_uint16_cuda = _matrix_product_ops_cuda.matrix_product_naive_uint16
_matrix_product_naive_uint32_cuda = _matrix_product_ops_cuda.matrix_product_naive_uint32
_matrix_product_naive_uint64_cuda = _matrix_product_ops_cuda.matrix_product_naive_uint64

# Import low-level type-specific functions for matrix_product_tiled
_matrix_product_tiled_float32_cuda = _matrix_product_ops_cuda.matrix_product_tiled_float32
_matrix_product_tiled_float64_cuda = _matrix_product_ops_cuda.matrix_product_tiled_float64
_matrix_product_tiled_int8_cuda = _matrix_product_ops_cuda.matrix_product_tiled_int8
_matrix_product_tiled_int16_cuda = _matrix_product_ops_cuda.matrix_product_tiled_int16
_matrix_product_tiled_int32_cuda = _matrix_product_ops_cuda.matrix_product_tiled_int32
_matrix_product_tiled_int64_cuda = _matrix_product_ops_cuda.matrix_product_tiled_int64
_matrix_product_tiled_uint8_cuda = _matrix_product_ops_cuda.matrix_product_tiled_uint8
_matrix_product_tiled_uint16_cuda = _matrix_product_ops_cuda.matrix_product_tiled_uint16
_matrix_product_tiled_uint32_cuda = _matrix_product_ops_cuda.matrix_product_tiled_uint32
_matrix_product_tiled_uint64_cuda = _matrix_product_ops_cuda.matrix_product_tiled_uint64

# Import low-level type-specific functions for matrix_product_warp
_matrix_product_warp_float32_cuda = _matrix_product_ops_cuda.matrix_product_warp_float32
_matrix_product_warp_float64_cuda = _matrix_product_ops_cuda.matrix_product_warp_float64
_matrix_product_warp_int8_cuda = _matrix_product_ops_cuda.matrix_product_warp_int8
_matrix_product_warp_int16_cuda = _matrix_product_ops_cuda.matrix_product_warp_int16
_matrix_product_warp_int32_cuda = _matrix_product_ops_cuda.matrix_product_warp_int32
_matrix_product_warp_int64_cuda = _matrix_product_ops_cuda.matrix_product_warp_int64
_matrix_product_warp_uint8_cuda = _matrix_product_ops_cuda.matrix_product_warp_uint8
_matrix_product_warp_uint16_cuda = _matrix_product_ops_cuda.matrix_product_warp_uint16
_matrix_product_warp_uint32_cuda = _matrix_product_ops_cuda.matrix_product_warp_uint32
_matrix_product_warp_uint64_cuda = _matrix_product_ops_cuda.matrix_product_warp_uint64

# Import low-level type-specific functions for matrix_product_cublas (float/double only)
_matrix_product_cublas_float32_cuda = _matrix_product_ops_cuda.matrix_product_cublas_float32
_matrix_product_cublas_float64_cuda = _matrix_product_ops_cuda.matrix_product_cublas_float64

# Import low-level type-specific functions for matrix_product_cutlass (float/double only)
_matrix_product_cutlass_float32_cuda = _matrix_product_ops_cuda.matrix_product_cutlass_float32
_matrix_product_cutlass_float64_cuda = _matrix_product_ops_cuda.matrix_product_cutlass_float64

# Import low-level type-specific functions for matrix_product_tensor (float/double only)
_matrix_product_tensor_float32_cuda = _matrix_product_ops_cuda.matrix_product_tensor_float32
_matrix_product_tensor_float64_cuda = _matrix_product_ops_cuda.matrix_product_tensor_float64

T = TypeVar('T', bound=np.generic)

# Type mapping for matrix product operations (all supported types)
_MATRIX_PRODUCT_TYPE_DISPATCH_MAP = {
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

# Type mapping for cublas/cutlass/tensor operations (float/double only)
_CUBLAS_TYPE_DISPATCH_MAP = {
    np.dtype(np.float32): 'float32',
    np.dtype(np.float64): 'float64',
}

def _validate_matrix_product_inputs(a: np.ndarray, b: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix product operations."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{operation_name}: Input arrays must be 2-dimensional")

    if a.dtype != b.dtype:
        raise ValueError(f"{operation_name}: Input arrays must have the same dtype")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"{operation_name}: Matrix dimensions incompatible for multiplication: "
                        f"({a.shape[0]}, {a.shape[1]}) @ ({b.shape[0]}, {b.shape[1]})")

def _validate_matrix_product_cublas_inputs(a: np.ndarray, b: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix product operations with cublas/cutlass (float/double only)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{operation_name}: Input arrays must be 2-dimensional")

    if a.dtype != b.dtype:
        raise ValueError(f"{operation_name}: Input arrays must have the same dtype")

    if a.dtype not in _CUBLAS_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}. cublas/cutlass operations support: float32, float64")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"{operation_name}: Matrix dimensions incompatible for multiplication: "
                        f"({a.shape[0]}, {a.shape[1]}) @ ({b.shape[0]}, {b.shape[1]})")

def _validate_matrix_product_tensor_inputs(a: np.ndarray, b: np.ndarray, operation_name: str) -> None:
    """Validate inputs for matrix product operations with tensor cores (float/double only)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{operation_name}: Input arrays must be 2-dimensional")

    if a.dtype != b.dtype:
        raise ValueError(f"{operation_name}: Input arrays must have the same dtype")

    if a.dtype not in _CUBLAS_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}. tensor operations support: float32, float64")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"{operation_name}: Matrix dimensions incompatible for multiplication: "
                        f"({a.shape[0]}, {a.shape[1]}) @ ({b.shape[0]}, {b.shape[1]})")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions for matrix_product_naive

def matrix_product_naive_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using naive algorithm for float32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_float32_cuda(a_contig, b_contig)

def matrix_product_naive_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using naive algorithm for float64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_float64_cuda(a_contig, b_contig)

def matrix_product_naive_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix multiplication using naive algorithm for int8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_int8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int8_cuda(a_contig, b_contig)

def matrix_product_naive_int16(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix multiplication using naive algorithm for int16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_int16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int16_cuda(a_contig, b_contig)

def matrix_product_naive_int32(a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix multiplication using naive algorithm for int32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_int32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int32_cuda(a_contig, b_contig)

def matrix_product_naive_int64(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix multiplication using naive algorithm for int64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_int64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_int64_cuda(a_contig, b_contig)

def matrix_product_naive_uint8(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix multiplication using naive algorithm for uint8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_uint8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint8_cuda(a_contig, b_contig)

def matrix_product_naive_uint16(a: NDArray[np.uint16], b: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix multiplication using naive algorithm for uint16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_uint16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint16_cuda(a_contig, b_contig)

def matrix_product_naive_uint32(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix multiplication using naive algorithm for uint32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_uint32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint32_cuda(a_contig, b_contig)

def matrix_product_naive_uint64(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix multiplication using naive algorithm for uint64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_naive_uint64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_uint64_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_product_tiled

def matrix_product_tiled_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using tiled algorithm for float32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_float32_cuda(a_contig, b_contig)

def matrix_product_tiled_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using tiled algorithm for float64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_float64_cuda(a_contig, b_contig)

def matrix_product_tiled_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix multiplication using tiled algorithm for int8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_int8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int8_cuda(a_contig, b_contig)

def matrix_product_tiled_int16(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix multiplication using tiled algorithm for int16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_int16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int16_cuda(a_contig, b_contig)

def matrix_product_tiled_int32(a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix multiplication using tiled algorithm for int32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_int32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int32_cuda(a_contig, b_contig)

def matrix_product_tiled_int64(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix multiplication using tiled algorithm for int64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_int64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_int64_cuda(a_contig, b_contig)

def matrix_product_tiled_uint8(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix multiplication using tiled algorithm for uint8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_uint8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint8_cuda(a_contig, b_contig)

def matrix_product_tiled_uint16(a: NDArray[np.uint16], b: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix multiplication using tiled algorithm for uint16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_uint16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint16_cuda(a_contig, b_contig)

def matrix_product_tiled_uint32(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix multiplication using tiled algorithm for uint32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_uint32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint32_cuda(a_contig, b_contig)

def matrix_product_tiled_uint64(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix multiplication using tiled algorithm for uint64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled_uint64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_uint64_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_product_warp

def matrix_product_warp_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using warp algorithm for float32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_float32_cuda(a_contig, b_contig)

def matrix_product_warp_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using warp algorithm for float64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_float64_cuda(a_contig, b_contig)

def matrix_product_warp_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Matrix multiplication using warp algorithm for int8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_int8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_int8_cuda(a_contig, b_contig)

def matrix_product_warp_int16(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Matrix multiplication using warp algorithm for int16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_int16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_int16_cuda(a_contig, b_contig)

def matrix_product_warp_int32(a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
    """Matrix multiplication using warp algorithm for int32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_int32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_int32_cuda(a_contig, b_contig)

def matrix_product_warp_int64(a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Matrix multiplication using warp algorithm for int64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_int64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_int64_cuda(a_contig, b_contig)

def matrix_product_warp_uint8(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Matrix multiplication using warp algorithm for uint8 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_uint8")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_uint8_cuda(a_contig, b_contig)

def matrix_product_warp_uint16(a: NDArray[np.uint16], b: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Matrix multiplication using warp algorithm for uint16 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_uint16")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_uint16_cuda(a_contig, b_contig)

def matrix_product_warp_uint32(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Matrix multiplication using warp algorithm for uint32 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_uint32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_uint32_cuda(a_contig, b_contig)

def matrix_product_warp_uint64(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Matrix multiplication using warp algorithm for uint64 arrays."""
    _validate_matrix_product_inputs(a, b, "matrix_product_warp_uint64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_uint64_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_product_cublas (float/double only)

def matrix_product_cublas_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using cuBLAS for float32 arrays."""
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cublas_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cublas_float32_cuda(a_contig, b_contig)

def matrix_product_cublas_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using cuBLAS for float64 arrays."""
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cublas_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cublas_float64_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_product_cutlass (float/double only)

def matrix_product_cutlass_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using CUTLASS for float32 arrays."""
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cutlass_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cutlass_float32_cuda(a_contig, b_contig)

def matrix_product_cutlass_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using CUTLASS for float64 arrays."""
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cutlass_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cutlass_float64_cuda(a_contig, b_contig)

# Low-level type-specific functions for matrix_product_tensor (float/double only)

def matrix_product_tensor_float32(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Matrix multiplication using Tensor Cores for float32 arrays."""
    _validate_matrix_product_tensor_inputs(a, b, "matrix_product_tensor_float32")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tensor_float32_cuda(a_contig, b_contig)

def matrix_product_tensor_float64(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix multiplication using Tensor Cores for float64 arrays."""
    _validate_matrix_product_tensor_inputs(a, b, "matrix_product_tensor_float64")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tensor_float64_cuda(a_contig, b_contig)

# High-level dispatch functions

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
    _validate_matrix_product_inputs(a, b, "matrix_product_naive")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_naive_cuda(a_contig, b_contig)

@overload
def matrix_product_tiled(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_tiled(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using tiled algorithm with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B
    """
    _validate_matrix_product_inputs(a, b, "matrix_product_tiled")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tiled_cuda(a_contig, b_contig)

@overload
def matrix_product_warp(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_warp(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using warp algorithm with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B
    """
    _validate_matrix_product_inputs(a, b, "matrix_product_warp")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_warp_cuda(a_contig, b_contig)

@overload
def matrix_product_cublas(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_cublas(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using cuBLAS with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B

    Note:
        This function supports only float32 and float64 types.
    """
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cublas")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cublas_cuda(a_contig, b_contig)

@overload
def matrix_product_cutlass(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_cutlass(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using CUTLASS with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, n)
        b: Input matrix B of shape (n, k)

    Returns:
        Result matrix C of shape (m, k) where C = A @ B

    Note:
        This function supports only float32 and float64 types.
    """
    _validate_matrix_product_cublas_inputs(a, b, "matrix_product_cutlass")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_cutlass_cuda(a_contig, b_contig)

@overload
def matrix_product_tensor(a: NDArray[T], b: NDArray[T]) -> NDArray[T]: ...

def matrix_product_tensor(a: Union[NDArray[T], np.ndarray], b: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Matrix multiplication using Tensor Cores with automatic type dispatch.

    Args:
        a: Input matrix A of shape (m, k)
        b: Input matrix B of shape (k, n)

    Returns:
        Result matrix C of shape (m, n) where C = A @ B

    Note:
        This function supports only float32 and float64 types.
    """
    _validate_matrix_product_tensor_inputs(a, b, "matrix_product_tensor")
    a_contig = _ensure_contiguous(a)
    b_contig = _ensure_contiguous(b)

    return _matrix_product_tensor_cuda(a_contig, b_contig)

# Export all functions
__all__ = [
    # High-level dispatch functions
    'matrix_product_naive',
    'matrix_product_tiled',
    'matrix_product_warp',
    'matrix_product_cublas',
    'matrix_product_cutlass',
    'matrix_product_tensor',

    # Low-level type-specific functions for matrix_product_naive
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

    # Low-level type-specific functions for matrix_product_tiled
    'matrix_product_tiled_float32',
    'matrix_product_tiled_float64',
    'matrix_product_tiled_int8',
    'matrix_product_tiled_int16',
    'matrix_product_tiled_int32',
    'matrix_product_tiled_int64',
    'matrix_product_tiled_uint8',
    'matrix_product_tiled_uint16',
    'matrix_product_tiled_uint32',
    'matrix_product_tiled_uint64',

    # Low-level type-specific functions for matrix_product_warp
    'matrix_product_warp_float32',
    'matrix_product_warp_float64',
    'matrix_product_warp_int8',
    'matrix_product_warp_int16',
    'matrix_product_warp_int32',
    'matrix_product_warp_int64',
    'matrix_product_warp_uint8',
    'matrix_product_warp_uint16',
    'matrix_product_warp_uint32',
    'matrix_product_warp_uint64',

    # Low-level type-specific functions for matrix_product_cublas
    'matrix_product_cublas_float32',
    'matrix_product_cublas_float64',

    # Low-level type-specific functions for matrix_product_cutlass
    'matrix_product_cutlass_float32',
    'matrix_product_cutlass_float64',

    # Low-level type-specific functions for matrix_product_tensor
    'matrix_product_tensor_float32',
    'matrix_product_tensor_float64',
]
