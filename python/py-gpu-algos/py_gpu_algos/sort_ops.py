# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/sort_ops.py

"""
Sort operations module for py-gpu-algos

This module provides high-level Python interfaces for sort operations,
automatically dispatching to the appropriate CUDA backend based on array types.

Sort operations work with 3D tensors and perform in-place sorting along specified dimensions.
Currently supports bitonic sort which requires the sort dimension to be a power of 2.
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, Literal
import warnings

try:
    from . import _sort_ops_cuda
    # Import functions from the CUDA module
    _tensor_sort_bitonic_cuda = _sort_ops_cuda.tensor_sort_bitonic

    # Import low-level type-specific functions
    _tensor_sort_bitonic_float32_cuda = _sort_ops_cuda.tensor_sort_bitonic_float32
    _tensor_sort_bitonic_float64_cuda = _sort_ops_cuda.tensor_sort_bitonic_float64
    _tensor_sort_bitonic_int8_cuda = _sort_ops_cuda.tensor_sort_bitonic_int8
    _tensor_sort_bitonic_int16_cuda = _sort_ops_cuda.tensor_sort_bitonic_int16
    _tensor_sort_bitonic_int32_cuda = _sort_ops_cuda.tensor_sort_bitonic_int32
    _tensor_sort_bitonic_int64_cuda = _sort_ops_cuda.tensor_sort_bitonic_int64
    _tensor_sort_bitonic_uint8_cuda = _sort_ops_cuda.tensor_sort_bitonic_uint8
    _tensor_sort_bitonic_uint16_cuda = _sort_ops_cuda.tensor_sort_bitonic_uint16
    _tensor_sort_bitonic_uint32_cuda = _sort_ops_cuda.tensor_sort_bitonic_uint32
    _tensor_sort_bitonic_uint64_cuda = _sort_ops_cuda.tensor_sort_bitonic_uint64

    _CUDA_AVAILABLE = True
except ImportError as e:
    _CUDA_AVAILABLE = False
    warnings.warn(f"CUDA backend not available: {e}")

T = TypeVar('T', bound=np.generic)

# Type mapping for sort operations (all supported types)
_SORT_TYPE_DISPATCH_MAP = {
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

def _validate_tensor_sort_inputs(tensor: np.ndarray, sort_dim: str, operation_name: str) -> None:
    """Validate inputs for tensor sort operations."""
    if tensor.ndim != 3:
        raise ValueError(f"{operation_name}: Input array must be 3-dimensional")

    if tensor.dtype not in _SORT_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {tensor.dtype}")

    if sort_dim not in ("rows", "cols", "sheets"):
        raise ValueError(f"{operation_name}: sort_dim must be one of: 'rows', 'cols', 'sheets'")

    # Validate that sort dimension is power of 2 for bitonic sort
    if sort_dim == "rows":
        target_size = tensor.shape[0]
    elif sort_dim == "cols":
        target_size = tensor.shape[1]
    else:  # sort_dim == "sheets"
        target_size = tensor.shape[2]

    if (target_size & (target_size - 1)) != 0:
        raise ValueError(f"{operation_name}: Sort dimension size ({target_size}) must be a power of 2 for bitonic sort")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions for tensor_sort_bitonic

def tensor_sort_bitonic_float32(tensor: NDArray[np.float32], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for float32 arrays (in-place operation).

    Args:
        tensor: 3D tensor to sort in-place
        sort_dim: Dimension to sort along ("rows", "cols", or "sheets")

    Note:
        This operation modifies the input tensor in-place.
        The size of the sort dimension must be a power of 2.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_float32")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_float32_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_float64(tensor: NDArray[np.float64], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for float64 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_float64")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_float64_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_int8(tensor: NDArray[np.int8], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for int8 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_int8")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_int8_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_int16(tensor: NDArray[np.int16], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for int16 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_int16")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_int16_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_int32(tensor: NDArray[np.int32], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for int32 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_int32")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_int32_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_int64(tensor: NDArray[np.int64], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for int64 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_int64")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_int64_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_uint8(tensor: NDArray[np.uint8], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for uint8 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_uint8")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_uint8_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_uint16(tensor: NDArray[np.uint16], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for uint16 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_uint16")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_uint16_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_uint32(tensor: NDArray[np.uint32], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for uint32 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_uint32")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_uint32_cuda(tensor_contig, sort_dim)

def tensor_sort_bitonic_uint64(tensor: NDArray[np.uint64], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort for uint64 arrays (in-place operation)."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic_uint64")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_uint64_cuda(tensor_contig, sort_dim)

# High-level dispatch function

def tensor_sort_bitonic(tensor: Union[NDArray[T], np.ndarray], sort_dim: Literal["rows", "cols", "sheets"]) -> None:
    """3D tensor bitonic sort with automatic type dispatch (in-place operation).

    This function performs an in-place bitonic sort on a 3D tensor along the specified dimension.
    Bitonic sort is a comparison-based sorting algorithm that works efficiently on GPU hardware.

    Args:
        tensor: 3D tensor to sort in-place
        sort_dim: Dimension to sort along:
            - "rows": Sort along dimension 0 (first dimension)
            - "cols": Sort along dimension 1 (second dimension)
            - "sheets": Sort along dimension 2 (third dimension)

    Raises:
        RuntimeError: If CUDA backend is not available
        ValueError: If input validation fails or sort dimension size is not a power of 2

    Note:
        - This operation modifies the input tensor in-place
        - The size of the sort dimension must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, ...)
        - All numeric types are supported: float32, float64, int8-64, uint8-64

    Examples:
        >>> import numpy as np
        >>> tensor = np.random.randint(0, 100, (8, 4, 16), dtype=np.int32)  # 8x4x16 tensor
        >>> print("Original shape:", tensor.shape)
        >>> tensor_sort_bitonic(tensor, "rows")  # Sort along first dimension (size 8 = 2^3)
        >>> print("Sorted along rows in-place")
        >>>
        >>> # Sort along sheets dimension
        >>> tensor_sort_bitonic(tensor, "sheets")  # Sort along third dimension (size 16 = 2^4)
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_tensor_sort_inputs(tensor, sort_dim, "tensor_sort_bitonic")
    tensor_contig = _ensure_contiguous(tensor)

    _tensor_sort_bitonic_cuda(tensor_contig, sort_dim)

# Export all functions
__all__ = [
    # High-level dispatch function
    'tensor_sort_bitonic',

    # Low-level type-specific functions
    'tensor_sort_bitonic_float32',
    'tensor_sort_bitonic_float64',
    'tensor_sort_bitonic_int8',
    'tensor_sort_bitonic_int16',
    'tensor_sort_bitonic_int32',
    'tensor_sort_bitonic_int64',
    'tensor_sort_bitonic_uint8',
    'tensor_sort_bitonic_uint16',
    'tensor_sort_bitonic_uint32',
    'tensor_sort_bitonic_uint64',
]
