# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/vector_ops.py

"""
Vector operations module for py-gpu-algos

This module provides high-level Python interfaces for vector operations,
automatically dispatching to the appropriate CUDA backend based on array types.
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, overload, Literal
import warnings

try:
    from ._module_loader import get_cuda_module

    # Load the CUDA module from build directory
    _vector_ops_cuda = get_cuda_module('_vector_ops_cuda')

    if _vector_ops_cuda is not None:
        # Import functions from the CUDA module
        _vector_cumsum_serial_cuda = _vector_ops_cuda.vector_cumsum_serial
        _vector_cumsum_parallel_cuda = _vector_ops_cuda.vector_cumsum_parallel
        _vector_cummax_parallel_cuda = _vector_ops_cuda.vector_cummax_parallel
        _vector_scan_parallel_cuda = _vector_ops_cuda.vector_scan_parallel

        # Import low-level type-specific functions
        _vector_cumsum_serial_float32_cuda = _vector_ops_cuda.vector_cumsum_serial_float32
        _vector_cumsum_serial_float64_cuda = _vector_ops_cuda.vector_cumsum_serial_float64
        _vector_cumsum_serial_int8_cuda = _vector_ops_cuda.vector_cumsum_serial_int8
        _vector_cumsum_serial_int16_cuda = _vector_ops_cuda.vector_cumsum_serial_int16
        _vector_cumsum_serial_int32_cuda = _vector_ops_cuda.vector_cumsum_serial_int32
        _vector_cumsum_serial_int64_cuda = _vector_ops_cuda.vector_cumsum_serial_int64
        _vector_cumsum_serial_uint8_cuda = _vector_ops_cuda.vector_cumsum_serial_uint8
        _vector_cumsum_serial_uint16_cuda = _vector_ops_cuda.vector_cumsum_serial_uint16
        _vector_cumsum_serial_uint32_cuda = _vector_ops_cuda.vector_cumsum_serial_uint32
        _vector_cumsum_serial_uint64_cuda = _vector_ops_cuda.vector_cumsum_serial_uint64

        _vector_cumsum_parallel_float32_cuda = _vector_ops_cuda.vector_cumsum_parallel_float32
        _vector_cumsum_parallel_float64_cuda = _vector_ops_cuda.vector_cumsum_parallel_float64
        _vector_cumsum_parallel_int8_cuda = _vector_ops_cuda.vector_cumsum_parallel_int8
        _vector_cumsum_parallel_int16_cuda = _vector_ops_cuda.vector_cumsum_parallel_int16
        _vector_cumsum_parallel_int32_cuda = _vector_ops_cuda.vector_cumsum_parallel_int32
        _vector_cumsum_parallel_int64_cuda = _vector_ops_cuda.vector_cumsum_parallel_int64
        _vector_cumsum_parallel_uint8_cuda = _vector_ops_cuda.vector_cumsum_parallel_uint8
        _vector_cumsum_parallel_uint16_cuda = _vector_ops_cuda.vector_cumsum_parallel_uint16
        _vector_cumsum_parallel_uint32_cuda = _vector_ops_cuda.vector_cumsum_parallel_uint32
        _vector_cumsum_parallel_uint64_cuda = _vector_ops_cuda.vector_cumsum_parallel_uint64

        _vector_cummax_parallel_float32_cuda = _vector_ops_cuda.vector_cummax_parallel_float32
        _vector_cummax_parallel_float64_cuda = _vector_ops_cuda.vector_cummax_parallel_float64
        _vector_cummax_parallel_int8_cuda = _vector_ops_cuda.vector_cummax_parallel_int8
        _vector_cummax_parallel_int16_cuda = _vector_ops_cuda.vector_cummax_parallel_int16
        _vector_cummax_parallel_int32_cuda = _vector_ops_cuda.vector_cummax_parallel_int32
        _vector_cummax_parallel_int64_cuda = _vector_ops_cuda.vector_cummax_parallel_int64
        _vector_cummax_parallel_uint8_cuda = _vector_ops_cuda.vector_cummax_parallel_uint8
        _vector_cummax_parallel_uint16_cuda = _vector_ops_cuda.vector_cummax_parallel_uint16
        _vector_cummax_parallel_uint32_cuda = _vector_ops_cuda.vector_cummax_parallel_uint32
        _vector_cummax_parallel_uint64_cuda = _vector_ops_cuda.vector_cummax_parallel_uint64

        # Scan operations with specific operations
        _vector_scan_parallel_max_float32_cuda = _vector_ops_cuda.vector_scan_parallel_max_float32
        _vector_scan_parallel_max_float64_cuda = _vector_ops_cuda.vector_scan_parallel_max_float64
        _vector_scan_parallel_max_int8_cuda = _vector_ops_cuda.vector_scan_parallel_max_int8
        _vector_scan_parallel_max_int16_cuda = _vector_ops_cuda.vector_scan_parallel_max_int16
        _vector_scan_parallel_max_int32_cuda = _vector_ops_cuda.vector_scan_parallel_max_int32
        _vector_scan_parallel_max_int64_cuda = _vector_ops_cuda.vector_scan_parallel_max_int64
        _vector_scan_parallel_max_uint8_cuda = _vector_ops_cuda.vector_scan_parallel_max_uint8
        _vector_scan_parallel_max_uint16_cuda = _vector_ops_cuda.vector_scan_parallel_max_uint16
        _vector_scan_parallel_max_uint32_cuda = _vector_ops_cuda.vector_scan_parallel_max_uint32
        _vector_scan_parallel_max_uint64_cuda = _vector_ops_cuda.vector_scan_parallel_max_uint64

        _vector_scan_parallel_min_float32_cuda = _vector_ops_cuda.vector_scan_parallel_min_float32
        _vector_scan_parallel_min_float64_cuda = _vector_ops_cuda.vector_scan_parallel_min_float64
        _vector_scan_parallel_min_int8_cuda = _vector_ops_cuda.vector_scan_parallel_min_int8
        _vector_scan_parallel_min_int16_cuda = _vector_ops_cuda.vector_scan_parallel_min_int16
        _vector_scan_parallel_min_int32_cuda = _vector_ops_cuda.vector_scan_parallel_min_int32
        _vector_scan_parallel_min_int64_cuda = _vector_ops_cuda.vector_scan_parallel_min_int64
        _vector_scan_parallel_min_uint8_cuda = _vector_ops_cuda.vector_scan_parallel_min_uint8
        _vector_scan_parallel_min_uint16_cuda = _vector_ops_cuda.vector_scan_parallel_min_uint16
        _vector_scan_parallel_min_uint32_cuda = _vector_ops_cuda.vector_scan_parallel_min_uint32
        _vector_scan_parallel_min_uint64_cuda = _vector_ops_cuda.vector_scan_parallel_min_uint64

        _vector_scan_parallel_sum_float32_cuda = _vector_ops_cuda.vector_scan_parallel_sum_float32
        _vector_scan_parallel_sum_float64_cuda = _vector_ops_cuda.vector_scan_parallel_sum_float64
        _vector_scan_parallel_sum_int8_cuda = _vector_ops_cuda.vector_scan_parallel_sum_int8
        _vector_scan_parallel_sum_int16_cuda = _vector_ops_cuda.vector_scan_parallel_sum_int16
        _vector_scan_parallel_sum_int32_cuda = _vector_ops_cuda.vector_scan_parallel_sum_int32
        _vector_scan_parallel_sum_int64_cuda = _vector_ops_cuda.vector_scan_parallel_sum_int64
        _vector_scan_parallel_sum_uint8_cuda = _vector_ops_cuda.vector_scan_parallel_sum_uint8
        _vector_scan_parallel_sum_uint16_cuda = _vector_ops_cuda.vector_scan_parallel_sum_uint16
        _vector_scan_parallel_sum_uint32_cuda = _vector_ops_cuda.vector_scan_parallel_sum_uint32
        _vector_scan_parallel_sum_uint64_cuda = _vector_ops_cuda.vector_scan_parallel_sum_uint64

        _vector_scan_parallel_prod_float32_cuda = _vector_ops_cuda.vector_scan_parallel_prod_float32
        _vector_scan_parallel_prod_float64_cuda = _vector_ops_cuda.vector_scan_parallel_prod_float64
        _vector_scan_parallel_prod_int8_cuda = _vector_ops_cuda.vector_scan_parallel_prod_int8
        _vector_scan_parallel_prod_int16_cuda = _vector_ops_cuda.vector_scan_parallel_prod_int16
        _vector_scan_parallel_prod_int32_cuda = _vector_ops_cuda.vector_scan_parallel_prod_int32
        _vector_scan_parallel_prod_int64_cuda = _vector_ops_cuda.vector_scan_parallel_prod_int64
        _vector_scan_parallel_prod_uint8_cuda = _vector_ops_cuda.vector_scan_parallel_prod_uint8
        _vector_scan_parallel_prod_uint16_cuda = _vector_ops_cuda.vector_scan_parallel_prod_uint16
        _vector_scan_parallel_prod_uint32_cuda = _vector_ops_cuda.vector_scan_parallel_prod_uint32
        _vector_scan_parallel_prod_uint64_cuda = _vector_ops_cuda.vector_scan_parallel_prod_uint64

        _CUDA_AVAILABLE = True
    else:
        _CUDA_AVAILABLE = False
        warnings.warn("CUDA backend not available: Could not load _vector_ops_cuda module from build directory")

except Exception as e:
    _CUDA_AVAILABLE = False
    warnings.warn(f"CUDA backend not available: {e}")

T = TypeVar('T', bound=np.generic)

# Type mapping for scan operations (all supported types)
_SCAN_TYPE_DISPATCH_MAP = {
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


def _validate_vector_scan_inputs(a: np.ndarray, operation_name: str) -> None:
    """Validate inputs for vector scan operations (more restrictive type support)."""
    if a.ndim != 1:
        raise ValueError(f"{operation_name}: Input array must be 1-dimensional")

    if a.dtype not in _SCAN_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {a.dtype}. Scan operations support: float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions for cumsum_serial

def vector_cumsum_serial_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cumulative sum using serial algorithm for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_float32_cuda(a_contig)

def vector_cumsum_serial_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative sum using serial algorithm for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_float64_cuda(a_contig)

def vector_cumsum_serial_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Cumulative sum using serial algorithm for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_int8_cuda(a_contig)

def vector_cumsum_serial_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Cumulative sum using serial algorithm for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_int16_cuda(a_contig)

def vector_cumsum_serial_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Cumulative sum using serial algorithm for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_int32_cuda(a_contig)

def vector_cumsum_serial_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Cumulative sum using serial algorithm for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_int64_cuda(a_contig)

def vector_cumsum_serial_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Cumulative sum using serial algorithm for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_uint8_cuda(a_contig)

def vector_cumsum_serial_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Cumulative sum using serial algorithm for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_uint16_cuda(a_contig)

def vector_cumsum_serial_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Cumulative sum using serial algorithm for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_uint32_cuda(a_contig)

def vector_cumsum_serial_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Cumulative sum using serial algorithm for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_uint64_cuda(a_contig)

# Low-level type-specific functions for cumsum_parallel

def vector_cumsum_parallel_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cumulative sum using parallel algorithm for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_float32_cuda(a_contig)

def vector_cumsum_parallel_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative sum using parallel algorithm for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_float64_cuda(a_contig)

def vector_cumsum_parallel_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Cumulative sum using parallel algorithm for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_int8_cuda(a_contig)

def vector_cumsum_parallel_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Cumulative sum using parallel algorithm for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_int16_cuda(a_contig)

def vector_cumsum_parallel_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Cumulative sum using parallel algorithm for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_int32_cuda(a_contig)

def vector_cumsum_parallel_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Cumulative sum using parallel algorithm for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_int64_cuda(a_contig)

def vector_cumsum_parallel_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Cumulative sum using parallel algorithm for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_uint8_cuda(a_contig)

def vector_cumsum_parallel_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Cumulative sum using parallel algorithm for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_uint16_cuda(a_contig)

def vector_cumsum_parallel_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Cumulative sum using parallel algorithm for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_uint32_cuda(a_contig)

def vector_cumsum_parallel_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Cumulative sum using parallel algorithm for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_uint64_cuda(a_contig)

# Low-level type-specific functions for cummax_parallel

def vector_cummax_parallel_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cumulative maximum using parallel algorithm for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_float32_cuda(a_contig)

def vector_cummax_parallel_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative maximum using parallel algorithm for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_float64_cuda(a_contig)

def vector_cummax_parallel_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Cumulative maximum using parallel algorithm for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_int8_cuda(a_contig)

def vector_cummax_parallel_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Cumulative maximum using parallel algorithm for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_int16_cuda(a_contig)

def vector_cummax_parallel_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Cumulative maximum using parallel algorithm for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_int32_cuda(a_contig)

def vector_cummax_parallel_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Cumulative maximum using parallel algorithm for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_int64_cuda(a_contig)

def vector_cummax_parallel_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Cumulative maximum using parallel algorithm for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_uint8_cuda(a_contig)

def vector_cummax_parallel_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Cumulative maximum using parallel algorithm for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_uint16_cuda(a_contig)

def vector_cummax_parallel_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Cumulative maximum using parallel algorithm for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_uint32_cuda(a_contig)

def vector_cummax_parallel_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Cumulative maximum using parallel algorithm for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_uint64_cuda(a_contig)

# Low-level type-specific functions for scan_parallel with specific operations

def vector_scan_parallel_max_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Parallel scan with max operation for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_float32_cuda(a_contig)

def vector_scan_parallel_max_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parallel scan with max operation for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_float64_cuda(a_contig)

def vector_scan_parallel_max_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Parallel scan with max operation for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_int8_cuda(a_contig)

def vector_scan_parallel_max_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Parallel scan with max operation for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_int16_cuda(a_contig)

def vector_scan_parallel_max_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Parallel scan with max operation for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_int32_cuda(a_contig)

def vector_scan_parallel_max_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Parallel scan with max operation for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_int64_cuda(a_contig)

def vector_scan_parallel_max_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Parallel scan with max operation for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_uint8_cuda(a_contig)

def vector_scan_parallel_max_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Parallel scan with max operation for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_uint16_cuda(a_contig)

def vector_scan_parallel_max_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Parallel scan with max operation for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_uint32_cuda(a_contig)

def vector_scan_parallel_max_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Parallel scan with max operation for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_max_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_max_uint64_cuda(a_contig)

def vector_scan_parallel_min_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Parallel scan with min operation for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_float32_cuda(a_contig)

def vector_scan_parallel_min_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parallel scan with min operation for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_float64_cuda(a_contig)

def vector_scan_parallel_min_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Parallel scan with min operation for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_int8_cuda(a_contig)

def vector_scan_parallel_min_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Parallel scan with min operation for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_int16_cuda(a_contig)

def vector_scan_parallel_min_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Parallel scan with min operation for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_int32_cuda(a_contig)

def vector_scan_parallel_min_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Parallel scan with min operation for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_int64_cuda(a_contig)

def vector_scan_parallel_min_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Parallel scan with min operation for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_uint8_cuda(a_contig)

def vector_scan_parallel_min_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Parallel scan with min operation for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_uint16_cuda(a_contig)

def vector_scan_parallel_min_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Parallel scan with min operation for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_uint32_cuda(a_contig)

def vector_scan_parallel_min_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Parallel scan with min operation for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_min_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_min_uint64_cuda(a_contig)

def vector_scan_parallel_sum_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Parallel scan with sum operation for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_float32_cuda(a_contig)

def vector_scan_parallel_sum_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parallel scan with sum operation for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_float64_cuda(a_contig)

def vector_scan_parallel_sum_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Parallel scan with sum operation for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_int8_cuda(a_contig)

def vector_scan_parallel_sum_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Parallel scan with sum operation for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_int16_cuda(a_contig)

def vector_scan_parallel_sum_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Parallel scan with sum operation for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_int32_cuda(a_contig)

def vector_scan_parallel_sum_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Parallel scan with sum operation for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_int64_cuda(a_contig)

def vector_scan_parallel_sum_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Parallel scan with sum operation for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_uint8_cuda(a_contig)

def vector_scan_parallel_sum_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Parallel scan with sum operation for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_uint16_cuda(a_contig)

def vector_scan_parallel_sum_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Parallel scan with sum operation for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_uint32_cuda(a_contig)

def vector_scan_parallel_sum_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Parallel scan with sum operation for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_sum_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_sum_uint64_cuda(a_contig)

def vector_scan_parallel_prod_float32(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Parallel scan with prod operation for float32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_float32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_float32_cuda(a_contig)

def vector_scan_parallel_prod_float64(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parallel scan with prod operation for float64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_float64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_float64_cuda(a_contig)

def vector_scan_parallel_prod_int8(a: NDArray[np.int8]) -> NDArray[np.int8]:
    """Parallel scan with prod operation for int8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_int8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_int8_cuda(a_contig)

def vector_scan_parallel_prod_int16(a: NDArray[np.int16]) -> NDArray[np.int16]:
    """Parallel scan with prod operation for int16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_int16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_int16_cuda(a_contig)

def vector_scan_parallel_prod_int32(a: NDArray[np.int32]) -> NDArray[np.int32]:
    """Parallel scan with prod operation for int32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_int32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_int32_cuda(a_contig)

def vector_scan_parallel_prod_int64(a: NDArray[np.int64]) -> NDArray[np.int64]:
    """Parallel scan with prod operation for int64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_int64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_int64_cuda(a_contig)

def vector_scan_parallel_prod_uint8(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Parallel scan with prod operation for uint8 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_uint8")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_uint8_cuda(a_contig)

def vector_scan_parallel_prod_uint16(a: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Parallel scan with prod operation for uint16 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_uint16")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_uint16_cuda(a_contig)

def vector_scan_parallel_prod_uint32(a: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Parallel scan with prod operation for uint32 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_uint32")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_uint32_cuda(a_contig)

def vector_scan_parallel_prod_uint64(a: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Parallel scan with prod operation for uint64 arrays."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel_prod_uint64")
    a_contig = _ensure_contiguous(a)

    return _vector_scan_parallel_prod_uint64_cuda(a_contig)

# High-level dispatch functions

@overload
def vector_cumsum_serial(a: NDArray[T]) -> NDArray[T]: ...

def vector_cumsum_serial(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Cumulative sum using serial algorithm with automatic type dispatch.

    This function automatically selects the appropriate low-level implementation
    based on the dtype of the input array.

    Args:
        a: Input vector of shape (n,)

    Returns:
        Result vector of shape (n,) where result[i] = sum(a[0:i+1])

    Raises:
        RuntimeError: If CUDA backend is not available
        ValueError: If input validation fails or unsupported dtype

    Examples:
        >>> import numpy as np
        >>> a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        >>> result = vector_cumsum_serial(a)
        >>> print(result)
        [1. 3. 6. 10. 15.]
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_serial")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_serial_cuda(a_contig)

@overload
def vector_cumsum_parallel(a: NDArray[T]) -> NDArray[T]: ...

def vector_cumsum_parallel(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Cumulative sum using parallel algorithm with automatic type dispatch.

    Args:
        a: Input vector of shape (n,)

    Returns:
        Result vector of shape (n,) where result[i] = sum(a[0:i+1])
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cumsum_parallel")
    a_contig = _ensure_contiguous(a)

    return _vector_cumsum_parallel_cuda(a_contig)

@overload
def vector_cummax_parallel(a: NDArray[T]) -> NDArray[T]: ...

def vector_cummax_parallel(a: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """Cumulative maximum using parallel algorithm with automatic type dispatch.

    Args:
        a: Input vector of shape (n,)

    Returns:
        Result vector of shape (n,) where result[i] = max(a[0:i+1])
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_cummax_parallel")
    a_contig = _ensure_contiguous(a)

    return _vector_cummax_parallel_cuda(a_contig)

@overload
def vector_scan_parallel(a: NDArray[T], operation: Literal["max", "min", "sum", "prod"]) -> NDArray[T]: ...

def vector_scan_parallel(a: Union[NDArray[T], np.ndarray], operation: Literal["max", "min", "sum", "prod"]) -> Union[NDArray[T], np.ndarray]:
    """Parallel scan operation with automatic type dispatch.

    Args:
        a: Input vector of shape (n,)
        operation: Scan operation to perform ("max", "min", "sum", "prod")

    Returns:
        Result vector of shape (n,) where result[i] = operation(a[0:i+1])

    Note:
        This function supports a reduced set of types: float32, float64, int32, int64
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend not available")

    _validate_vector_scan_inputs(a, "vector_scan_parallel")
    a_contig = _ensure_contiguous(a)

    if operation not in ["max", "min", "sum", "prod"]:
        raise ValueError(f"Unsupported operation: {operation}. Must be one of: max, min, sum, prod")

    return _vector_scan_parallel_cuda(a_contig, operation)

# Export all functions
__all__ = [
    # High-level dispatch functions
    'vector_cumsum_serial',
    'vector_cumsum_parallel',
    'vector_cummax_parallel',
    'vector_scan_parallel',

    # Low-level type-specific functions for cumsum_serial
    'vector_cumsum_serial_float32',
    'vector_cumsum_serial_float64',
    'vector_cumsum_serial_int8',
    'vector_cumsum_serial_int16',
    'vector_cumsum_serial_int32',
    'vector_cumsum_serial_int64',
    'vector_cumsum_serial_uint8',
    'vector_cumsum_serial_uint16',
    'vector_cumsum_serial_uint32',
    'vector_cumsum_serial_uint64',

    # Low-level type-specific functions for cumsum_parallel
    'vector_cumsum_parallel_float32',
    'vector_cumsum_parallel_float64',
    'vector_cumsum_parallel_int8',
    'vector_cumsum_parallel_int16',
    'vector_cumsum_parallel_int32',
    'vector_cumsum_parallel_int64',
    'vector_cumsum_parallel_uint8',
    'vector_cumsum_parallel_uint16',
    'vector_cumsum_parallel_uint32',
    'vector_cumsum_parallel_uint64',

    # Low-level type-specific functions for cummax_parallel
    'vector_cummax_parallel_float32',
    'vector_cummax_parallel_float64',
    'vector_cummax_parallel_int8',
    'vector_cummax_parallel_int16',
    'vector_cummax_parallel_int32',
    'vector_cummax_parallel_int64',
    'vector_cummax_parallel_uint8',
    'vector_cummax_parallel_uint16',
    'vector_cummax_parallel_uint32',
    'vector_cummax_parallel_uint64',

    # Low-level type-specific functions for scan_parallel
    'vector_scan_parallel_max_float32',
    'vector_scan_parallel_max_float64',
    'vector_scan_parallel_max_int8',
    'vector_scan_parallel_max_int16',
    'vector_scan_parallel_max_int32',
    'vector_scan_parallel_max_int64',
    'vector_scan_parallel_max_uint8',
    'vector_scan_parallel_max_uint16',
    'vector_scan_parallel_max_uint32',
    'vector_scan_parallel_max_uint64',
    'vector_scan_parallel_min_float32',
    'vector_scan_parallel_min_float64',
    'vector_scan_parallel_min_int8',
    'vector_scan_parallel_min_int16',
    'vector_scan_parallel_min_int32',
    'vector_scan_parallel_min_int64',
    'vector_scan_parallel_min_uint8',
    'vector_scan_parallel_min_uint16',
    'vector_scan_parallel_min_uint32',
    'vector_scan_parallel_min_uint64',
    'vector_scan_parallel_sum_float32',
    'vector_scan_parallel_sum_float64',
    'vector_scan_parallel_sum_int8',
    'vector_scan_parallel_sum_int16',
    'vector_scan_parallel_sum_int32',
    'vector_scan_parallel_sum_int64',
    'vector_scan_parallel_sum_uint8',
    'vector_scan_parallel_sum_uint16',
    'vector_scan_parallel_sum_uint32',
    'vector_scan_parallel_sum_uint64',
    'vector_scan_parallel_prod_float32',
    'vector_scan_parallel_prod_float64',
    'vector_scan_parallel_prod_int8',
    'vector_scan_parallel_prod_int16',
    'vector_scan_parallel_prod_int32',
    'vector_scan_parallel_prod_int64',
    'vector_scan_parallel_prod_uint8',
    'vector_scan_parallel_prod_uint16',
    'vector_scan_parallel_prod_uint32',
    'vector_scan_parallel_prod_uint64',
]
