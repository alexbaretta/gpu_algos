# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/glm_ops.py

"""
GLM operations module for py-gpu-algos

This module provides high-level Python interfaces for GLM (Generalized Linear Model) operations,
automatically dispatching to the appropriate CUDA backend based on array types.

GLM operations work with 3D tensors representing:
- X: Features tensor (nfeatures, ntasks, nobs)
- Y: Targets tensor (ntargets, ntasks, nobs)
- M: Model tensor (nfeatures, ntargets, ntasks)
"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Union, overload
import warnings

try:
    from ._module_loader import get_cuda_module

    # Load the CUDA module from build directory
    _glm_ops_cuda = get_cuda_module('_glm_ops_cuda')
    # Import functions from the CUDA module
    _glm_predict_naive_cuda = _glm_ops_cuda.glm_predict_naive
    _glm_gradient_naive_cuda = _glm_ops_cuda.glm_gradient_naive
    _glm_gradient_xyyhat_cuda = _glm_ops_cuda.glm_gradient_xyyhat

    # Import low-level type-specific functions
    _glm_predict_naive_float32_cuda = _glm_ops_cuda.glm_predict_naive_float32
    _glm_predict_naive_float64_cuda = _glm_ops_cuda.glm_predict_naive_float64
    _glm_predict_naive_int32_cuda = _glm_ops_cuda.glm_predict_naive_int32
    _glm_predict_naive_int64_cuda = _glm_ops_cuda.glm_predict_naive_int64

    _glm_gradient_naive_float32_cuda = _glm_ops_cuda.glm_gradient_naive_float32
    _glm_gradient_naive_float64_cuda = _glm_ops_cuda.glm_gradient_naive_float64
    _glm_gradient_naive_int32_cuda = _glm_ops_cuda.glm_gradient_naive_int32
    _glm_gradient_naive_int64_cuda = _glm_ops_cuda.glm_gradient_naive_int64

    _glm_gradient_xyyhat_float32_cuda = _glm_ops_cuda.glm_gradient_xyyhat_float32
    _glm_gradient_xyyhat_float64_cuda = _glm_ops_cuda.glm_gradient_xyyhat_float64
    _glm_gradient_xyyhat_int32_cuda = _glm_ops_cuda.glm_gradient_xyyhat_int32
    _glm_gradient_xyyhat_int64_cuda = _glm_ops_cuda.glm_gradient_xyyhat_int64
except Exception as e:
    raise e

T = TypeVar('T', bound=np.generic)

# Type mapping for GLM operations (reduced set of supported types)
_GLM_TYPE_DISPATCH_MAP = {
    np.dtype(np.float32): 'float32',
    np.dtype(np.float64): 'float64',
    np.dtype(np.int32): 'int32',
    np.dtype(np.int64): 'int64',
}

def _validate_glm_predict_inputs(X: np.ndarray, M: np.ndarray, operation_name: str) -> None:
    """Validate inputs for GLM predict operations."""
    if X.ndim != 3 or M.ndim != 3:
        raise ValueError(f"{operation_name}: Input arrays must be 3-dimensional")

    if X.dtype != M.dtype:
        raise ValueError(f"{operation_name}: Input arrays must have the same dtype")

    if X.dtype not in _GLM_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {X.dtype}. GLM operations support: float32, float64, int32, int64")

    # Validate tensor dimensions for prediction
    nfeatures_X, ntasks_X, nobs_X = X.shape
    nfeatures_M, ntargets_M, ntasks_M = M.shape

    if nfeatures_X != nfeatures_M:
        raise ValueError(f"{operation_name}: Feature dimensions must match: X({nfeatures_X}) vs M({nfeatures_M})")

    if ntasks_X != ntasks_M:
        raise ValueError(f"{operation_name}: Task dimensions must match: X({ntasks_X}) vs M({ntasks_M})")

def _validate_glm_gradient_inputs(X: np.ndarray, Y: np.ndarray, M: np.ndarray, operation_name: str) -> None:
    """Validate inputs for GLM gradient operations."""
    if X.ndim != 3 or Y.ndim != 3 or M.ndim != 3:
        raise ValueError(f"{operation_name}: Input arrays must be 3-dimensional")

    if X.dtype != Y.dtype or X.dtype != M.dtype:
        raise ValueError(f"{operation_name}: All input arrays must have the same dtype")

    if X.dtype not in _GLM_TYPE_DISPATCH_MAP:
        raise ValueError(f"{operation_name}: Unsupported dtype {X.dtype}. GLM operations support: float32, float64, int32, int64")

    # Validate tensor dimensions for gradient computation
    nfeatures_X, ntasks_X, nobs_X = X.shape
    ntargets_Y, ntasks_Y, nobs_Y = Y.shape
    nfeatures_M, ntargets_M, ntasks_M = M.shape

    if ntasks_X != ntasks_Y or nobs_X != nobs_Y:
        raise ValueError(f"{operation_name}: X and Y must have compatible task/obs dimensions: "
                        f"X({ntasks_X}, {nobs_X}) vs Y({ntasks_Y}, {nobs_Y})")

    if nfeatures_X != nfeatures_M or ntargets_Y != ntargets_M or ntasks_X != ntasks_M:
        raise ValueError(f"{operation_name}: Tensor dimensions must be compatible: "
                        f"X({nfeatures_X}, {ntasks_X}), Y({ntargets_Y}, {ntasks_Y}), M({nfeatures_M}, {ntargets_M}, {ntasks_M})")

def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr

# Low-level type-specific functions for glm_predict_naive

def glm_predict_naive_float32(X: NDArray[np.float32], M: NDArray[np.float32]) -> NDArray[np.float32]:
    """GLM prediction using naive algorithm for float32 arrays.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Predictions tensor of shape (ntargets, ntasks, nobs)
    """
    _validate_glm_predict_inputs(X, M, "glm_predict_naive_float32")
    X_contig = _ensure_contiguous(X)
    M_contig = _ensure_contiguous(M)

    return _glm_predict_naive_float32_cuda(X_contig, M_contig)

def glm_predict_naive_float64(X: NDArray[np.float64], M: NDArray[np.float64]) -> NDArray[np.float64]:
    """GLM prediction using naive algorithm for float64 arrays."""
    _validate_glm_predict_inputs(X, M, "glm_predict_naive_float64")
    X_contig = _ensure_contiguous(X)
    M_contig = _ensure_contiguous(M)

    return _glm_predict_naive_float64_cuda(X_contig, M_contig)

def glm_predict_naive_int32(X: NDArray[np.int32], M: NDArray[np.int32]) -> NDArray[np.int32]:
    """GLM prediction using naive algorithm for int32 arrays."""
    _validate_glm_predict_inputs(X, M, "glm_predict_naive_int32")
    X_contig = _ensure_contiguous(X)
    M_contig = _ensure_contiguous(M)

    return _glm_predict_naive_int32_cuda(X_contig, M_contig)

def glm_predict_naive_int64(X: NDArray[np.int64], M: NDArray[np.int64]) -> NDArray[np.int64]:
    """GLM prediction using naive algorithm for int64 arrays."""
    _validate_glm_predict_inputs(X, M, "glm_predict_naive_int64")
    X_contig = _ensure_contiguous(X)
    M_contig = _ensure_contiguous(M)

    return _glm_predict_naive_int64_cuda(X_contig, M_contig)

# Low-level type-specific functions for glm_gradient_naive

def glm_gradient_naive_float32(X: NDArray[np.float32], Y: NDArray[np.float32], M: NDArray[np.float32]) -> NDArray[np.float32]:
    """GLM gradient computation using naive algorithm for float32 arrays.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        Y: Targets tensor of shape (ntargets, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Gradient tensor of shape (nfeatures, ntargets, ntasks)
    """
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_naive_float32")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_naive_float32_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_naive_float64(X: NDArray[np.float64], Y: NDArray[np.float64], M: NDArray[np.float64]) -> NDArray[np.float64]:
    """GLM gradient computation using naive algorithm for float64 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_naive_float64")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_naive_float64_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_naive_int32(X: NDArray[np.int32], Y: NDArray[np.int32], M: NDArray[np.int32]) -> NDArray[np.int32]:
    """GLM gradient computation using naive algorithm for int32 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_naive_int32")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_naive_int32_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_naive_int64(X: NDArray[np.int64], Y: NDArray[np.int64], M: NDArray[np.int64]) -> NDArray[np.int64]:
    """GLM gradient computation using naive algorithm for int64 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_naive_int64")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_naive_int64_cuda(X_contig, Y_contig, M_contig)

# Low-level type-specific functions for glm_gradient_xyyhat

def glm_gradient_xyyhat_float32(X: NDArray[np.float32], Y: NDArray[np.float32], M: NDArray[np.float32]) -> NDArray[np.float32]:
    """GLM gradient computation using XYYhat algorithm for float32 arrays.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        Y: Targets tensor of shape (ntargets, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Gradient tensor of shape (nfeatures, ntargets, ntasks)
    """
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_xyyhat_float32")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_xyyhat_float32_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_xyyhat_float64(X: NDArray[np.float64], Y: NDArray[np.float64], M: NDArray[np.float64]) -> NDArray[np.float64]:
    """GLM gradient computation using XYYhat algorithm for float64 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_xyyhat_float64")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_xyyhat_float64_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_xyyhat_int32(X: NDArray[np.int32], Y: NDArray[np.int32], M: NDArray[np.int32]) -> NDArray[np.int32]:
    """GLM gradient computation using XYYhat algorithm for int32 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_xyyhat_int32")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_xyyhat_int32_cuda(X_contig, Y_contig, M_contig)

def glm_gradient_xyyhat_int64(X: NDArray[np.int64], Y: NDArray[np.int64], M: NDArray[np.int64]) -> NDArray[np.int64]:
    """GLM gradient computation using XYYhat algorithm for int64 arrays."""
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_xyyhat_int64")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_xyyhat_int64_cuda(X_contig, Y_contig, M_contig)

# High-level dispatch functions

@overload
def glm_predict_naive(X: NDArray[T], M: NDArray[T]) -> NDArray[T]: ...

def glm_predict_naive(X: Union[NDArray[T], np.ndarray], M: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """GLM prediction using naive algorithm with automatic type dispatch.

    This function computes predictions using a linear model for multitask learning:
    Ŷ[target, task, obs] = SUM_feature M[feature, target, task] * X[feature, task, obs]

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Predictions tensor of shape (ntargets, ntasks, nobs)

    Raises:
        RuntimeError: If CUDA backend is not available
        ValueError: If input validation fails or unsupported dtype

    Note:
        This function supports a reduced set of types: float32, float64, int32, int64

    Examples:
        >>> import numpy as np
        >>> X = np.random.randn(10, 5, 100).astype(np.float32)  # 10 features, 5 tasks, 100 obs
        >>> M = np.random.randn(10, 3, 5).astype(np.float32)    # 10 features, 3 targets, 5 tasks
        >>> Yhat = glm_predict_naive(X, M)
        >>> print(Yhat.shape)
        (3, 5, 100)  # 3 targets, 5 tasks, 100 obs
    """
    _validate_glm_predict_inputs(X, M, "glm_predict_naive")
    X_contig = _ensure_contiguous(X)
    M_contig = _ensure_contiguous(M)

    return _glm_predict_naive_cuda(X_contig, M_contig)

@overload
def glm_gradient_naive(X: NDArray[T], Y: NDArray[T], M: NDArray[T]) -> NDArray[T]: ...

def glm_gradient_naive(X: Union[NDArray[T], np.ndarray], Y: Union[NDArray[T], np.ndarray], M: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """GLM gradient computation using naive algorithm with automatic type dispatch.

    This function computes the gradient of the squared error loss function for linear regression:
    grad_M[feature, target, task] = 2 * SUM_obs (Ŷ[target, task, obs] - Y[target, task, obs]) * X[feature, task, obs]

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        Y: Targets tensor of shape (ntargets, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Gradient tensor of shape (nfeatures, ntargets, ntasks)

    Note:
        This function supports a reduced set of types: float32, float64, int32, int64
    """
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_naive")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_naive_cuda(X_contig, Y_contig, M_contig)

@overload
def glm_gradient_xyyhat(X: NDArray[T], Y: NDArray[T], M: NDArray[T]) -> NDArray[T]: ...

def glm_gradient_xyyhat(X: Union[NDArray[T], np.ndarray], Y: Union[NDArray[T], np.ndarray], M: Union[NDArray[T], np.ndarray]) -> Union[NDArray[T], np.ndarray]:
    """GLM gradient computation using optimized XYYhat algorithm with automatic type dispatch.

    This function computes the same gradient as glm_gradient_naive but uses an optimized
    implementation that precomputes predictions (Yhat) for better performance.

    Args:
        X: Features tensor of shape (nfeatures, ntasks, nobs)
        Y: Targets tensor of shape (ntargets, ntasks, nobs)
        M: Model tensor of shape (nfeatures, ntargets, ntasks)

    Returns:
        Gradient tensor of shape (nfeatures, ntargets, ntasks)

    Note:
        This function supports a reduced set of types: float32, float64, int32, int64
    """
    _validate_glm_gradient_inputs(X, Y, M, "glm_gradient_xyyhat")
    X_contig = _ensure_contiguous(X)
    Y_contig = _ensure_contiguous(Y)
    M_contig = _ensure_contiguous(M)

    return _glm_gradient_xyyhat_cuda(X_contig, Y_contig, M_contig)

# Export all functions
__all__ = [
    # High-level dispatch functions
    'glm_predict_naive',
    'glm_gradient_naive',
    'glm_gradient_xyyhat',

    # Low-level type-specific functions for glm_predict_naive
    'glm_predict_naive_float32',
    'glm_predict_naive_float64',
    'glm_predict_naive_int32',
    'glm_predict_naive_int64',

    # Low-level type-specific functions for glm_gradient_naive
    'glm_gradient_naive_float32',
    'glm_gradient_naive_float64',
    'glm_gradient_naive_int32',
    'glm_gradient_naive_int64',

    # Low-level type-specific functions for glm_gradient_xyyhat
    'glm_gradient_xyyhat_float32',
    'glm_gradient_xyyhat_float64',
    'glm_gradient_xyyhat_int32',
    'glm_gradient_xyyhat_int64',
]
