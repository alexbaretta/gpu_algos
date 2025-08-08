# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/__init__.py

"""
py-gpu-algos: Python bindings for GPU algorithms

This package provides Python bindings for CUDA and HIP GPU kernels,
exposing them as functions operating on NumPy arrays.
"""

from .vector_ops import *
from .matrix_ops import *
from .glm_ops import *
from .sort_ops import *

# from .vector_ops import (
#     # High-level dispatch functions
#     vector_cumsum_serial,
#     vector_cumsum_parallel,
#     vector_cummax_parallel,
#     vector_scan_parallel,

#     # Low-level type-specific functions for cumsum_serial
#     vector_cumsum_serial_float32,
#     vector_cumsum_serial_float64,
#     vector_cumsum_serial_int8,
#     vector_cumsum_serial_int16,
#     vector_cumsum_serial_int32,
#     vector_cumsum_serial_int64,
#     vector_cumsum_serial_uint8,
#     vector_cumsum_serial_uint16,
#     vector_cumsum_serial_uint32,
#     vector_cumsum_serial_uint64,

#     # Low-level type-specific functions for cumsum_parallel
#     vector_cumsum_parallel_float32,
#     vector_cumsum_parallel_float64,
#     vector_cumsum_parallel_int8,
#     vector_cumsum_parallel_int16,
#     vector_cumsum_parallel_int32,
#     vector_cumsum_parallel_int64,
#     vector_cumsum_parallel_uint8,
#     vector_cumsum_parallel_uint16,
#     vector_cumsum_parallel_uint32,
#     vector_cumsum_parallel_uint64,

#     # Low-level type-specific functions for cummax_parallel
#     vector_cummax_parallel_float32,
#     vector_cummax_parallel_float64,
#     vector_cummax_parallel_int8,
#     vector_cummax_parallel_int16,
#     vector_cummax_parallel_int32,
#     vector_cummax_parallel_int64,
#     vector_cummax_parallel_uint8,
#     vector_cummax_parallel_uint16,
#     vector_cummax_parallel_uint32,
#     vector_cummax_parallel_uint64,

#     # Low-level type-specific functions for scan_parallel
#     vector_scan_parallel_max_float32,
#     vector_scan_parallel_max_float64,
#     vector_scan_parallel_max_int32,
#     vector_scan_parallel_max_int64,
#     vector_scan_parallel_min_float32,
#     vector_scan_parallel_min_float64,
#     vector_scan_parallel_min_int32,
#     vector_scan_parallel_min_int64,
#     vector_scan_parallel_sum_float32,
#     vector_scan_parallel_sum_float64,
#     vector_scan_parallel_sum_int32,
#     vector_scan_parallel_sum_int64,
#     vector_scan_parallel_prod_float32,
#     vector_scan_parallel_prod_float64,
#     vector_scan_parallel_prod_int32,
#     vector_scan_parallel_prod_int64,
# )

# from .matrix_ops import (
#     # Matrix product naive
#     matrix_product_naive,
#     # matrix_product_naive_float16,  # TODO: Handle float16 properly
#     matrix_product_naive_float32,
#     matrix_product_naive_float64,
#     matrix_product_naive_int8,
#     matrix_product_naive_int16,
#     matrix_product_naive_int32,
#     matrix_product_naive_int64,
#     matrix_product_naive_uint8,
#     matrix_product_naive_uint16,
#     matrix_product_naive_uint32,
#     matrix_product_naive_uint64,

#     # Matrix product tiled
#     matrix_product_tiled,
#     matrix_product_tiled_float32,
#     matrix_product_tiled_float64,
#     matrix_product_tiled_int32,
#     matrix_product_tiled_int64,

#     # Matrix transpose striped
#     matrix_transpose_striped,
#     matrix_transpose_striped_float32,
#     matrix_transpose_striped_float64,
#     matrix_transpose_striped_int8,
#     matrix_transpose_striped_int16,
#     matrix_transpose_striped_int32,
#     matrix_transpose_striped_int64,
#     matrix_transpose_striped_uint8,
#     matrix_transpose_striped_uint16,
#     matrix_transpose_striped_uint32,
#     matrix_transpose_striped_uint64,
# )

# from .glm_ops import (
#     # High-level dispatch functions
#     glm_predict_naive,
#     glm_gradient_naive,
#     glm_gradient_xyyhat,

#     # Low-level type-specific functions for glm_predict_naive
#     glm_predict_naive_float32,
#     glm_predict_naive_float64,
#     glm_predict_naive_int32,
#     glm_predict_naive_int64,

#     # Low-level type-specific functions for glm_gradient_naive
#     glm_gradient_naive_float32,
#     glm_gradient_naive_float64,
#     glm_gradient_naive_int32,
#     glm_gradient_naive_int64,

#     # Low-level type-specific functions for glm_gradient_xyyhat
#     glm_gradient_xyyhat_float32,
#     glm_gradient_xyyhat_float64,
#     glm_gradient_xyyhat_int32,
#     glm_gradient_xyyhat_int64,
# )

# from .sort_ops import (
#     # High-level dispatch function
#     tensor_sort_bitonic,

#     # Low-level type-specific functions
#     tensor_sort_bitonic_float32,
#     tensor_sort_bitonic_float64,
#     tensor_sort_bitonic_int8,
#     tensor_sort_bitonic_int16,
#     tensor_sort_bitonic_int32,
#     tensor_sort_bitonic_int64,
#     tensor_sort_bitonic_uint8,
#     tensor_sort_bitonic_uint16,
#     tensor_sort_bitonic_uint32,
#     tensor_sort_bitonic_uint64,
# )

__version__ = "0.1.0"
# __all__ = [
#     # Matrix operations - product naive
#     "matrix_product_naive",
#     # "matrix_product_naive_float16",  # TODO: Handle float16 properly
#     "matrix_product_naive_float32",
#     "matrix_product_naive_float64",
#     "matrix_product_naive_int8",
#     "matrix_product_naive_int16",
#     "matrix_product_naive_int32",
#     "matrix_product_naive_int64",
#     "matrix_product_naive_uint8",
#     "matrix_product_naive_uint16",
#     "matrix_product_naive_uint32",
#     "matrix_product_naive_uint64",

#     # Matrix operations - product tiled
#     "matrix_product_tiled",
#     "matrix_product_tiled_float32",
#     "matrix_product_tiled_float64",
#     "matrix_product_tiled_int32",
#     "matrix_product_tiled_int64",

#     # Matrix operations - transpose striped
#     "matrix_transpose_striped",
#     "matrix_transpose_striped_float32",
#     "matrix_transpose_striped_float64",
#     "matrix_transpose_striped_int8",
#     "matrix_transpose_striped_int16",
#     "matrix_transpose_striped_int32",
#     "matrix_transpose_striped_int64",
#     "matrix_transpose_striped_uint8",
#     "matrix_transpose_striped_uint16",
#     "matrix_transpose_striped_uint32",
#     "matrix_transpose_striped_uint64",

#     # Vector operations - high-level dispatch functions
#     "vector_cumsum_serial",
#     "vector_cumsum_parallel",
#     "vector_cummax_parallel",
#     "vector_scan_parallel",

#     # Vector operations - low-level type-specific functions for cumsum_serial
#     "vector_cumsum_serial_float32",
#     "vector_cumsum_serial_float64",
#     "vector_cumsum_serial_int8",
#     "vector_cumsum_serial_int16",
#     "vector_cumsum_serial_int32",
#     "vector_cumsum_serial_int64",
#     "vector_cumsum_serial_uint8",
#     "vector_cumsum_serial_uint16",
#     "vector_cumsum_serial_uint32",
#     "vector_cumsum_serial_uint64",

#     # Vector operations - low-level type-specific functions for cumsum_parallel
#     "vector_cumsum_parallel_float32",
#     "vector_cumsum_parallel_float64",
#     "vector_cumsum_parallel_int8",
#     "vector_cumsum_parallel_int16",
#     "vector_cumsum_parallel_int32",
#     "vector_cumsum_parallel_int64",
#     "vector_cumsum_parallel_uint8",
#     "vector_cumsum_parallel_uint16",
#     "vector_cumsum_parallel_uint32",
#     "vector_cumsum_parallel_uint64",

#     # Vector operations - low-level type-specific functions for cummax_parallel
#     "vector_cummax_parallel_float32",
#     "vector_cummax_parallel_float64",
#     "vector_cummax_parallel_int8",
#     "vector_cummax_parallel_int16",
#     "vector_cummax_parallel_int32",
#     "vector_cummax_parallel_int64",
#     "vector_cummax_parallel_uint8",
#     "vector_cummax_parallel_uint16",
#     "vector_cummax_parallel_uint32",
#     "vector_cummax_parallel_uint64",

#     # Vector operations - low-level type-specific functions for scan_parallel
#     "vector_scan_parallel_max_float32",
#     "vector_scan_parallel_max_float64",
#     "vector_scan_parallel_max_int32",
#     "vector_scan_parallel_max_int64",
#     "vector_scan_parallel_min_float32",
#     "vector_scan_parallel_min_float64",
#     "vector_scan_parallel_min_int32",
#     "vector_scan_parallel_min_int64",
#     "vector_scan_parallel_sum_float32",
#     "vector_scan_parallel_sum_float64",
#     "vector_scan_parallel_sum_int32",
#     "vector_scan_parallel_sum_int64",
#     "vector_scan_parallel_prod_float32",
#     "vector_scan_parallel_prod_float64",
#     "vector_scan_parallel_prod_int32",
#     "vector_scan_parallel_prod_int64",

#     # GLM operations - high-level dispatch functions
#     "glm_predict_naive",
#     "glm_gradient_naive",
#     "glm_gradient_xyyhat",

#     # GLM operations - low-level type-specific functions for glm_predict_naive
#     "glm_predict_naive_float32",
#     "glm_predict_naive_float64",
#     "glm_predict_naive_int32",
#     "glm_predict_naive_int64",

#     # GLM operations - low-level type-specific functions for glm_gradient_naive
#     "glm_gradient_naive_float32",
#     "glm_gradient_naive_float64",
#     "glm_gradient_naive_int32",
#     "glm_gradient_naive_int64",

#     # GLM operations - low-level type-specific functions for glm_gradient_xyyhat
#     "glm_gradient_xyyhat_float32",
#     "glm_gradient_xyyhat_float64",
#     "glm_gradient_xyyhat_int32",
#     "glm_gradient_xyyhat_int64",

#     # Sort operations - high-level dispatch function
#     "tensor_sort_bitonic",

#     # Sort operations - low-level type-specific functions
#     "tensor_sort_bitonic_float32",
#     "tensor_sort_bitonic_float64",
#     "tensor_sort_bitonic_int8",
#     "tensor_sort_bitonic_int16",
#     "tensor_sort_bitonic_int32",
#     "tensor_sort_bitonic_int64",
#     "tensor_sort_bitonic_uint8",
#     "tensor_sort_bitonic_uint16",
#     "tensor_sort_bitonic_uint32",
#     "tensor_sort_bitonic_uint64",
# ]
