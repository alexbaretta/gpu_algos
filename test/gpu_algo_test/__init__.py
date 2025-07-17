"""
GPU Algorithm Testing Package

This package provides functionality to test GPU algorithm executables with various
problem sizes and data types.

Copyright (c) 2025 Alessandro Baretta
All rights reserved.
"""

from .gpu_algo_test import (
    DATA_TYPES,
    PROBLEM_SIZES,
    SPECIAL_DATA_TYPES,
    GPUAlgoTest,
    determine_bin_directory,
    expand_special_sizes,
    expand_special_types,
    find_cmake_binary_directory,
)

__version__ = "1.0.0"
__all__ = [
    "GPUAlgoTest",
    "DATA_TYPES",
    "PROBLEM_SIZES",
    "SPECIAL_DATA_TYPES",
    "expand_special_sizes",
    "expand_special_types",
    "find_cmake_binary_directory",
    "determine_bin_directory",
]
