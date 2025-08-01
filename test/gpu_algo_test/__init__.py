"""
GPU Algorithm Testing Package

This package provides functionality to test GPU algorithm executables with various
problem sizes and data types.

source path: test/gpu_algo_test/__init__.py
"""

COPYRIGHT = '''
Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

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
