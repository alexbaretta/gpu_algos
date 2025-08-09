#!/usr/bin/env python3
"""
GPU Algorithm Testing Module

source path: test/gpu_algo_test/gpu_algo_test.py

This module provides functionality to test GPU algorithm executables with various
problem sizes and data types. It can be used both as a Python package and as a
command line script.

When used as a command line script, it tests all executables in the binary directory
with various problem sizes:
- Smaller than 1 warp (< 32 elements)
- Exactly 1 warp (32 elements)
- Several warps ending on a warp boundary (64, 96, 128 elements)
- Several warps with a partial warp (33, 50, 100 elements)
- A full block (1024 elements)
- Several blocks ending on a block boundary (2048, 3072 elements)
- Several blocks ending with a partial block (1025, 1500, 2500 elements)

For each problem size, it tests all supported data types.
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



import argparse
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

import ijson

# Data types to test
DATA_TYPES = [
    "half",
    "float",
    "double",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

# Special data type groups
SPECIAL_DATA_TYPES = {
    "floating": ["half", "float", "double"],
    "signed": ["int8", "int16", "int32", "int64"],
    "unsigned": ["uint8", "uint16", "uint32", "uint64"],
    "integer": [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ],
}

# Problem sizes to test
PROBLEM_SIZES = {
    "smaller_than_warp": [8, 16, 24],
    "exactly_one_warp": [32],
    "several_warps_boundary": [64, 96, 128],
    "several_warps_partial": [33, 50, 100],
    "full_block": [1024],
    "several_blocks_boundary": [2048, 3072],
    "several_blocks_partial": [1025, 1500, 2500],
}

# Executable-specific data type support mapping
# Key: executable name pattern (supports fnmatch wildcards)
# Value: set of supported data types
EXECUTABLE_TYPE_SUPPORT = {
    "matrix_product_cublas": {"half", "float", "double"},
    "matrix_transpose_cublas": {"float", "double"},

    # rocBLAS-based implementations only support float and double
    "*rocblas*": {"int8", "uint8", "half", "float", "double"},

    # gradient descent optimizers require floating point types
    "*gradient*optimizer*": {"half", "float", "double"},

    # Tensor core implementations support a specific subset of types
    "*tensor": {"int8", "uint8", "half", "float", "double"},

    # Default: all implementations support all types unless specified otherwise
    # Note: This catch-all pattern must be last
    "*": set(DATA_TYPES),
}


def find_cmake_binary_directory(cmake_root: Path, preset_name: str) -> Path:
    """Find the binary directory for a given CMake preset.

    Args:
        cmake_root: Path to the CMake root directory containing preset files
        preset_name: Name of the CMake preset to use

    Returns:
        Path to the binary directory

    Raises:
        FileNotFoundError: If preset files are not found
        ValueError: If preset is not found or invalid
    """
    preset_files = [
        cmake_root / "CMakePresets.json",
        cmake_root / "CMakeUserPresets.json",
    ]

    all_presets = {}

    # Read all preset files
    for preset_file in preset_files:
        if preset_file.exists():
            try:
                with open(preset_file, "r") as f:
                    preset_data = json.load(f)

                # Extract configure presets
                if "configurePresets" in preset_data:
                    for preset in preset_data["configurePresets"]:
                        if "name" in preset:
                            all_presets[preset["name"]] = preset

            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid CMake preset file {preset_file}: {e}")

    if not all_presets:
        raise FileNotFoundError(f"No CMake preset files found in {cmake_root}")

    if preset_name not in all_presets:
        available_presets = list(all_presets.keys())
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {available_presets}"
        )

    preset = all_presets[preset_name]

    # Determine binary directory
    if "binaryDir" in preset:
        binary_dir_str = preset["binaryDir"]

        # Handle CMake variable substitutions
        binary_dir_str = binary_dir_str.replace("${sourceDir}", str(cmake_root))
        binary_dir_str = binary_dir_str.replace("${workspaceRoot}", str(cmake_root))

        binary_dir = Path(binary_dir_str)
        if not binary_dir.is_absolute():
            binary_dir = cmake_root / binary_dir
    else:
        # Default CMake behavior: build in <source>/build/<preset-name>
        binary_dir = cmake_root / "build" / preset_name

    return binary_dir.resolve()


def determine_bin_directory(
    bin_path: Optional[str] = None,
    cmake_root: Optional[str] = None,
    preset: str = "debug",
) -> Path:
    """Determine the binary directory to use for testing.

    Args:
        bin_path: Direct path to bin directory (overrides CMake preset logic)
        cmake_root: Path to CMake root directory (defaults to current directory)
        preset: CMake preset name (defaults to "debug")

    Returns:
        Path to the bin directory containing executables

    Raises:
        FileNotFoundError: If directories are not found
        ValueError: If CMake preset configuration is invalid
    """
    if bin_path:
        # Direct bin path specified - use it directly
        bin_dir = Path(bin_path).resolve()
        if not bin_dir.exists():
            raise FileNotFoundError(f"Binary directory not found: {bin_dir}")
        return bin_dir

    # Use CMake preset logic
    cmake_root_path = Path(cmake_root or ".").resolve()

    if not cmake_root_path.exists():
        raise FileNotFoundError(f"CMake root directory not found: {cmake_root_path}")

    # Find the build directory from CMake presets
    build_dir = find_cmake_binary_directory(cmake_root_path, preset)
    bin_dir = build_dir / "bin"

    if not bin_dir.exists():
        raise FileNotFoundError(f"Binary directory not found: {bin_dir}")

    return bin_dir


def expand_special_sizes(size_specs: List[str]) -> Set[int]:
    """Expand special size specifications to actual size values.

    Args:
        size_specs: List of size specifications (can be numbers or special names)

    Returns:
        Set of integer sizes

    Special size names are the keys from PROBLEM_SIZES dict:
    - smaller_than_warp, exactly_one_warp, several_warps_boundary,
      several_warps_partial, full_block, several_blocks_boundary, several_blocks_partial
    """
    expanded_sizes = set()

    for spec in size_specs:
        spec = spec.strip()

        # Check if it's a special size name
        if spec in PROBLEM_SIZES:
            expanded_sizes.update(PROBLEM_SIZES[spec])
        else:
            # Try to parse as integer
            try:
                expanded_sizes.add(int(spec))
            except ValueError:
                raise ValueError(
                    f"Invalid size specification: '{spec}'.\n"
                    f"Must be a number or one of: {list(PROBLEM_SIZES.keys())}"
                )

    return expanded_sizes


def expand_special_types(type_specs: List[str]) -> Set[str]:
    """Expand special type specifications to actual data type names.

    Args:
        type_specs: List of type specifications (can be type names or special groups)

    Returns:
        Set of data type names

    Special type groups:
    - floating: half, float, double
    - signed: int8, int16, int32, int64
    - unsigned: uint8, uint16, uint32, uint64
    - integer: all integer types (signed + unsigned)
    """
    expanded_types = set()

    for spec in type_specs:
        spec = spec.strip()

        # Check if it's a special type group
        if spec in SPECIAL_DATA_TYPES:
            expanded_types.update(SPECIAL_DATA_TYPES[spec])
        elif spec in DATA_TYPES:
            # Regular data type
            expanded_types.add(spec)
        else:
            raise ValueError(
                f"Invalid data type specification: '{spec}'.\n"
                f"Must be one of {DATA_TYPES} or one of {list(SPECIAL_DATA_TYPES.keys())}"
            )

    return expanded_types


class GPUAlgoTest:
    """Main class for testing GPU algorithms."""

    def __init__(self, bin_dir: Path, cmake_root: Optional[Path] = None, preset: str = "debug",
                 verbose: bool = False, selected_executables: Optional[Set[str]] = None,
                 selected_sizes: Optional[Set[int]] = None, max_problem_size:int = 2**28,
                 selected_types: Optional[Set[str]] = None,
                 dryrun: bool = False, tol_bits: int = 4, output_file: Optional[str] = None,
                 rerun_failures_file: Optional[str] = None, timeout: int = 300, hip_only: bool = False, cuda_only: bool = False):
        """Initialize the GPU algorithm tester.

        Args:
            bin_dir: Directory containing executable files
            cmake_root: CMake project root directory (optional)
            preset: CMake preset name (default: debug)
            verbose: Enable verbose logging
            selected_executables: Set of executable names to test (None for all)
            selected_sizes: Set of problem sizes to test (None for all)
            max_problem_size: max product of all problem dimension sizes
            selected_types: Set of data types to test (None for all)
            dryrun: Only check executable existence, don't run tests
            tol_bits: Number of bits of precision loss for floating point tolerance (default: 4)
            output_file: Output file for detailed results (None for no output)
            rerun_failures_file: Path to previous test results file (JSONL format) to rerun only failed tests
            timeout: Timeout for test execution in seconds (default: 300 seconds)
            hip_only: Only test HIP executables (default: False)
            cuda_only: Only test CUDA executables (default: False)
        """
        self.bin_dir = Path(bin_dir)
        self.cmake_root = cmake_root
        self.preset = preset
        self.verbose = verbose
        self.selected_executables = selected_executables
        self.selected_sizes = selected_sizes
        self.max_problem_size = max_problem_size
        self.selected_types = selected_types or set(DATA_TYPES)
        self.dryrun = dryrun
        self.output_file = output_file
        self.rerun_failures_file = rerun_failures_file
        self.timeout = timeout
        self.hip_only = hip_only
        self.cuda_only = cuda_only
        self.tol_bits = tol_bits
        self.size_option_regex = re.compile(r"-[A-Z][ ,]")

        if not self.bin_dir.exists():
            raise FileNotFoundError(f"Binary directory not found: {self.bin_dir}")

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Discover executables
        self.executables = self._discover_executables()
        self.executable_names = [ exe.name for exe in self.executables ]
        if not self.executables:
            raise RuntimeError(f"No executables found in {self.bin_dir}")

        self.logger.info(f"Found {len(self.executables)} executables in {self.bin_dir}")

        # Print platform filtering information if applicable
        if self.hip_only or self.cuda_only:
            platform = "HIP" if self.hip_only else "CUDA"
            self.logger.info(f"Platform filtering: {platform} executables only")

        # Print executable paths for debugging
        if self.verbose or self.dryrun:
            self.logger.info("Discovered executables:")
            for exe in self.executables:
                abs_path = exe.resolve()
                exists = abs_path.exists()
                executable = abs_path.is_file() and os.access(abs_path, os.X_OK)
                platform_info = ""
                if self.verbose:
                    if self._is_hip_executable(exe):
                        platform_info = " [HIP]"
                    else:
                        platform_info = " [CUDA]"
                        pass
                    pass
                self.logger.info(
                    f"  {exe.name}{platform_info}: {abs_path} (exists={exists}, executable={executable})"
                )
                pass
            pass

        # Open output file if specified
        self.output_handle = None
        if self.output_file:
            self.output_handle = open(self.output_file, 'w')
            pass

        # Parse failed tests from previous results if specified
        self.failed_tests = None
        if self.rerun_failures_file:
            self.failed_tests = self._parse_failed_tests()
            if self.failed_tests:
                self.logger.info(f"Found {len(self.failed_tests)} failed tests to rerun")
            else:
                self.logger.error("No failed tests found in previous results file")
                sys.exit(1)
                pass
            pass

    def _write_streaming_result(self, result: Dict) -> None:
        """Write a single test result to the streaming output file as a JSON Lines entry."""
        if self.output_handle:
            # Write each result as a single JSON object on its own line
            json.dump(result, self.output_handle, separators=(',', ':'))
            self.output_handle.write('\n')
            self.output_handle.flush()

    def _close_streaming_output(self) -> None:
        """Close the streaming output file."""
        if self.output_handle:
            self.output_handle.close()
            self.output_handle = None

    def _is_hip_executable(self, executable: Path) -> bool:
        """Check if an executable is a HIP implementation.

        Args:
            executable: Path to the executable

        Returns:
            True if this appears to be a HIP executable
        """
        # HIP executables typically have "hip_" prefix
        return executable.name.startswith("hip_")

    def _is_cuda_executable(self, executable: Path) -> bool:
        """Check if an executable is a CUDA implementation.

        Args:
            executable: Path to the executable

        Returns:
            True if this appears to be a CUDA executable
        """
        # CUDA executables typically don't have "hip_" prefix
        # and aren't other special types (like CPU-only implementations)
        return not self._is_hip_executable(executable)

    def _discover_executables(self) -> List[Path]:
        """Discover all executable files in the bin directory."""
        executables = []
        if self.selected_executables is not None:
            accept_patterns = [ pattern for pattern in self.selected_executables if not pattern.startswith('-')]
            reject_patterns = [ pattern[1:] for pattern in self.selected_executables if pattern.startswith('-')]
        else:
            accept_patterns = []
            reject_patterns = []
            pass
        for file_path in self.bin_dir.iterdir():
            if file_path.is_file() and os.access(file_path, os.X_OK):
                # Apply platform filtering first
                if self.hip_only and not self._is_hip_executable(file_path):
                    continue
                if self.cuda_only and not self._is_cuda_executable(file_path):
                    continue

                reject_executable = False
                for reject_pattern in reject_patterns:
                    if file_path.name == reject_pattern or fnmatch.fnmatch(file_path.name, reject_pattern):
                        if self.verbose:
                            self.logger.warning(f'Rejecting {file_path.name} because it matches -{reject_pattern}')
                            pass
                        reject_executable = True
                        break # We found a negative match, no need to check other patterns
                    pass

                # Filter by selected executables if specified
                if reject_executable:
                    continue
                if len(accept_patterns) == 0:
                    executables.append(file_path)
                else:
                    # Check if filename matches any of the patterns (exact match or glob)
                    for accept_pattern in accept_patterns:
                        if file_path.name == accept_pattern or fnmatch.fnmatch(file_path.name, accept_pattern):
                            executables.append(file_path)
                            break  # Found a match, no need to check other patterns
                        pass
                    pass
                pass
            pass
        return sorted(executables)

    def _get_filtered_problem_sizes(self) -> Dict[str, List[int]]:
        """Get problem sizes filtered by selection criteria."""
        if self.selected_sizes is None:
            return PROBLEM_SIZES

        filtered_sizes = {}
        for category, sizes in PROBLEM_SIZES.items():
            filtered = [size for size in sizes if size in self.selected_sizes]
            if filtered:
                filtered_sizes[category] = filtered
        return filtered_sizes

    def _get_executable_help(self, executable: Path) -> str:
        """Get help text from an executable."""
        try:
            abs_path = executable.resolve()
            result = subprocess.run(
                [str(abs_path), "--help"], capture_output=True, text=True, timeout=10
            )
            return result.stdout
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.warning(f"Failed to get help for {executable.name}: {e}")
            return ""

    def _supports_option(self, executable: Path, option: str) -> bool:
        """Check if an executable supports a specific command line option."""
        help_text = self._get_executable_help(executable)
        return option in help_text

    def _supports_data_type(self, executable: Path, data_type: str) -> bool:
        """Check if an executable supports a specific data type."""
        help_text = self._get_executable_help(executable)
        return "--type" in help_text

    def _get_size_options(self, executable: Path) -> List[str]:
        """Determine the size option for an executable (-n, -m, etc.)."""
        help_text = self._get_executable_help(executable)
        found_options = self.size_option_regex.findall(help_text)
        one_letter_options = [ ('-' + opt[1]) for opt in found_options]
        return one_letter_options

    def _get_supported_data_types(self, executable: Path) -> Set[str]:
        """Get the set of data types supported by a specific executable.

        Args:
            executable: Path to the executable

        Returns:
            Set of supported data type names
        """
        exe_name = executable.name

        # Check patterns in order (more specific patterns should come first)
        for pattern, supported_types in EXECUTABLE_TYPE_SUPPORT.items():
            if fnmatch.fnmatch(exe_name, pattern):
                return supported_types.copy()

        # Default fallback (should not happen due to "*" pattern)
        return set(DATA_TYPES)

    def _get_filtered_data_types(self, executable: Path) -> Set[str]:
        """Get data types filtered by both user selection and executable support.

        Args:
            executable: Path to the executable

        Returns:
            Set of data types that are both selected by user and supported by executable
        """
        supported_types = self._get_supported_data_types(executable)
        filtered_types = self.selected_types.intersection(supported_types)

        # Log skipped types if verbose mode and some types were filtered out
        skipped_types = self.selected_types - supported_types
        if skipped_types and (self.verbose or self.dryrun):
            self.logger.info(f"  Skipping unsupported data types for {executable.name}: {sorted(skipped_types)}")

        return filtered_types

    def _extract_performance_metrics(self, stdout: str) -> Dict[str, float]:
        """Extract performance metrics from stdout."""
        metrics = {}

        # Look for timing information and tolerance
        timing_patterns = [
            (r"Max error\s*:\s*([\d.e-]+|nan|inf)", "max_error"),
            (r"Max error rel\s*:\s*([\d.e-]+|nan|inf)", "max_error_rel"),
            (r"Tolerance pct\s*:\s*([\d.e-]+)%", "tolerance"),
            (r"Gross speedup\s*:\s*([\d.e-]+)", "gross_speedup"),
            (r"Net speedup\s*:\s*([\d.e-]+)", "net_speedup"),
            (r"Compute kernel:\s*([\d.]+)\s*ms", "kernel_time_ms"),
            (r"DONE:\s*([\d.]+)\s*ms total", "total_time_ms"),
        ]

        for pattern, key in timing_patterns:
            match = re.search(pattern, stdout)
            if match:
                try:
                    value_str = match.group(1).lower()
                    metrics[key] = float(value_str)
                except ValueError:
                    metrics[key] = value_str
                    pass

        # Look for SUCCESS/FAILURE status
        if "[SUCCESS]" in stdout:
            metrics["tolerance_success"] = True
        elif "[FAILURE]" in stdout:
            metrics["tolerance_success"] = False

        return metrics

    def _is_integer_type(self, data_type: str) -> bool:
        """Check if a data type is an integer type."""
        return data_type in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

    def _is_floating_type(self, data_type: str) -> bool:
        """Check if a data type is a floating point type."""
        return data_type in ["half", "float", "double"]

    def _check_correctness(self, metrics: Dict[str, float], data_type: str, executable: Path) -> tuple[bool, str]:
        """Check if results are correct based on executable-reported tolerance or fallback logic.

        Args:
            metrics: Dictionary containing max_error, max_error_rel, and optionally tolerance_success
            data_type: The data type being tested
            executable: Path to the executable being tested

        Returns:
            Tuple of (is_correct, reason_string)
        """
        import math

        # First priority: Check if executable reported success/failure
        if "tolerance_success" in metrics:
            if metrics["tolerance_success"]:
                return True, "Executable reported [SUCCESS]"
            else:
                return False, "Executable reported [FAILURE]"

        # Second priority: Use executable-reported tolerance if available
        max_error_rel = metrics.get("max_error_rel", float("inf"))
        tolerance = metrics.get("tolerance")

        if tolerance is not None and not math.isnan(max_error_rel):
            if max_error_rel <= tolerance:
                return True, f"max_error_rel <= tolerance ({max_error_rel:.6g} <= {tolerance:.6g})"
            else:
                return False, f"max_error_rel > tolerance ({max_error_rel:.6g} > {tolerance:.6g})"

        # Fallback for legacy executables or when tolerance not reported
        max_error = metrics.get("max_error", float("inf"))

        # Handle NaN cases explicitly
        if math.isnan(max_error):
            return False, "max_error is NaN"

        # Error of 0 always satisfies correctness check
        if max_error == 0:
            return True, "max_error == 0"

        # Simple fallback tolerance based on data type
        if self._is_integer_type(data_type):
            return False, "Integer types require exact results (fallback)"
        else:
            # For floating types, use a generous fallback tolerance
            if math.isnan(max_error_rel):
                # Use absolute error fallback
                if data_type == "half":
                    fallback_abs_tol = 1e-3
                elif data_type in ["float", "single"]:
                    fallback_abs_tol = 1e-6
                else:  # double
                    fallback_abs_tol = 1e-12

                if max_error <= fallback_abs_tol:
                    return True, f'max_error <= {fallback_abs_tol} (fallback absolute tolerance)'
                else:
                    return False, f'max_error > {fallback_abs_tol} (fallback absolute tolerance)'
            else:
                # Use percent error fallback
                fallback_rel_tol = 1.0  # 1% fallback tolerance
                if max_error_rel <= fallback_rel_tol:
                    return True, f'max_error_rel <= {fallback_rel_tol}% (fallback tolerance)'
                else:
                    return False, f'max_error_rel > {fallback_rel_tol}% (fallback tolerance)'

    def test_executable(self, executable: Path, data_type: str, size_i: int, sizes: List[int]) -> Optional[Dict]:
        """Test a single executable with specific parameters.

        Args:
            executable: Path to the executable file
            data_type: Data type to test (e.g., "float", "int32")
            size_i: Index of size for first problem dimension
            sizes: List of available sizes

        Returns:
            Dictionary containing test results and performance metrics
        """
        test_info = {
            "executable": executable.name,
            "data_type": data_type,
            "size_i": size_i,
            "sizes": sizes,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Build command arguments using helper methods
            command_name = executable.name
            cmd = [str(executable)]

            if self.verbose or self.dryrun:
                self.logger.debug(f"    {command_name}: Testing type={data_type}, size_i={size_i}, sizes={str(sizes)}")
                pass

            # Add size argument
            size_options = self._get_size_options(executable)
            if size_options:
                # Cycle through the size options, and correspondingly through the available size values
                i = size_i
                size_product = 1
                for size_option in size_options:
                    # Prevent unreasonably large problem sizes by ensuring that the product of all dimensions <= max_problem_size
                    size = min(sizes[i % len(sizes)], self.max_problem_size // size_product)
                    size_product *= size
                    cmd.extend([size_option, str(size)])
                    i += 1
                    pass
            else:
                # Fallback to positional argument if no size option found
                raise Exception(f"no size option found in executable {command_name}")
                pass

            # Add data type argument
            if self._supports_data_type(executable, data_type):
                cmd.extend(["--type", data_type])
            else:
                # Fallback to positional argument if --type not supported
                cmd.append(data_type)
                pass

            # Add tolerance bits argument for floating point types
            if self._is_floating_type(data_type) and self._supports_option(executable, "--tol-bits"):
                cmd.extend(["--tol-bits", str(self.tol_bits)])
                pass

            # Capture the command line for debugging and replication
            cmdline = ' '.join(cmd)

            # In dry run mode, just print the command and return a mock result
            if self.dryrun:
                print(f"dryrun: {cmdline}")
                return None
            else:
                # self.logger.debug(f"Running: {cmdline}")

                # Execute the command
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout, cwd=self.bin_dir
                )

                # Extract performance metrics
                metrics = self._extract_performance_metrics(result.stdout)

                # Determine run success
                run_success = result.returncode == 0

                # Check correctness (only if run was successful and we have metrics)
                correct = False

                # Combine all results
                test_result = {
                    **test_info,
                    "cmdline": cmdline,
                    "run_success": run_success,
                    "return_code": result.returncode,
                    "metrics": metrics,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                if run_success and metrics:
                    correct, correctness_reason = self._check_correctness(metrics, data_type, executable)
                    test_result["correct"] = correct
                    test_result["correctness_reason"] = correctness_reason
                    if correct:
                        self.logger.debug(f"       Correct: {cmdline}")
                    else:
                        self.logger.debug(f"       Tolerance exceeded: {cmdline}")
                        pass
                else:
                    self.logger.debug(f"       Failed: {cmdline}")
                    pass

                return test_result

        except subprocess.TimeoutExpired:
            self.logger.warning(f"       Timeout: {cmdline}")
            return {
                **test_info,
                "cmdline": ' '.join(cmd) if 'cmd' in locals() else "",
                "run_success": False,
                "return_code": -1,
                "error": f"Timeout after {self.timeout} seconds",
                "metrics": {},
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            self.logger.error(f"           {e}")
            return {
                **test_info,
                "cmdline": ' '.join(cmd) if 'cmd' in locals() else "",
                "run_success": False,
                "return_code": -1,
                "error": str(e),
                "metrics": {},
                "stdout": "",
                "stderr": "",
            }
        finally:
            pass

    def test_all_sizes_and_types(self, executable: Path) -> List[Dict]:
        """Test an executable with all problem sizes and data types."""
        results = []

        self.logger.info(f"Testing {executable.name}")

        # Get data types filtered by both user selection and executable support
        filtered_data_types = self._get_filtered_data_types(executable)
        if not filtered_data_types:
            self.logger.warning(f"  No supported data types for {executable.name}, skipping")
            return results

        problem_sizes = self._get_filtered_problem_sizes()

        for category, sizes in problem_sizes.items():
            self.logger.debug(f"  Testing {category}")

            for size_i in range(len(sizes)):
                for data_type in filtered_data_types:
                    result = self.test_executable(executable, data_type, size_i, sizes)
                    if result is not None:
                        result["category"] = category
                        results.append(result)

                        # Write result immediately for streaming output
                        self._write_streaming_result(result)

                        if not result["run_success"] and (self.verbose or self.dryrun):
                            error_msg = result.get("error", f"Exit code: {result.get('return_code', 'unknown')}")
                            self.logger.warning(f"    Failed: {error_msg}")

        return results

    def run_failed_tests(self) -> Dict[str, List[Dict]]:
        """Run only the previously failed tests.

        Returns:
            Dictionary mapping executable names to their test results
        """
        if not self.failed_tests:
            self.logger.warning("No failed tests to rerun")
            return {}

        self.logger.info(f"Rerunning {len(self.failed_tests)} failed tests")
        if self.output_file:
            self.logger.info(f"Detailed results streaming to: {self.output_file}")

        all_results = {}

        # Group failed tests by executable
        tests_by_executable = {}
        for test in self.failed_tests:
            exe_name = test['executable']
            if exe_name not in tests_by_executable:
                tests_by_executable[exe_name] = []
            tests_by_executable[exe_name].append(test)

        for exe_name, failed_tests in tests_by_executable.items():
            self.logger.info(f"Rerunning {len(failed_tests)} tests for {exe_name}")

            # Find the executable path
            executable = None
            for exe_path in self.executables:
                if exe_path.name == exe_name:
                    executable = exe_path
                    break

            if not executable:
                error_msg = f"Executable {exe_name} not found in bin directory"
                self.logger.error(error_msg)
                all_results[exe_name] = [{"error": error_msg, "executable": exe_name}]
                continue

            results = []
            filtered_data_types = self._get_filtered_data_types(executable)
            for test in failed_tests:
                if test['data_type'] not in filtered_data_types:
                    continue
                sizes = test["sizes"]
                if self.verbose or self.dryrun:
                    self.logger.debug(f"  Rerunning: {exe_name} size_i={test['size_i']} sizes={sizes} type={test['data_type']}")

                try:
                    result = self.test_executable(executable, test['data_type'], test['size_i'], test['sizes'])
                    if result is not None:
                        result["category"] = test['category']
                        result["rerun"] = True
                        result["previous_run_success"] = test['previous_run_success']
                        result["previous_correct"] = test['previous_correct']
                        results.append(result)

                        # Write result immediately for streaming output
                        self._write_streaming_result(result)

                        if not result["run_success"] and (self.verbose or self.dryrun):
                            error_msg = result.get("error", f"Exit code: {result.get('return_code', 'unknown')}")
                            self.logger.warning(f"    Still failing: {error_msg}")
                        elif result["run_success"] and result.get("correct", False) and (self.verbose or self.dryrun):
                            self.logger.info(f"    Now passing!")

                except Exception as e:
                    import traceback
                    error_details = f"Exception: {type(e).__name__}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                    self.logger.error(f"Error retesting {exe_name}: {error_details}")

                    error_result = {
                        "error": str(e),
                        "error_details": error_details,
                        "executable": exe_name,
                        "data_type": test['data_type'],
                        "size_i": test['size_i'],
                        "sizes": sizes,
                        "rerun": True
                    }
                    self._write_streaming_result(error_result)
                    results.append(error_result)

            all_results[exe_name] = results

        # Close streaming output
        self._close_streaming_output()

        return all_results

    def run_all_tests(self) -> Dict[str, List[Dict]]:
        """Run tests on all executables.

        Returns:
            Dictionary mapping executable names to their test results
        """
        # If rerunning failed tests, use the specialized method
        if self.failed_tests is not None:
            return self.run_failed_tests()

        if self.output_file:
            self.logger.info(f"Detailed results streaming to: {self.output_file}")

        all_results = {}

        for executable in self.executables:
            try:
                results = self.test_all_sizes_and_types(executable)
                all_results[executable.name] = results

                # Note: Individual results are now written immediately in test_all_sizes_and_types
            except Exception as e:
                import traceback
                error_details = f"Exception: {type(e).__name__}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                self.logger.error(f"Error testing {executable.name}: {error_details}")

                # Write error result to streaming output
                error_result = {"error": str(e), "error_details": error_details, "executable": executable.name}
                self._write_streaming_result(error_result)
                all_results[executable.name] = [error_result]

        # Close streaming output
        self._close_streaming_output()

        return all_results

    def generate_report(self, results: Dict[str, List[Dict]]) -> str:
        """Generate a summary report of test results."""
        report = []
        report.append("=" * 80)
        report.append("GPU Algorithm Test Report")
        report.append("=" * 80)

        total_tests = 0
        total_execution_passed = 0
        total_correct = 0

        # Collect failed tests for detailed reporting
        execution_failures = []
        correctness_failures = []

        for exe_name, exe_results in results.items():
            report.append(f"\n{exe_name}:")
            report.append("-" * (len(exe_name) + 1))

            # Count different types of results
            execution_passed = sum(1 for r in exe_results if r.get("run_success", False))
            execution_failed = len(exe_results) - execution_passed
            correct = sum(1 for r in exe_results if r.get("correct", False))
            correctness_failed = execution_passed - correct

            report.append(
                f"  Tests: {len(exe_results)} total, {execution_passed} executed successfully, {execution_failed} execution failures"
            )

            if correctness_failed > 0:
                report.append(
                    f"  Correctness: {correct} correct, {correctness_failed} correctness failures"
                )
            else:
                report.append(
                    f"  Correctness: {correct} correct"
                )

            total_tests += len(exe_results)
            total_execution_passed += execution_passed
            total_correct += correct

            # Collect failed tests for detailed reporting
            for result in exe_results:
                if not result.get("run_success", False):
                    execution_failures.append(result)
                elif not result.get("correct", False):
                    correctness_failures.append(result)

            # Group execution failures by error type
            errors = {}
            for result in exe_results:
                if not result.get("run_success", False):
                    error = result.get("error", "Unknown error")
                    errors[error] = errors.get(error, 0) + 1

            if errors:
                report.append("  Common execution errors:")
                for error, count in sorted(errors.items()):
                    report.append(f"    {error}: {count} occurrences")

        report.append(f"\n{'=' * 80}")
        if total_tests > 0:
            report.append(
                f"Overall: {total_correct}/{total_tests} tests passed both execution and correctness checks ({total_correct/total_tests*100:.1f}%)"
            )
        else:
            report.append("Overall: No tests were executed")
        report.append("=" * 80)

        # Add detailed failure sections
        if execution_failures:
            report.append(f"\n{'=' * 80}")
            report.append("TESTS THAT FAILED TO RUN")
            report.append("=" * 80)
            report.append(f"\nTotal execution failures: {len(execution_failures)}")
            report.append("\nCommand lines:")
            for i, failure in enumerate(execution_failures, 1):
                exe_name = failure.get("executable", "unknown")
                data_type = failure.get("data_type", "unknown")
                size_i = failure.get("size_i", "unknown")
                sizes = failure.get("sizes", "unknown")
                cmdline = failure.get("cmdline", "No command line available")
                error = failure.get("error", "Unknown error")
                report.append(f"{i:3d}. {exe_name} (type={data_type}, size_i={size_i}, sizes={str(sizes)})")
                report.append(f"     Command: {cmdline}")
                report.append(f"     Error: {error}")

        if correctness_failures:
            report.append(f"\n{'=' * 80}")
            report.append("TESTS THAT FAILED CORRECTNESS CHECKS")
            report.append("=" * 80)
            report.append(f"\nTotal correctness failures: {len(correctness_failures)}")
            report.append("\nCommand lines:")
            for i, failure in enumerate(correctness_failures, 1):
                exe_name = failure.get("executable", "unknown")
                data_type = failure.get("data_type", "unknown")
                size_i = failure.get("size_i", "unknown")
                sizes = failure.get("sizes", "unknown")
                cmdline = failure.get("cmdline", "No command line available")
                metrics = failure.get("metrics", {})
                max_error = metrics.get("max_error", "N/A")
                max_error_rel = metrics.get("max_error_rel", "N/A")
                report.append(f"{i:3d}. {exe_name} (type={data_type}, size_i={size_i}, sizes={str(sizes)})")
                report.append(f"     Command: {cmdline}")
                report.append(f"     Max error: {max_error}, Max error %: {max_error_rel}")

        return "\n".join(report)

    def _parse_failed_tests(self) -> List[Dict]:
        """Parse failed tests from previous results file.

        Returns:
            List of test specifications for failed tests (executable, data_type, size_i, sizes)
        """
        failed_tests = []
        results_file = Path(self.rerun_failures_file)

        if not results_file.exists():
            raise FileNotFoundError(f"Previous results file not found: {results_file}")

        try:
            with open(results_file, 'r') as f:
                if self.verbose:
                    skipped_executables = set()
                    pass
                for result in ijson.items(f, '', multiple_values=True):
                    # Skip entries that don't have the required fields
                    if not all(key in result for key in ['executable', 'data_type', 'size_i', 'sizes']):
                        continue

                    # Check if this test failed (either execution failure or correctness failure)
                    run_success = result.get('run_success', True)
                    correct = result.get('correct', True)

                    # A test is considered failed if:
                    # 1. It failed to run successfully (run_success == False), OR
                    # 2. It ran successfully but gave incorrect results (correct == False)
                    is_failed = not run_success or not correct

                    if is_failed:
                        # Filter by executable if specified
                        executable = result['executable']
                        if self.verbose and executable not in self.executable_names and executable not in skipped_executables:
                            self.logger.info(f'Skipping deselected {executable}')
                            skipped_executables.add(executable)
                            continue

                        failed_test = {
                            'executable': result['executable'],
                            'data_type': result['data_type'],
                            'size_i': result['size_i'],
                            'sizes': result['sizes'],
                            'category': result.get('category', 'unknown'),
                            'previous_run_success': run_success,
                            'previous_correct': correct,
                        }
                        failed_tests.append(failed_test)

        except Exception as e:
            raise RuntimeError(f"Error parsing previous results file {results_file}: {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_failed_tests = []
        for test in failed_tests:
            key = (test['executable'], test['data_type'], test['size_i'], tuple(test['sizes']))
            if key not in seen:
                seen.add(key)
                unique_failed_tests.append(test)

        return unique_failed_tests


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Test GPU algorithm executables with various problem sizes and data types.\n"
        "Use --bin-path for direct binary directory, or --cmake-root/--preset for CMake preset logic."
        f"\n{COPYRIGHT}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bin-path", help="Direct path to bin directory (overrides CMake preset logic)",
    )

    parser.add_argument(
        "--cmake-root",
        help="Path to CMake root directory (defaults to current directory)",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--preset", default="debug", help="CMake preset name (defaults to 'debug')"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "-o", "--output", help="Output file for detailed results (JSON format)"
    )

    parser.add_argument(
        "--executables",
        help="Comma-separated list of executable names or glob patterns to test (default: all).\n"
        "Supports shell-style wildcards: * matches any number of characters, ? matches a single character.\n"
        "Examples: 'vector_*' matches all executables starting with 'vector_',\n"
        "'*_sort' matches all executables ending with '_sort'",
    )

    parser.add_argument(
        "--sizes",
        help="Comma-separated list of problem sizes to test (default: all).\n"
        "Can include numbers or special size categories:\n"
        + ", ".join(PROBLEM_SIZES.keys()),
    )

    parser.add_argument(
        "--max-problem-size",
        help="Maximum total size of the problem (product of all dimensions).\n",
        type=int,
        # This is a fairly arbitrary default, but it should prevent running unreasonably
        # large tests in most cases
        default=2 ** 28,
    )

    parser.add_argument(
        "--types",
        help="Comma-separated list of data types to test (default: all).\n"
        "Can include specific types or special groups: floating, signed, unsigned, integer.\n"
        f"Available types: {', '.join(DATA_TYPES)}",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only check for executable existence, don't run tests",
    )

    parser.add_argument(
        "--tol-bits",
        type=int,
        default=6,
        help="Number of bits of precision loss for floating point tolerance (default: 6)",
    )

    parser.add_argument(
        "--rerun-failures",
        help="Path to previous test results file (JSONL format) to rerun only failed tests",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for test execution in seconds (default: 300 seconds)",
    )

    parser.add_argument(
        "--hip-only",
        action="store_true",
        help="Only test HIP executables (executables with 'hip_' prefix)",
    )

    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Only test CUDA executables (executables without 'hip_' prefix)",
    )

    args = parser.parse_args()

    try:
        # Validate mutually exclusive options
        if args.hip_only and args.cuda_only:
            print("Error: --hip-only and --cuda-only options are mutually exclusive", file=sys.stderr)
            return 1

        # Parse selected executables
        selected_executables = None
        if args.executables:
            selected_executables = set(
                name.strip() for name in args.executables.split(",")
            )

        # Parse selected sizes
        selected_sizes = None
        if args.sizes:
            try:
                size_specs = [size.strip() for size in args.sizes.split(",")]
                selected_sizes = expand_special_sizes(size_specs)
            except ValueError as e:
                print(f"Error parsing sizes: {e}", file=sys.stderr)
                return 1

        # Parse selected types
        selected_types = None
        if args.types:
            try:
                type_specs = [dtype.strip() for dtype in args.types.split(",")]
                selected_types = expand_special_types(type_specs)
            except ValueError as e:
                print(f"Error parsing types: {e}", file=sys.stderr)
                return 1

        # Determine binary directory
        try:
            bin_dir = determine_bin_directory(
                bin_path=args.bin_path, cmake_root=args.cmake_root, preset=args.preset
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error determining binary directory: {e}", file=sys.stderr)
            return 1

        # Create test runner
        tester = GPUAlgoTest(
            bin_dir,
            args.cmake_root,
            args.preset,
            args.verbose,
            selected_executables,
            selected_sizes,
            args.max_problem_size,
            selected_types,
            args.dryrun,
            args.tol_bits,
            args.output,
            args.rerun_failures,
            args.timeout,
            args.hip_only,
            args.cuda_only,
        )

        # Run all tests
        if args.dryrun:
            print("Dry run mode: checking executable existence...")
        elif args.rerun_failures:
            if selected_executables:
                platform_msg = ""
                if args.hip_only:
                    platform_msg = " (HIP only)"
                elif args.cuda_only:
                    platform_msg = " (CUDA only)"
                print(f"Rerunning failed tests for executables: {', '.join(selected_executables)}{platform_msg}")
            else:
                platform_msg = ""
                if args.hip_only:
                    platform_msg = " (HIP executables only)"
                elif args.cuda_only:
                    platform_msg = " (CUDA executables only)"
                print(f"Rerunning all failed tests from previous results{platform_msg}...")
        else:
            platform_msg = ""
            if args.hip_only:
                platform_msg = " (HIP executables only)"
            elif args.cuda_only:
                platform_msg = " (CUDA executables only)"
            print(f"Starting GPU algorithm tests{platform_msg}...")

        results = tester.run_all_tests()

        # Generate and print report
        report = tester.generate_report(results)
        print(report)

        # Return exit code based on results
        total_tests = sum(len(exe_results) for exe_results in results.values())
        total_execution_passed = sum(
            sum(1 for r in exe_results if r.get("run_success", False))
            for exe_results in results.values()
        )
        total_correctness_passed = sum(
            sum(1 for r in exe_results if r.get("correct", False))
            for exe_results in results.values()
        )

        # Calculate failures
        execution_failures = total_tests - total_execution_passed
        correctness_failures = total_execution_passed - total_correctness_passed

        # Print summary statistics
        print(f"\n{'=' * 80}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total benchmarks: {total_tests}")
        print(f"Execution failures: {execution_failures}")
        print(f"Successful executions: {total_execution_passed}")
        print(f"Correctness failures: {correctness_failures}")
        print(f"Correct results: {total_correctness_passed}")

        if args.dryrun:
            if execution_failures == 0:
                print("\nAll executables found!")
                return 0
            else:
                print(f"\n{execution_failures} executables not found or not executable.")
                return 1
        else:
            if execution_failures == 0 and correctness_failures == 0:
                print("\nAll benchmarks passed execution and correctness checks!")
                return 0
            elif execution_failures > 0 and correctness_failures > 0:
                print(f"\n{execution_failures} benchmarks failed to execute, {correctness_failures} had incorrect results.")
                # Prioritize execution failures (exit code 1) over correctness failures (exit code 2)
                return 1
            elif execution_failures > 0:
                print(f"\n{execution_failures} benchmarks failed to execute.")
                return 1
            else:  # correctness_failures > 0
                print(f"\n{correctness_failures} benchmarks had incorrect results.")
                return 2

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main())
