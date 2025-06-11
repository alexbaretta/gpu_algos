#!/usr/bin/env python3
"""
GPU Algorithm Testing Module

Copyright (c) 2025 Alessandro Baretta
All rights reserved.

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

COPYRIGHT = (
    "Copyright (c) 2025 Alessandro Baretta\n"
    "All rights reserved."
)



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

# Constants for GPU architecture
WARP_SIZE = 32
BLOCK_SIZE = 1024

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
    # cuBLAS-based implementations only support float and double
    "*cublas*": {"float", "double"},

    # rocBLAS-based implementations only support float and double
    "*rocblas*": {"float", "double"},

    # Cooperative kernels might have limitations (can be updated as issues are discovered)
    "*cooperative*": set(DATA_TYPES),  # Assume full support for now

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
                 selected_sizes: Optional[Set[int]] = None, selected_types: Optional[Set[str]] = None,
                 dry_run: bool = False, float_abs_tol: Optional[float] = None, float_pct_tol: Optional[float] = None,
                 int_abs_tol: float = 0.0, int_pct_tol: float = 0.0):
        """Initialize the GPU algorithm tester.

        Args:
            bin_dir: Directory containing executable files
            cmake_root: CMake project root directory (optional)
            preset: CMake preset name (default: debug)
            verbose: Enable verbose logging
            selected_executables: Set of executable names to test (None for all)
            selected_sizes: Set of problem sizes to test (None for all)
            selected_types: Set of data types to test (None for all)
            dry_run: Only check executable existence, don't run tests
            float_abs_tol: Absolute tolerance for floating point types (None for no absolute check)
            float_pct_tol: Percent tolerance for floating point types (None for type-specific defaults)
            int_abs_tol: Absolute tolerance for integer types (default: 0.0 - perfect accuracy required)
            int_pct_tol: Percent tolerance for integer types (default: 0.0 - perfect accuracy required)
        """
        self.bin_dir = Path(bin_dir)
        self.cmake_root = cmake_root
        self.preset = preset
        self.verbose = verbose
        self.selected_executables = selected_executables
        self.selected_sizes = selected_sizes
        self.selected_types = selected_types or set(DATA_TYPES)
        self.dry_run = dry_run

        # Tolerance values with IEEE 754-based defaults
        self.float_abs_tol = float_abs_tol
        self.int_abs_tol = int_abs_tol
        self.int_pct_tol = int_pct_tol

        # Set type-specific percent tolerances based on IEEE 754 precision
        # Assuming 4 bits of precision loss in algorithms:
        # float32: 24-4=20 bits → 2^(-20) ≈ 9.54e-7 relative → 9.54e-5%
        # float64: 53-4=49 bits → 2^(-49) ≈ 1.78e-15 relative → 1.78e-13%
        if float_pct_tol is not None:
            self.float_pct_tol = float_pct_tol
        else:
            # Use type-specific defaults - will be determined per data type in _check_correctness
            self.float_pct_tol = None

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
        if not self.executables:
            raise RuntimeError(f"No executables found in {self.bin_dir}")

        self.logger.info(f"Found {len(self.executables)} executables in {self.bin_dir}")

        # Print executable paths for debugging
        if self.verbose or self.dry_run:
            self.logger.info("Discovered executables:")
            for exe in self.executables:
                abs_path = exe.resolve()
                exists = abs_path.exists()
                executable = abs_path.is_file() and os.access(abs_path, os.X_OK)
                self.logger.info(
                    f"  {exe.name}: {abs_path} (exists={exists}, executable={executable})"
                )

    def _discover_executables(self) -> List[Path]:
        """Discover all executable files in the bin directory."""
        executables = []
        for file_path in self.bin_dir.iterdir():
            if file_path.is_file() and os.access(file_path, os.X_OK):
                # Filter by selected executables if specified
                if self.selected_executables is None:
                    executables.append(file_path)
                else:
                    # Check if filename matches any of the patterns (exact match or glob)
                    for pattern in self.selected_executables:
                        if file_path.name == pattern or fnmatch.fnmatch(file_path.name, pattern):
                            executables.append(file_path)
                            break  # Found a match, no need to check other patterns
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

    def _get_size_option(self, executable: Path) -> Optional[str]:
        """Determine the size option for an executable (-n, -m, etc.)."""
        help_text = self._get_executable_help(executable)

        # Look for common size options
        if "-n " in help_text or "-n, " in help_text:
            return "-n"
        elif "-m " in help_text or "-m, " in help_text:
            return "-m"
        elif "--size" in help_text:
            return "--size"

        return None

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
        if skipped_types and (self.verbose or self.dry_run):
            self.logger.info(f"  Skipping unsupported data types for {executable.name}: {sorted(skipped_types)}")

        return filtered_types

    def _run_executable(
        self, executable: Path, args: List[str], timeout: int = 30
    ) -> Tuple[bool, str, str]:
        """Run an executable with given arguments.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        abs_path = executable.resolve()
        cmd = [str(abs_path)] + args
        self.logger.debug(f"Running: {' '.join(cmd)}")

        # Print absolute path for debugging
        if self.verbose:
            self.logger.debug(f"Absolute executable path: {abs_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.bin_dir.parent,
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout running {executable.name} with args: {args}")
            return False, "", "Timeout"
        except Exception as e:
            self.logger.error(f"Error running {executable.name}: {e}")
            return False, "", str(e)

    def _extract_performance_metrics(self, stdout: str) -> Dict[str, float]:
        """Extract performance metrics from stdout."""
        metrics = {}

        # Look for timing information
        timing_patterns = [
            (r"Max error\s*:\s*([\d.e-]+)", "max_error"),
            (r"Max error pct\s*:\s*([\d.e-]+)", "max_error_pct"),
            (r"Gross speedup\s*:\s*([\d.e-]+)", "gross_speedup"),
            (r"Net speedup\s*:\s*([\d.e-]+)", "net_speedup"),
            (r"Compute kernel:\s*([\d.]+)\s*ms", "kernel_time_ms"),
            (r"DONE:\s*([\d.]+)\s*ms total", "total_time_ms"),
        ]

        for pattern, key in timing_patterns:
            match = re.search(pattern, stdout)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    pass

        return metrics

    def _is_integer_type(self, data_type: str) -> bool:
        """Check if a data type is an integer type."""
        return data_type in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

    def _is_floating_type(self, data_type: str) -> bool:
        """Check if a data type is a floating point type."""
        return data_type in ["half", "float", "double"]

    def _check_correctness(self, metrics: Dict[str, float], data_type: str) -> bool:
        """Check if results are correct based on error thresholds.

        Args:
            metrics: Dictionary containing max_error and optionally max_error_pct
            data_type: The data type being tested

        Returns:
            True if results are correct (errors below thresholds), False otherwise
        """
        max_error = metrics.get("max_error", float("inf"))
        max_error_pct = metrics.get("max_error_pct", float("inf"))

        # Error of 0 always satisfies correctness check
        if max_error == 0.0:
            return True

        # Determine appropriate tolerances based on data type
        if self._is_integer_type(data_type):
            abs_tol = self.int_abs_tol
            pct_tol = self.int_pct_tol
        elif self._is_floating_type(data_type):
            abs_tol = self.float_abs_tol

            # Use IEEE 754-based percent tolerance if not explicitly set
            if self.float_pct_tol is not None:
                pct_tol = self.float_pct_tol
            else:
                # Type-specific defaults based on IEEE 754 precision with 4 bits of precision loss
                if data_type == "half":
                    # half precision: 10-bit significand + 1 implicit → 11-4=7 bits → 2^(-7) ≈ 0.0078 → 0.78%
                    pct_tol = 0.78
                elif data_type in ["float", "single"]:
                    # float32: 23-bit significand + 1 implicit → 24-4=20 bits → 2^(-20) ≈ 9.54e-7 → 9.54e-5%
                    pct_tol = 9.54e-5
                elif data_type == "double":
                    # float64: 52-bit significand + 1 implicit → 53-4=49 bits → 2^(-49) ≈ 1.78e-15 → 1.78e-13%
                    pct_tol = 1.78e-13
                else:
                    # Unknown floating type, use float32 default
                    pct_tol = 9.54e-5
        else:
            # Unknown type, use floating point tolerances as default
            abs_tol = self.float_abs_tol
            pct_tol = self.float_pct_tol if self.float_pct_tol is not None else 9.54e-5

        # Check absolute tolerance (if specified)
        if abs_tol is not None and max_error > abs_tol:
            return False

        # Check percent tolerance
        if pct_tol is not None and max_error_pct > pct_tol:
            return False

        return True

    def test_executable(self, executable: Path, data_type: str, size: int) -> Dict:
        """Test a single executable with specific parameters.

        Args:
            executable: Path to the executable file
            data_type: Data type to test (e.g., "float", "int32")
            size: Problem size to test

        Returns:
            Dictionary containing test results and performance metrics
        """
        test_info = {
            "executable": executable.name,
            "data_type": data_type,
            "size": size,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Build command arguments using helper methods
            cmd = [str(executable)]

            # Add size argument
            size_option = self._get_size_option(executable)
            if size_option:
                cmd.extend([size_option, str(size)])
            else:
                # Fallback to positional argument if no size option found
                cmd.append(str(size))

            # Add data type argument
            if self._supports_data_type(executable, data_type):
                cmd.extend(["--type", data_type])
            else:
                # Fallback to positional argument if --type not supported
                cmd.append(data_type)

            self.logger.debug(f"Running: {' '.join(cmd)}")

            # Execute the command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=self.bin_dir
            )

            # Extract performance metrics
            metrics = self._extract_performance_metrics(result.stdout)

            # Determine run success
            run_success = result.returncode == 0

            # Check correctness (only if run was successful and we have metrics)
            correct = False
            if run_success and metrics:
                correct = self._check_correctness(metrics, data_type)

            # Combine all results
            test_result = {
                **test_info,
                "run_success": run_success,
                "correct": correct,
                "return_code": result.returncode,
                "metrics": metrics,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            return test_result

        except subprocess.TimeoutExpired:
            return {
                **test_info,
                "run_success": False,
                "correct": False,
                "return_code": -1,
                "error": "Timeout after 30 seconds",
                "metrics": {},
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            return {
                **test_info,
                "run_success": False,
                "correct": False,
                "return_code": -1,
                "error": str(e),
                "metrics": {},
                "stdout": "",
                "stderr": "",
            }

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

            for size in sizes:
                for data_type in filtered_data_types:
                    if self.verbose or self.dry_run:
                        self.logger.debug(f"    Testing size={size}, type={data_type}")

                    result = self.test_executable(executable, data_type, size)
                    result["category"] = category
                    results.append(result)

                    if not result["run_success"] and (self.verbose or self.dry_run):
                        error_msg = result.get("error", f"Exit code: {result.get('return_code', 'unknown')}")
                        self.logger.warning(f"    Failed: {error_msg}")

        return results

    def run_all_tests(self) -> Dict[str, List[Dict]]:
        """Run tests on all executables.

        Returns:
            Dictionary mapping executable names to their test results
        """
        all_results = {}

        for executable in self.executables:
            try:
                results = self.test_all_sizes_and_types(executable)
                all_results[executable.name] = results
            except Exception as e:
                import traceback
                error_details = f"Exception: {type(e).__name__}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                self.logger.error(f"Error testing {executable.name}: {error_details}")
                all_results[executable.name] = [{"error": str(e), "error_details": error_details}]

        return all_results

    def generate_report(self, results: Dict[str, List[Dict]]) -> str:
        """Generate a summary report of test results."""
        report = []
        report.append("=" * 80)
        report.append("GPU Algorithm Test Report")
        report.append("=" * 80)

        total_tests = 0
        total_passed = 0

        for exe_name, exe_results in results.items():
            report.append(f"\n{exe_name}:")
            report.append("-" * (len(exe_name) + 1))

            passed = sum(1 for r in exe_results if r.get("run_success", False))
            failed = len(exe_results) - passed

            report.append(
                f"  Tests: {len(exe_results)} total, {passed} passed, {failed} failed"
            )

            total_tests += len(exe_results)
            total_passed += passed

            # Group failures by error type
            errors = {}
            for result in exe_results:
                if not result.get("run_success", False):
                    error = result.get("error", "Unknown error")
                    errors[error] = errors.get(error, 0) + 1

            if errors:
                report.append("  Common errors:")
                for error, count in sorted(errors.items()):
                    report.append(f"    {error}: {count} occurrences")

        report.append(f"\n{'=' * 80}")
        report.append(
            f"Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)"
        )
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Test GPU algorithm executables with various problem sizes and data types.\n"
        "Use --bin-path for direct binary directory, or --cmake-root/--preset for CMake preset logic."
        f"\n{COPYRIGHT}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bin-path", help="Direct path to bin directory (overrides CMake preset logic)"
    )

    parser.add_argument(
        "--cmake-root",
        help="Path to CMake root directory (defaults to current directory)",
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
        "--abs-tol",
        type=float,
        help="Absolute tolerance for floating point types (no default - only percent tolerance used)",
    )

    parser.add_argument(
        "--pct-tol",
        type=float,
        help="Percent tolerance for floating point types (default: IEEE 754-based per type)",
    )

    parser.add_argument(
        "--int-abs-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for integer types (default: 0.0 - perfect accuracy required)",
    )

    parser.add_argument(
        "--int-pct-tol",
        type=float,
        default=0.0,
        help="Percent tolerance for integer types (default: 0.0 - perfect accuracy required)",
    )

    args = parser.parse_args()

    try:
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
            selected_types,
            args.dryrun,
            args.abs_tol,
            args.pct_tol,
            args.int_abs_tol,
            args.int_pct_tol,
        )

        # Run all tests
        if args.dryrun:
            print("Dry run mode: checking executable existence...")
        else:
            print("Starting GPU algorithm tests...")

        results = tester.run_all_tests()

        # Generate and print report
        report = tester.generate_report(results)
        print(report)

        # Save detailed results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")

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
        correctness_failures = total_tests - total_correctness_passed

        # Print summary statistics
        print(f"\n{'=' * 80}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total benchmarks: {total_tests}")
        print(f"Execution failures: {execution_failures}")
        print(f"Correctness failures: {correctness_failures}")
        print(f"Successful executions: {total_execution_passed}")
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


if __name__ == "__main__":
    sys.exit(main())
