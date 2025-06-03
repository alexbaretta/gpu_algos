"""
Basic tests for gpu_algo_test package.

Copyright (c) 2025 Alessandro Baretta
All rights reserved.
"""

import unittest
from pathlib import Path

from gpu_algo_test import (
    GPUAlgoTest,
    WARP_SIZE,
    BLOCK_SIZE,
    DATA_TYPES,
    PROBLEM_SIZES,
    SPECIAL_DATA_TYPES,
    expand_special_sizes,
    expand_special_types,
)


class TestConstants(unittest.TestCase):
    """Test package constants."""

    def test_gpu_constants(self):
        """Test GPU architecture constants."""
        self.assertEqual(WARP_SIZE, 32)
        self.assertEqual(BLOCK_SIZE, 1024)

    def test_data_types(self):
        """Test data types list."""
        expected_types = [
            "half", "float", "double",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64"
        ]
        self.assertEqual(DATA_TYPES, expected_types)

    def test_special_data_types(self):
        """Test special data type groups."""
        self.assertIn("floating", SPECIAL_DATA_TYPES)
        self.assertIn("signed", SPECIAL_DATA_TYPES)
        self.assertIn("unsigned", SPECIAL_DATA_TYPES)
        self.assertIn("integer", SPECIAL_DATA_TYPES)

    def test_problem_sizes(self):
        """Test problem size categories."""
        required_categories = [
            "smaller_than_warp",
            "exactly_one_warp",
            "several_warps_boundary",
            "several_warps_partial",
            "full_block",
            "several_blocks_boundary",
            "several_blocks_partial"
        ]
        for category in required_categories:
            self.assertIn(category, PROBLEM_SIZES)


class TestExpansionFunctions(unittest.TestCase):
    """Test special value expansion functions."""

    def test_expand_special_sizes(self):
        """Test special size expansion."""
        # Test single special size
        result = expand_special_sizes(["exactly_one_warp"])
        self.assertEqual(result, {32})

        # Test multiple special sizes
        result = expand_special_sizes(["exactly_one_warp", "full_block"])
        self.assertEqual(result, {32, 1024})

        # Test mixed special and regular sizes
        result = expand_special_sizes(["exactly_one_warp", "100"])
        self.assertEqual(result, {32, 100})

        # Test invalid size
        with self.assertRaises(ValueError):
            expand_special_sizes(["invalid_size"])

    def test_expand_special_types(self):
        """Test special type expansion."""
        # Test single special type
        result = expand_special_types(["floating"])
        self.assertEqual(result, {"half", "float", "double"})

        # Test multiple special types
        result = expand_special_types(["floating", "signed"])
        expected = {"half", "float", "double", "int8", "int16", "int32", "int64"}
        self.assertEqual(result, expected)

        # Test mixed special and regular types
        result = expand_special_types(["floating", "uint8"])
        expected = {"half", "float", "double", "uint8"}
        self.assertEqual(result, expected)

        # Test invalid type
        with self.assertRaises(ValueError):
            expand_special_types(["invalid_type"])


class TestGPUAlgoTest(unittest.TestCase):
    """Test GPUAlgoTest class (basic validation only)."""

    def test_invalid_bin_dir(self):
        """Test that invalid bin directory raises error."""
        with self.assertRaises(FileNotFoundError):
            GPUAlgoTest("/nonexistent/directory")


if __name__ == "__main__":
    unittest.main()
