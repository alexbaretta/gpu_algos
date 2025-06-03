"""
Unit tests for CMake preset functionality.

Copyright (c) 2025 Alessandro Baretta
All rights reserved.
"""

import json
import tempfile
import unittest
from pathlib import Path

from gpu_algo_test import determine_bin_directory, find_cmake_binary_directory


class TestCMakePresets(unittest.TestCase):
    """Test CMake preset parsing and directory determination."""

    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cmake_root = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directories."""
        self.temp_dir.cleanup()

    def create_cmake_presets_file(self, presets_data, filename="CMakePresets.json"):
        """Helper to create a CMake presets file."""
        preset_file = self.cmake_root / filename
        with open(preset_file, "w") as f:
            json.dump(presets_data, f, indent=2)
        return preset_file

    def create_bin_directory(self, binary_dir_path):
        """Helper to create a bin directory structure."""
        bin_dir = self.cmake_root / binary_dir_path / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        return bin_dir

    def test_find_cmake_binary_directory_basic(self):
        """Test basic CMake preset parsing."""
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "debug", "binaryDir": "${sourceDir}/builds/debug"},
                {"name": "release", "binaryDir": "${sourceDir}/builds/release"},
            ],
        }

        self.create_cmake_presets_file(presets_data)

        # Test debug preset
        result = find_cmake_binary_directory(self.cmake_root, "debug")
        expected = self.cmake_root / "builds" / "debug"
        self.assertEqual(result, expected)

        # Test release preset
        result = find_cmake_binary_directory(self.cmake_root, "release")
        expected = self.cmake_root / "builds" / "release"
        self.assertEqual(result, expected)

    def test_find_cmake_binary_directory_variable_substitution(self):
        """Test CMake variable substitution."""
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "test_sourcedir", "binaryDir": "${sourceDir}/build"},
                {"name": "test_workspaceroot", "binaryDir": "${workspaceRoot}/build"},
            ],
        }

        self.create_cmake_presets_file(presets_data)

        # Test ${sourceDir} substitution
        result = find_cmake_binary_directory(self.cmake_root, "test_sourcedir")
        expected = self.cmake_root / "build"
        self.assertEqual(result, expected)

        # Test ${workspaceRoot} substitution
        result = find_cmake_binary_directory(self.cmake_root, "test_workspaceroot")
        expected = self.cmake_root / "build"
        self.assertEqual(result, expected)

    def test_find_cmake_binary_directory_relative_paths(self):
        """Test handling of relative paths in binaryDir."""
        presets_data = {
            "version": 3,
            "configurePresets": [{"name": "relative", "binaryDir": "build/debug"}],
        }

        self.create_cmake_presets_file(presets_data)

        result = find_cmake_binary_directory(self.cmake_root, "relative")
        expected = self.cmake_root / "build" / "debug"
        self.assertEqual(result, expected)

    def test_find_cmake_binary_directory_default_location(self):
        """Test default binary directory when binaryDir is not specified."""
        presets_data = {
            "version": 3,
            "configurePresets": [
                {
                    "name": "no_binary_dir"
                    # No binaryDir specified
                }
            ],
        }

        self.create_cmake_presets_file(presets_data)

        result = find_cmake_binary_directory(self.cmake_root, "no_binary_dir")
        expected = self.cmake_root / "build" / "no_binary_dir"
        self.assertEqual(result, expected)

    def test_find_cmake_binary_directory_user_presets(self):
        """Test reading from CMakeUserPresets.json."""
        # Create main presets file
        main_presets = {
            "version": 3,
            "configurePresets": [
                {"name": "main_preset", "binaryDir": "${sourceDir}/builds/main"}
            ],
        }
        self.create_cmake_presets_file(main_presets, "CMakePresets.json")

        # Create user presets file
        user_presets = {
            "version": 3,
            "configurePresets": [
                {"name": "user_preset", "binaryDir": "${sourceDir}/builds/user"}
            ],
        }
        self.create_cmake_presets_file(user_presets, "CMakeUserPresets.json")

        # Test main preset
        result = find_cmake_binary_directory(self.cmake_root, "main_preset")
        expected = self.cmake_root / "builds" / "main"
        self.assertEqual(result, expected)

        # Test user preset
        result = find_cmake_binary_directory(self.cmake_root, "user_preset")
        expected = self.cmake_root / "builds" / "user"
        self.assertEqual(result, expected)

    def test_find_cmake_binary_directory_preset_not_found(self):
        """Test error when preset is not found."""
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "existing_preset", "binaryDir": "${sourceDir}/builds/debug"}
            ],
        }

        self.create_cmake_presets_file(presets_data)

        with self.assertRaises(ValueError) as context:
            find_cmake_binary_directory(self.cmake_root, "nonexistent_preset")

        self.assertIn("Preset 'nonexistent_preset' not found", str(context.exception))
        self.assertIn("Available presets: ['existing_preset']", str(context.exception))

    def test_find_cmake_binary_directory_no_presets_file(self):
        """Test error when no preset files exist."""
        with self.assertRaises(FileNotFoundError) as context:
            find_cmake_binary_directory(self.cmake_root, "any_preset")

        self.assertIn("No CMake preset files found", str(context.exception))

    def test_find_cmake_binary_directory_invalid_json(self):
        """Test error handling for invalid JSON in preset files."""
        # Create invalid JSON file
        preset_file = self.cmake_root / "CMakePresets.json"
        with open(preset_file, "w") as f:
            f.write("{ invalid json }")

        with self.assertRaises(ValueError) as context:
            find_cmake_binary_directory(self.cmake_root, "any_preset")

        self.assertIn("Invalid CMake preset file", str(context.exception))

    def test_determine_bin_directory_direct_path(self):
        """Test determine_bin_directory with direct bin path."""
        # Create a bin directory
        bin_dir = self.cmake_root / "custom_bin"
        bin_dir.mkdir()

        result = determine_bin_directory(bin_path=str(bin_dir))
        self.assertEqual(result, bin_dir)

    def test_determine_bin_directory_direct_path_not_found(self):
        """Test error when direct bin path doesn't exist."""
        nonexistent_path = self.cmake_root / "nonexistent_bin"

        with self.assertRaises(FileNotFoundError) as context:
            determine_bin_directory(bin_path=str(nonexistent_path))

        self.assertIn("Binary directory not found", str(context.exception))

    def test_determine_bin_directory_cmake_preset(self):
        """Test determine_bin_directory with CMake preset logic."""
        # Create CMake presets file
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "debug", "binaryDir": "${sourceDir}/builds/debug"}
            ],
        }
        self.create_cmake_presets_file(presets_data)

        # Create the expected bin directory
        bin_dir = self.create_bin_directory("builds/debug")

        result = determine_bin_directory(
            cmake_root=str(self.cmake_root), preset="debug"
        )
        self.assertEqual(result, bin_dir)

    def test_determine_bin_directory_cmake_root_not_found(self):
        """Test error when CMake root directory doesn't exist."""
        nonexistent_root = self.cmake_root / "nonexistent"

        with self.assertRaises(FileNotFoundError) as context:
            determine_bin_directory(cmake_root=str(nonexistent_root))

        self.assertIn("CMake root directory not found", str(context.exception))

    def test_determine_bin_directory_bin_dir_not_found(self):
        """Test error when bin directory doesn't exist in build directory."""
        # Create CMake presets file
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "debug", "binaryDir": "${sourceDir}/builds/debug"}
            ],
        }
        self.create_cmake_presets_file(presets_data)

        # Create build directory but NOT the bin subdirectory
        build_dir = self.cmake_root / "builds" / "debug"
        build_dir.mkdir(parents=True)

        with self.assertRaises(FileNotFoundError) as context:
            determine_bin_directory(cmake_root=str(self.cmake_root), preset="debug")

        self.assertIn("Binary directory not found", str(context.exception))

    def test_determine_bin_directory_defaults(self):
        """Test determine_bin_directory with default values."""
        # Create CMake presets file in current working directory logic
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "debug", "binaryDir": "${sourceDir}/builds/debug"}
            ],
        }
        self.create_cmake_presets_file(presets_data)

        # Create the expected bin directory
        bin_dir = self.create_bin_directory("builds/debug")

        # Test with cmake_root=None (should use current directory logic)
        # and preset="debug" (default)
        result = determine_bin_directory(cmake_root=str(self.cmake_root))
        self.assertEqual(result, bin_dir)

    def test_determine_bin_directory_bin_path_overrides_preset(self):
        """Test that bin_path overrides CMake preset logic."""
        # Create CMake presets file
        presets_data = {
            "version": 3,
            "configurePresets": [
                {"name": "debug", "binaryDir": "${sourceDir}/builds/debug"}
            ],
        }
        self.create_cmake_presets_file(presets_data)

        # Create a different bin directory
        custom_bin = self.cmake_root / "custom_bin"
        custom_bin.mkdir()

        # bin_path should override the preset logic
        result = determine_bin_directory(
            bin_path=str(custom_bin), cmake_root=str(self.cmake_root), preset="debug"
        )
        self.assertEqual(result, custom_bin)


if __name__ == "__main__":
    unittest.main()
