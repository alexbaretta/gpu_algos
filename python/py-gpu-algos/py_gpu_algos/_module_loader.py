# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/py_gpu_algos/_module_loader.py

"""
Module loader for py-gpu-algos

This module dynamically finds and loads compiled CUDA modules from the build directory.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any

def _find_build_directory() -> Optional[Path]:
    """Find the build directory relative to the project root."""
    # Start from the current file's directory and work up to find the project root
    current_dir = Path(__file__).parent
    project_root = None

    # Look for the project root (contains CMakeLists.txt)
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "CMakeLists.txt").exists():
            project_root = parent.parent.parent  # Go up to the main project root
            break

    if project_root is None:
        return None

    # Look for build directories
    builds_dir = project_root / "builds"
    if not builds_dir.exists():
        return None

    # Prefer release over debug
    release_dir = builds_dir / "release" / "python" / "py-gpu-algos"
    debug_dir = builds_dir / "debug" / "python" / "py-gpu-algos"

    if release_dir.exists():
        return release_dir
    elif debug_dir.exists():
        return debug_dir

    return None

def load_cuda_module(module_name: str) -> Optional[Any]:
    """
    Load a CUDA module from the build directory.

    Args:
        module_name: Name of the module to load (e.g., '_matrix_ops_cuda')

    Returns:
        The loaded module or None if not found
    """
    build_dir = _find_build_directory()
    if build_dir is None:
        return None

    # Look for the .so file
    so_files = list(build_dir.glob(f"{module_name}*.so"))
    if not so_files:
        return None

    # Use the first found file
    so_file = so_files[0]

    try:
        # Load the module using importlib
        spec = importlib.util.spec_from_file_location(module_name, so_file)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module
    except Exception as e:
        print(f"Warning: Failed to load {module_name} from {so_file}: {e}")
        return None

# Cache for loaded modules
_loaded_modules: Dict[str, Any] = {}

def get_cuda_module(module_name: str) -> Optional[Any]:
    """
    Get a CUDA module, loading it if necessary.

    Args:
        module_name: Name of the module to get

    Returns:
        The module or None if not available
    """
    if module_name not in _loaded_modules:
        _loaded_modules[module_name] = load_cuda_module(module_name)

    return _loaded_modules[module_name]
