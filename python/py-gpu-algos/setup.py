# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/setup.py

"""
Setup script for py-gpu-algos Python package
"""

import os
import subprocess
import sys
import shutil
import glob
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

# Package metadata
PACKAGE_NAME = "py-gpu-algos"
VERSION = "0.1.0"
DESCRIPTION = "Python bindings for GPU algorithms (CUDA and HIP)"
AUTHOR = "Alessandro Baretta"
EMAIL = "alessandro@example.com"

# Get the long description from README
current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text(encoding="utf-8")

class CMakeBuildExt(build_ext):
    """Custom build extension that uses project build scripts to build the C++ extensions."""

    user_options = build_ext.user_options + [
        ('debug', None, 'Build with debug configuration'),
        ('release', None, 'Build with release configuration'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.debug = None
        self.release = None

    def finalize_options(self):
        super().finalize_options()
        # Default to release if neither specified
        if not self.debug and not self.release:
            self.release = True

    def run(self):
        """Run the build process using project build scripts."""
        # Determine build type from options
        if self.debug:
            build_type = "debug"
        else:
            build_type = "release"

        print(f"Building with {build_type} configuration...")

        # Find project root (should be ../../ from setup_original.py location)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent

        # Find the appropriate build script
        build_script = project_root / "scripts" / f"{build_type}_build.sh"

        if not build_script.exists():
            raise RuntimeError(f"Build script not found: {build_script}")

        if not build_script.is_file():
            raise RuntimeError(f"Build script is not a file: {build_script}")

        # Make sure build script is executable
        os.chmod(build_script, 0o755)

        print(f"Running build script: {build_script}")

        # Run the build script from project root
        try:
            # Add target for Python modules specifically
            subprocess.check_call(
                [str(build_script), "--target", "py-gpu-algos-modules"],
                cwd=str(project_root),
                env=os.environ.copy()
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build script failed with exit code {e.returncode}")

        # Find built Python modules in the build directory
        build_output_dir = project_root / "builds" / build_type
        module_pattern = str(build_output_dir / "python" / "py-gpu-algos" / "*.so")
        built_modules = glob.glob(module_pattern)

        if not built_modules:
            raise RuntimeError(f"No Python modules found in {module_pattern}")

        # Copy built modules to package directory
        package_dir = current_dir / "py_gpu_algos"
        package_dir.mkdir(exist_ok=True)

        for module_path in built_modules:
            module_name = Path(module_path).name
            dest_path = package_dir / module_name
            print(f"Copying {module_path} to {dest_path}")
            shutil.copy2(module_path, dest_path)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    # Include the compiled CUDA module within the package
    package_data={
        'py_gpu_algos': ['*.so'],
    },
    include_package_data=True,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
    },
    cmdclass={"build_ext": CMakeBuildExt},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="gpu cuda hip matrix computation",
    zip_safe=False,
)
