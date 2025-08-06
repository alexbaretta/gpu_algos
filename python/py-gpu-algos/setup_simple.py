"""
Setup script for py-gpu-algos Python package
"""

from pathlib import Path
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

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # This will find py_gpu_algos package
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    # Include the compiled CUDA module within the package
    package_data={
        'py_gpu_algos': ['*.so'],
    },
    include_package_data=True,
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
