#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: python/py-gpu-algos/build.sh

# Build script for py-gpu-algos Python package
# This script builds the package using CMake integration with the main project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}py-gpu-algos Build Script${NC}"
echo "========================"

# Get script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    echo -e "${RED}Error: Cannot find main CMakeLists.txt at $PROJECT_ROOT${NC}"
    echo "Make sure you're running this script from the py-gpu-algos directory"
    exit 1
fi

# Create build directory
BUILD_DIR="$PROJECT_ROOT/builds/py-gpu-algos"
echo -e "${YELLOW}Creating build directory: $BUILD_DIR${NC}"
mkdir -p "$BUILD_DIR"

# Navigate to build directory
cd "$BUILD_DIR"

# Check for CUDA
echo -e "${YELLOW}Checking for CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA found: $(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')${NC}"
else
    echo -e "${RED}✗ CUDA not found. Make sure CUDA toolkit is installed and nvcc is in PATH${NC}"
    exit 1
fi

# Check for Python and dependencies
echo -e "${YELLOW}Checking Python environment...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi

# Check for pybind11
echo -e "${YELLOW}Checking for pybind11...${NC}"
if python3 -c "import pybind11" &> /dev/null; then
    PYBIND11_VERSION=$(python3 -c "import pybind11; print(pybind11.__version__)")
    echo -e "${GREEN}✓ pybind11 found: $PYBIND11_VERSION${NC}"
else
    echo -e "${RED}✗ pybind11 not found. Install with: pip install pybind11${NC}"
    exit 1
fi

# Check for numpy
echo -e "${YELLOW}Checking for numpy...${NC}"
if python3 -c "import numpy" &> /dev/null; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    echo -e "${GREEN}✓ numpy found: $NUMPY_VERSION${NC}"
else
    echo -e "${RED}✗ numpy not found. Install with: pip install numpy${NC}"
    exit 1
fi

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake "$PROJECT_ROOT" \
    -DBUILD_PYTHON_PACKAGE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    "$@"  # Pass any additional arguments to cmake

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configuration completed${NC}"

# Build the project
echo -e "${YELLOW}Building py-gpu-algos...${NC}"
cmake --build . --target py-gpu-algos-modules -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build completed successfully${NC}"

# Check if the module was built
EXPECTED_MODULE="$BUILD_DIR/python/py-gpu-algos/_matrix_ops_cuda*.so"
if ls $EXPECTED_MODULE 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓ Python module built: $(ls $EXPECTED_MODULE)${NC}"
else
    echo -e "${RED}✗ Python module not found at expected location${NC}"
    echo "Expected: $EXPECTED_MODULE"
    exit 1
fi

# Optional: Run tests if requested
if [ "$1" = "test" ] || [ "$2" = "test" ]; then
    echo -e "${YELLOW}Running tests...${NC}"

    # Add build directory to Python path and run tests
    export PYTHONPATH="$BUILD_DIR/python/py-gpu-algos:$PYTHONPATH"
    cd "$SCRIPT_DIR"

    if python3 test_basic.py; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}🎉 py-gpu-algos build completed successfully!${NC}"
echo ""
echo "To use the package:"
echo "  export PYTHONPATH=\"$BUILD_DIR/python/py-gpu-algos:\$PYTHONPATH\""
echo "  python3 -c \"from py_gpu_algos import matrix_product_naive; print('Import successful!')\""
echo ""
echo "To run tests:"
echo "  $0 test"
