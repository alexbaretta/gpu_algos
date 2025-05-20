#!/bin/bash

# Function to handle errors
handle_error() {
    echo "[ERROR] $1"
    exit 1
}

# Check if we're in the project root (where CMakeLists.txt should be)
if [ ! -f "CMakeLists.txt" ]; then
    handle_error "This script must be run from the project root directory (where CMakeLists.txt is located)"
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "[INFO] Creating build directory..."
    mkdir -p build || handle_error "Failed to create build directory"
fi
cd build || handle_error "Failed to enter build directory"
echo "[INFO] Running CMake..."
cmake .. || handle_error "CMake configuration failed"


echo "[DONE] Run 'make' to build the project."
