#!/bin/bash

# Do not call exit here, as this script is sourced

# Function to handle errors
handle_error() {
    echo "[ERROR] $1"
}

# Check if we're in the project root (where CMakeLists.txt should be)
if [ ! -f "CMakeLists.txt" ]; then
    handle_error "This script must be run from the project root directory (where CMakeLists.txt is located)"
fi

# Parse command line arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            read -p "Are you sure you want to clean the build directory? (y/N): " confirm
            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                CLEAN_BUILD=true
            else
                echo "[INFO] Clean build cancelled"
            fi
            ;;
    esac
done

# Handle clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "[INFO] Cleaning build directory..."
    rm -rf build || handle_error "Failed to remove build directory"
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
