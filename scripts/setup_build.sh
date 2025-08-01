#!/bin/bash
# Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
