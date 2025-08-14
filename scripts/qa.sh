#!/bin/bash

# CUDA Memory Access Safety Checker
# This script searches for potentially unsafe memory access patterns in CUDA kernels
# that could lead to integer overflow and memory access errors.

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

set -euo pipefail

# Configuration
KERNEL_DIR="include/cuda"

echo "=== Memory Access Safety Checker ==="
echo "Searching for unsafe memory access patterns..."
echo ""

# Function to print section headers
print_section() {
    echo "=== $1 ==="
}

# Function to print findings
print_finding() {
    echo "[UNSAFE]"
}

print_safe() {
    echo "[SAFE] $1"
}

# Pattern 1: auto variables computed with CUDA special variables (most dangerous)
print_section "Pattern 1: auto variables with CUDA special variables"
echo "Searching for: const auto var = blockIdx.x * blockDim.x + threadIdx.x"
echo ""

pattern1_count=0
while IFS= read -r -d '' file; do
    if grep -n "const auto.*=.*blockIdx.*\*.*blockDim.*\+.*threadIdx" "$file" > /dev/null; then
        print_finding
        grep -n "const auto.*=.*blockIdx.*\*.*blockDim.*\+.*threadIdx" "$file" | while read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            echo "$file($line_num): $content"
        done
        pattern1_count=$((pattern1_count + 1))
    fi
done < <(find "$KERNEL_DIR" -name "*.cuh" -print0)

if [ $pattern1_count -eq 0 ]; then
    print_safe "No unsafe auto patterns found"
fi
echo ""

# Pattern 2: long variables computed with 32-bit arithmetic
print_section "Pattern 2: long variables with 32-bit arithmetic"
echo "Searching for: const long var = blockIdx.x * blockDim.x + threadIdx.x"
echo ""

pattern2_count=0
while IFS= read -r -d '' file; do
    if grep -n "const long.*=.*blockIdx.*\*.*blockDim.*\+.*threadIdx" "$file" | grep -v "long(" > /dev/null; then
        print_finding
        grep -n "const long.*=.*blockIdx.*\*.*blockDim.*\+.*threadIdx" "$file" | grep -v "long(" | while read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            echo "$file($line_num): $content"
        done
        pattern2_count=$((pattern2_count + 1))
    fi
done < <(find "$KERNEL_DIR" -name "*.cuh" -print0)

if [ $pattern2_count -eq 0 ]; then
    print_safe "No unsafe long patterns found"
fi
echo ""

# Pattern 3: auto variables with threadIdx + blockIdx * blockDim
print_section "Pattern 3: auto variables with threadIdx + blockIdx * blockDim"
echo "Searching for: const auto var = threadIdx.x + blockIdx.x * blockDim.x"
echo ""

pattern3_count=0
while IFS= read -r -d '' file; do
    if grep -n "const auto.*=.*threadIdx.*\+.*blockIdx.*\*.*blockDim" "$file" > /dev/null; then
        print_finding
        grep -n "const auto.*=.*threadIdx.*\+.*blockIdx.*\*.*blockDim" "$file" | while read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            echo "$file($line_num): $content"
        done
        pattern3_count=$((pattern3_count + 1))
    fi
done < <(find "$KERNEL_DIR" -name "*.cuh" -print0)

if [ $pattern3_count -eq 0 ]; then
    print_safe "No unsafe auto + CUDA special variable patterns found"
fi
echo ""

# Pattern 4: auto variables for grid/block calculations
print_section "Pattern 4: auto variables for grid/block calculations"
echo "Searching for: const auto var = gridDim.x * blockDim.x"
echo ""

pattern4_count=0
while IFS= read -r -d '' file; do
    if grep -n "const auto.*=.*gridDim.*\*.*blockDim" "$file" | grep -v "(long)" > /dev/null; then
        print_finding
        grep -n "const auto.*=.*gridDim.*\*.*blockDim" "$file" | grep -v "(long)" | while read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            echo "$file($line_num): $content"
        done
        pattern4_count=$((pattern4_count + 1))
    fi
done < <(find "$KERNEL_DIR" -name "*.cuh" -print0)

if [ $pattern4_count -eq 0 ]; then
    print_safe "No unsafe grid/block calculation patterns found"
fi
echo ""

# Pattern 5: bid_grid patterns
print_section "Pattern 5: bid_grid patterns"
echo "Searching for: const auto bid_grid = blockIdx.x"
echo ""

pattern5_count=0
while IFS= read -r -d '' file; do
    if grep -n "const auto bid_grid = blockIdx" "$file" > /dev/null; then
        print_finding
        grep -n "const auto bid_grid = blockIdx" "$file" | while read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            echo "$file($line_num): $content"
        done
        pattern5_count=$((pattern5_count + 1))
    fi
done < <(find "$KERNEL_DIR" -name "*.cuh" -print0)

if [ $pattern5_count -eq 0 ]; then
    print_safe "No unsafe bid_grid patterns found"
fi
echo ""

# Summary
print_section "Summary"
total_issues=$((pattern1_count + pattern2_count + pattern3_count + pattern4_count + pattern5_count))

if [ $total_issues -eq 0 ]; then
    echo "No unsafe memory access patterns found!"
    echo "All CUDA kernels appear to use safe memory access patterns."
else
    echo "Found $total_issues potentially unsafe patterns:"
    echo "  Pattern 1 (auto + CUDA vars): $pattern1_count"
    echo "  Pattern 2 (long + 32-bit): $pattern2_count"
    echo "  Pattern 3 (auto + threadIdx): $pattern3_count"
    echo "  Pattern 4 (auto grid/block): $pattern4_count"
    echo "  Pattern 5 (bid_grid): $pattern5_count"
    echo ""
fi

echo ""
echo "=== Safe Pattern Reference ==="
echo "Safe patterns to use:"
echo "  const long tid_grid = long(blockIdx.x) * long(blockDim.x) + long(threadIdx.x);"
echo "  const long global_col = long(blockIdx.x) * long(blockDim.x) + long(threadIdx.x);"
echo "  for (long i = 0; i < n; ++i) { ... }"
echo "  const long nthreads = long(gridDim.x) * long(blockDim.x);"
echo ""

exit $total_issues
