#!/usr/bin/env python3
"""
Example: Tensor Sort Bitonic Kernel

This example demonstrates how to use the tensor_sort_bitonic kernel
for in-place sorting of 3D tensors along specified dimensions.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with 3D tensor sorting."""
    print("=== Basic Usage: tensor_sort_bitonic ===")

    # Create example tensor with power-of-2 dimensions
    tensor = np.random.randint(0, 100, (8, 4, 16), dtype=np.int32)

    print(f"Original tensor shape: {tensor.shape} ({tensor.dtype})")
    print(f"Sample data (first slice): {tensor[0, 0, :8]}")

    # Sort along different dimensions
    for axis_name in ["rows", "cols", "sheets"]:
        tensor_copy = tensor.copy()

        print(f"\nSorting along '{axis_name}':")

        # GPU computation (in-place)
        start_time = time.time()
        py_gpu_algos.tensor_sort_bitonic(tensor_copy, axis_name)
        gpu_time = time.time() - start_time

        # Verify sorting
        if axis_name == "rows":
            axis_num = 0
            sample_slice = tensor_copy[:, 0, 0]
        elif axis_name == "cols":
            axis_num = 1
            sample_slice = tensor_copy[0, :, 0]
        else:  # sheets
            axis_num = 2
            sample_slice = tensor_copy[0, 0, :]

        is_sorted = np.all(sample_slice[:-1] <= sample_slice[1:])

        print(f"  Time: {gpu_time*1000:.2f} ms")
        print(f"  Sample sorted slice: {sample_slice}")
        print(f"  Is sorted: {is_sorted}")

    print()

def power_of_2_requirements():
    """Demonstrate power-of-2 dimension requirements."""
    print("=== Power-of-2 Requirements ===")

    print("Bitonic sort requires the sort dimension to be a power of 2")
    print()

    # Valid power-of-2 dimensions
    valid_shapes = [
        (8, 4, 16),   # 8=2^3, 4=2^2, 16=2^4
        (2, 8, 32),   # 2=2^1, 8=2^3, 32=2^5
        (16, 16, 4),  # All powers of 2
    ]

    print("✅ Valid shapes (all dimensions are powers of 2):")
    for shape in valid_shapes:
        tensor = np.random.randint(0, 50, shape, dtype=np.int32)
        try:
            py_gpu_algos.tensor_sort_bitonic(tensor, "rows")
            print(f"  {shape} - Success")
        except Exception as e:
            print(f"  {shape} - Failed: {e}")

    print()

    # Invalid non-power-of-2 dimensions
    invalid_shapes = [
        (7, 4, 8),    # 7 is not power of 2
        (8, 3, 16),   # 3 is not power of 2
        (8, 4, 15),   # 15 is not power of 2
    ]

    print("❌ Invalid shapes (contain non-power-of-2 dimensions):")
    for shape in invalid_shapes:
        tensor = np.random.randint(0, 50, shape, dtype=np.int32)
        try:
            py_gpu_algos.tensor_sort_bitonic(tensor, "rows")
            print(f"  {shape} - Unexpected success")
        except Exception as e:
            print(f"  {shape} - Expected failure: {str(e)[:50]}...")

    print()

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # Test different dtypes
    dtypes = [np.int32, np.int64, np.float32, np.float64, np.uint32]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        try:
            # Create tensor
            if dtype in [np.int32, np.int64, np.uint32]:
                tensor = np.random.randint(0, 100, (4, 8, 16), dtype=dtype)
            else:
                tensor = np.random.randn(4, 8, 16).astype(dtype) * 100

            original = tensor.copy()

            # Sort along columns (dimension 1, size 8)
            py_gpu_algos.tensor_sort_bitonic(tensor, "cols")

            # Verify sorting for one slice
            test_slice = tensor[0, :, 0]
            is_sorted = np.all(test_slice[:-1] <= test_slice[1:])

            print(f"  ✅ Supported - Sorting successful: {is_sorted}")
            print(f"  Sample: {original[0, :4, 0]} -> {tensor[0, :4, 0]}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def sorting_verification():
    """Verify sorting correctness against NumPy."""
    print("=== Sorting Verification ===")

    # Create test tensor
    np.random.seed(42)
    tensor = np.random.randint(0, 1000, (8, 4, 16), dtype=np.int32)

    for axis_name in ["rows", "cols", "sheets"]:
        print(f"Verifying sort along '{axis_name}':")

        # Get axis number
        if axis_name == "rows":
            axis_num = 0
        elif axis_name == "cols":
            axis_num = 1
        else:
            axis_num = 2

        # GPU sort (in-place)
        tensor_gpu = tensor.copy()
        py_gpu_algos.tensor_sort_bitonic(tensor_gpu, axis_name)

        # NumPy sort for comparison
        tensor_numpy = np.sort(tensor, axis=axis_num)

        # Compare results
        match = np.array_equal(tensor_gpu, tensor_numpy)

        print(f"  Results match NumPy: {match}")

        if not match:
            # Find first difference for debugging
            diff_indices = np.where(tensor_gpu != tensor_numpy)
            if len(diff_indices[0]) > 0:
                i, j, k = diff_indices[0][0], diff_indices[1][0], diff_indices[2][0]
                print(f"  First difference at ({i}, {j}, {k}): GPU={tensor_gpu[i,j,k]}, NumPy={tensor_numpy[i,j,k]}")

        print()

def performance_analysis():
    """Analyze sorting performance."""
    print("=== Performance Analysis ===")

    # Test different tensor sizes (all power-of-2)
    sizes = [
        (8, 4, 16),
        (16, 8, 32),
        (32, 16, 64),
        (64, 32, 128),
    ]

    for shape in sizes:
        print(f"Tensor shape {shape}:")

        # Create tensor
        tensor = np.random.randint(0, 10000, shape, dtype=np.int32)

        for axis_name in ["rows", "cols", "sheets"]:
            # Get axis info
            if axis_name == "rows":
                axis_num, sort_size = 0, shape[0]
            elif axis_name == "cols":
                axis_num, sort_size = 1, shape[1]
            else:
                axis_num, sort_size = 2, shape[2]

            # GPU timing
            tensor_gpu = tensor.copy()
            start_time = time.time()
            py_gpu_algos.tensor_sort_bitonic(tensor_gpu, axis_name)
            gpu_time = time.time() - start_time

            # NumPy timing
            start_time = time.time()
            tensor_numpy = np.sort(tensor, axis=axis_num)
            numpy_time = time.time() - start_time

            # Analysis
            speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

            print(f"  {axis_name:>6} (size {sort_size:>2}): GPU {gpu_time*1000:6.2f}ms, "
                  f"NumPy {numpy_time*1000:6.2f}ms, speedup {speedup:5.2f}x")

        print()

def in_place_operation_demo():
    """Demonstrate in-place operation characteristics."""
    print("=== In-Place Operation Demo ===")

    # Create tensor
    original = np.array([[[5, 2, 8, 1],
                         [7, 3, 6, 4]],
                        [[9, 1, 4, 6],
                         [2, 8, 3, 7]]], dtype=np.int32)

    print(f"Original tensor shape: {original.shape}")
    print("Original tensor:")
    for i in range(original.shape[0]):
        print(f"  Slice {i}: {original[i]}")

    # Sort along cols (dimension 1, size 2)
    tensor = original.copy()
    original_id = id(tensor)

    print(f"\nTensor ID before: {original_id}")
    py_gpu_algos.tensor_sort_bitonic(tensor, "cols")
    print(f"Tensor ID after:  {id(tensor)}")
    print(f"Same object: {id(tensor) == original_id}")

    print("\nAfter sorting along cols:")
    for i in range(tensor.shape[0]):
        print(f"  Slice {i}: {tensor[i]}")

    # Verify sorting within each column
    print("\nVerification - each column should be sorted:")
    for i in range(tensor.shape[0]):
        for k in range(tensor.shape[2]):
            col = tensor[i, :, k]
            is_sorted = np.all(col[:-1] <= col[1:])
            print(f"  Slice {i}, col {k}: {col} -> sorted: {is_sorted}")

    print()

def main():
    """Run all examples."""
    print("Tensor Sort Bitonic Kernel Examples")
    print("=" * 45)

    try:
        basic_usage()
        power_of_2_requirements()
        different_data_types()
        sorting_verification()
        performance_analysis()
        in_place_operation_demo()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
