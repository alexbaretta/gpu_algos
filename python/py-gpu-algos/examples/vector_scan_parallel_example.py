#!/usr/bin/env python3
"""
Example: Vector Scan Parallel Kernel

This example demonstrates how to use the vector_scan_parallel kernel
for computing parallel scan operations (sum, max, min, prod).
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with different scan operations."""
    print("=== Basic Usage: vector_scan_parallel ===")

    # Create example vector
    vec = np.array([2.0, 3.0, 1.0, 4.0, 2.0, 5.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print(f"Vector shape: {vec.shape}, dtype: {vec.dtype}")
    print()

    # Test all scan operations
    operations = ["sum", "max", "min", "prod"]

    for op in operations:
        print(f"Scan operation: {op}")

        # GPU computation
        start_time = time.time()
        result_gpu = py_gpu_algos.vector_scan_parallel(vec, op)
        gpu_time = time.time() - start_time

        # NumPy reference
        start_time = time.time()
        if op == "sum":
            result_numpy = np.cumsum(vec)
        elif op == "max":
            result_numpy = np.maximum.accumulate(vec)
        elif op == "min":
            result_numpy = np.minimum.accumulate(vec)
        elif op == "prod":
            result_numpy = np.cumprod(vec)
        numpy_time = time.time() - start_time

        # Compare results
        max_error = np.max(np.abs(result_gpu - result_numpy))
        speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"  GPU result:   {result_gpu}")
        print(f"  NumPy result: {result_numpy}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  GPU time: {gpu_time*1000:.2f} ms, Speedup: {speedup:.2f}x")
        print()

def operation_comparison():
    """Compare different scan operations side by side."""
    print("=== Operation Comparison ===")

    # Create test vector with interesting properties
    vec = np.array([2.0, -1.0, 3.0, 0.5, -2.0, 4.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print()

    # Compute all operations
    operations = {
        "sum": py_gpu_algos.vector_scan_parallel(vec, "sum"),
        "max": py_gpu_algos.vector_scan_parallel(vec, "max"),
        "min": py_gpu_algos.vector_scan_parallel(vec, "min"),
        "prod": py_gpu_algos.vector_scan_parallel(vec, "prod")
    }

    # Display results in table format
    print("Pos | Value |  Sum  |  Max  |  Min  | Prod ")
    print("----|-------|-------|-------|-------|------")
    for i, val in enumerate(vec):
        row = f"{i:>3} | {val:>5.1f} |"
        for op in ["sum", "max", "min", "prod"]:
            row += f" {operations[op][i]:>5.1f} |"
        print(row)

    print()

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # Note: scan operations have limited type support
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        # Create vector appropriate for the dtype
        if dtype in [np.int32, np.int64]:
            vec = np.array([2, 3, 1, 4], dtype=dtype)
        else:
            vec = np.array([2.0, 3.0, 1.0, 4.0], dtype=dtype)

        # Test sum operation (most stable)
        try:
            result = py_gpu_algos.vector_scan_parallel(vec, "sum")
            reference = np.cumsum(vec)

            print(f"  Input:  {vec}")
            print(f"  Result: {result}")
            print(f"  NumPy:  {reference}")

            # Check accuracy
            if np.issubdtype(dtype, np.integer):
                accuracy_ok = np.array_equal(result, reference)
                print(f"  Exact match: {accuracy_ok}")
            else:
                max_error = np.max(np.abs(result - reference))
                print(f"  Max error: {max_error:.2e}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create int32 vector
    vec = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    print(f"Input vector: {vec} ({vec.dtype})")
    print()

    # Test different operations with low-level functions
    operations = ["sum", "max", "min", "prod"]

    for op in operations:
        low_level_func_name = f"vector_scan_parallel_{op}_int32"

        if hasattr(py_gpu_algos, low_level_func_name):
            print(f"Operation: {op}")

            # High-level function
            result_high = py_gpu_algos.vector_scan_parallel(vec, op)

            # Low-level function
            low_level_func = getattr(py_gpu_algos, low_level_func_name)
            result_low = low_level_func(vec)

            print(f"  High-level: {result_high}")
            print(f"  Low-level:  {result_low}")
            print(f"  Match: {np.array_equal(result_high, result_low)}")
            print()
        else:
            print(f"Operation {op}: Low-level function not available")
            print()

def mathematical_properties():
    """Demonstrate mathematical properties of scan operations."""
    print("=== Mathematical Properties ===")

    vec = np.array([2.0, 3.0, 1.0, 4.0, 2.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print()

    # Sum scan properties
    sum_scan = py_gpu_algos.vector_scan_parallel(vec, "sum")
    print("Sum scan properties:")
    print(f"  Result: {sum_scan}")
    print(f"  Last element equals total sum: {sum_scan[-1]:.1f} == {np.sum(vec):.1f}")
    print(f"  Differences recover original: {np.allclose(np.diff(sum_scan), vec[1:])}")
    print()

    # Max scan properties
    max_scan = py_gpu_algos.vector_scan_parallel(vec, "max")
    print("Max scan properties:")
    print(f"  Result: {max_scan}")
    print(f"  Non-decreasing: {np.all(max_scan[1:] >= max_scan[:-1])}")
    print(f"  Last element equals global max: {max_scan[-1]:.1f} == {np.max(vec):.1f}")
    print()

    # Min scan properties
    min_scan = py_gpu_algos.vector_scan_parallel(vec, "min")
    print("Min scan properties:")
    print(f"  Result: {min_scan}")
    print(f"  Non-increasing: {np.all(min_scan[1:] <= min_scan[:-1])}")
    print(f"  Last element equals global min: {min_scan[-1]:.1f} == {np.min(vec):.1f}")
    print()

    # Product scan properties
    prod_scan = py_gpu_algos.vector_scan_parallel(vec, "prod")
    print("Product scan properties:")
    print(f"  Result: {prod_scan}")
    print(f"  Last element equals total product: {prod_scan[-1]:.1f} == {np.prod(vec):.1f}")
    print()

def performance_analysis():
    """Analyze performance across operations and sizes."""
    print("=== Performance Analysis ===")

    sizes = [1000, 10000, 100000]
    operations = ["sum", "max", "min"]  # Skip prod to avoid overflow

    for size in sizes:
        print(f"Vector size {size}:")

        # Create random vector
        np.random.seed(42)
        vec = np.random.randn(size).astype(np.float32)

        for op in operations:
            # GPU timing
            start_time = time.time()
            result_gpu = py_gpu_algos.vector_scan_parallel(vec, op)
            gpu_time = time.time() - start_time

            # NumPy timing
            start_time = time.time()
            if op == "sum":
                result_numpy = np.cumsum(vec)
            elif op == "max":
                result_numpy = np.maximum.accumulate(vec)
            elif op == "min":
                result_numpy = np.minimum.accumulate(vec)
            numpy_time = time.time() - start_time

            # Analysis
            speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')
            accuracy = np.allclose(result_gpu, result_numpy, rtol=1e-5)

            print(f"  {op:>3}: GPU {gpu_time*1000:6.2f}ms, NumPy {numpy_time*1000:6.2f}ms, "
                  f"speedup {speedup:5.2f}x, accurate: {accuracy}")

        print()

def product_scan_considerations():
    """Special considerations for product scan."""
    print("=== Product Scan Considerations ===")

    print("1. Overflow concerns with integer types:")
    vec_int = np.array([2, 3, 4, 5], dtype=np.int32)
    prod_result = py_gpu_algos.vector_scan_parallel(vec_int, "prod")
    print(f"  Input: {vec_int}")
    print(f"  Product scan: {prod_result}")
    print(f"  Manual check: [2, 2*3=6, 6*4=24, 24*5=120]")
    print()

    print("2. Precision with floating point:")
    vec_float = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32)
    prod_result_float = py_gpu_algos.vector_scan_parallel(vec_float, "prod")
    numpy_prod = np.cumprod(vec_float)

    print(f"  Input: {vec_float}")
    print(f"  GPU result:  {prod_result_float}")
    print(f"  NumPy result: {numpy_prod}")
    print(f"  Max error: {np.max(np.abs(prod_result_float - numpy_prod)):.2e}")
    print()

    print("3. Handling zeros:")
    vec_with_zero = np.array([2.0, 3.0, 0.0, 4.0, 5.0], dtype=np.float32)
    prod_with_zero = py_gpu_algos.vector_scan_parallel(vec_with_zero, "prod")
    print(f"  Input with zero: {vec_with_zero}")
    print(f"  Product scan: {prod_with_zero}")
    print("  Note: Once zero appears, all subsequent products are zero")
    print()

def comparison_with_specialized_functions():
    """Compare generic scan with specialized functions."""
    print("=== Comparison with Specialized Functions ===")

    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print()

    # Compare scan sum with cumsum functions
    scan_sum = py_gpu_algos.vector_scan_parallel(vec, "sum")
    cumsum_serial = py_gpu_algos.vector_cumsum_serial(vec)
    cumsum_parallel = py_gpu_algos.vector_cumsum_parallel(vec)

    print("Sum operations comparison:")
    print(f"  scan_parallel(sum): {scan_sum}")
    print(f"  cumsum_serial:      {cumsum_serial}")
    print(f"  cumsum_parallel:    {cumsum_parallel}")
    print(f"  All match: {np.allclose(scan_sum, cumsum_serial) and np.allclose(scan_sum, cumsum_parallel)}")
    print()

    # Compare scan max with cummax
    scan_max = py_gpu_algos.vector_scan_parallel(vec, "max")
    cummax_parallel = py_gpu_algos.vector_cummax_parallel(vec)

    print("Max operations comparison:")
    print(f"  scan_parallel(max): {scan_max}")
    print(f"  cummax_parallel:    {cummax_parallel}")
    print(f"  Match: {np.allclose(scan_max, cummax_parallel)}")
    print()

def main():
    """Run all examples."""
    print("Vector Scan Parallel Kernel Examples")
    print("=" * 50)

    try:
        basic_usage()
        operation_comparison()
        different_data_types()
        low_level_functions()
        mathematical_properties()
        performance_analysis()
        product_scan_considerations()
        comparison_with_specialized_functions()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
