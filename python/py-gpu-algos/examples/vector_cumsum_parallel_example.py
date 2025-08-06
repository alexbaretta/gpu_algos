#!/usr/bin/env python3
"""
Example: Vector Cumulative Sum Parallel Kernel

This example demonstrates how to use the vector_cumsum_parallel kernel
for computing cumulative sums using a parallel algorithm.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with parallel cumulative sum."""
    print("=== Basic Usage: vector_cumsum_parallel ===")

    # Create example vector
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print(f"Vector shape: {vec.shape}, dtype: {vec.dtype}")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.vector_cumsum_parallel(vec)
    gpu_time = time.time() - start_time

    # NumPy reference
    start_time = time.time()
    result_numpy = np.cumsum(vec)
    numpy_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(result_gpu - result_numpy))
    speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"GPU result:   {result_gpu}")
    print(f"NumPy result: {result_numpy}")
    print(f"Max error: {max_error:.2e}")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()

def compare_serial_vs_parallel():
    """Compare serial vs parallel cumsum algorithms."""
    print("=== Serial vs Parallel Algorithm Comparison ===")

    # Test with different sizes
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        print(f"Vector size {size}:")

        # Create random vector
        np.random.seed(42)
        vec = np.random.randn(size).astype(np.float32)

        # Serial algorithm
        start_time = time.time()
        result_serial = py_gpu_algos.vector_cumsum_serial(vec)
        serial_time = time.time() - start_time

        # Parallel algorithm
        start_time = time.time()
        result_parallel = py_gpu_algos.vector_cumsum_parallel(vec)
        parallel_time = time.time() - start_time

        # NumPy reference
        start_time = time.time()
        result_numpy = np.cumsum(vec)
        numpy_time = time.time() - start_time

        # Compare results
        serial_error = np.max(np.abs(result_serial - result_numpy))
        parallel_error = np.max(np.abs(result_parallel - result_numpy))
        algorithms_match = np.allclose(result_serial, result_parallel, rtol=1e-6)

        # Calculate speedups
        parallel_vs_serial = serial_time / parallel_time if parallel_time > 0 else float('inf')
        parallel_vs_numpy = numpy_time / parallel_time if parallel_time > 0 else float('inf')
        serial_vs_numpy = numpy_time / serial_time if serial_time > 0 else float('inf')

        print(f"  Serial:   {serial_time*1000:8.2f}ms (vs NumPy: {serial_vs_numpy:5.2f}x, error: {serial_error:.2e})")
        print(f"  Parallel: {parallel_time*1000:8.2f}ms (vs NumPy: {parallel_vs_numpy:5.2f}x, error: {parallel_error:.2e})")
        print(f"  Parallel speedup vs Serial: {parallel_vs_serial:.2f}x")
        print(f"  Algorithms match: {algorithms_match}")
        print()

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # Test different dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint32]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        # Create vector
        if dtype in [np.int32, np.int64, np.uint32]:
            vec = np.array([1, 3, 2, 4, 5, 1, 2, 3], dtype=dtype)
        else:
            vec = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 2.0, 3.0], dtype=dtype)

        # Compute cumulative sum
        result = py_gpu_algos.vector_cumsum_parallel(vec)
        reference = np.cumsum(vec)

        print(f"  Input:     {vec}")
        print(f"  GPU:       {result}")
        print(f"  NumPy:     {reference}")

        # Check accuracy
        if np.issubdtype(dtype, np.integer):
            accuracy_ok = np.array_equal(result, reference)
            print(f"  Exact match: {accuracy_ok}")
        else:
            max_error = np.max(np.abs(result - reference))
            print(f"  Max error: {max_error:.2e}")

        print()

def scalability_analysis():
    """Analyze parallel algorithm scalability."""
    print("=== Parallel Algorithm Scalability ===")

    # Test with power-of-2 sizes (optimal for parallel algorithms)
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    print("Power-of-2 sizes (optimal for parallel scan):")

    for size in sizes:
        # Create vector
        np.random.seed(42)
        vec = np.random.randn(size).astype(np.float32)

        # Time parallel algorithm
        start_time = time.time()
        result = py_gpu_algos.vector_cumsum_parallel(vec)
        gpu_time = time.time() - start_time

        # Time NumPy
        start_time = time.time()
        np_result = np.cumsum(vec)
        numpy_time = time.time() - start_time

        # Verify accuracy
        max_error = np.max(np.abs(result - np_result))
        speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

        # Calculate theoretical complexity
        # Parallel scan is O(log n) depth but O(n log n) work
        log_n = np.log2(size)

        print(f"  Size {size:>7} (2^{log_n:4.0f}): {gpu_time*1000:6.2f}ms "
              f"(speedup: {speedup:5.2f}x, error: {max_error:.2e})")

    print("\nNote: Parallel scan has O(log n) depth complexity")
    print()

def algorithm_demonstration():
    """Demonstrate the parallel scan algorithm concept."""
    print("=== Parallel Scan Algorithm Demonstration ===")

    # Use a small vector to show the algorithm concept
    vec = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)

    print(f"Input: {vec}")
    print("Parallel scan computes prefix sums using tree-based reduction")
    print()

    # Show the expected result
    result = py_gpu_algos.vector_cumsum_parallel(vec)
    numpy_result = np.cumsum(vec)

    print("Step-by-step prefix sums:")
    for i in range(len(vec)):
        prefix = vec[:i+1]
        expected = np.sum(prefix)
        actual = result[i]
        print(f"  Position {i}: sum({prefix}) = {expected} (GPU: {actual})")

    print(f"\nFinal result: {result}")
    print(f"NumPy result: {numpy_result}")
    print(f"Match: {np.array_equal(result, numpy_result)}")
    print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create float64 vector
    vec = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)

    print(f"Input vector: {vec} ({vec.dtype})")

    # High-level function (automatic dispatch)
    result_high = py_gpu_algos.vector_cumsum_parallel(vec)

    # Low-level function (explicit type)
    result_low = py_gpu_algos.vector_cumsum_parallel_float64(vec)

    # Compare
    print(f"High-level result: {result_high}")
    print(f"Low-level result:  {result_low}")
    print(f"Results match: {np.allclose(result_high, result_low)}")
    print()

def performance_optimization_tips():
    """Demonstrate performance optimization considerations."""
    print("=== Performance Optimization Tips ===")

    print("1. Vector Size Impact:")
    # Test both small and large vectors
    sizes = [100, 10000, 1000000]

    for size in sizes:
        vec = np.random.randn(size).astype(np.float32)

        start_time = time.time()
        result = py_gpu_algos.vector_cumsum_parallel(vec)
        gpu_time = time.time() - start_time

        start_time = time.time()
        np_result = np.cumsum(vec)
        numpy_time = time.time() - start_time

        efficiency = numpy_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  Size {size:>7}: GPU {gpu_time*1000:6.2f}ms, efficiency {efficiency:5.2f}x")

    print("\n2. Data Type Impact:")
    # Test different dtypes with same size
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    size = 100000

    for dtype in dtypes:
        if dtype in [np.int32, np.int64]:
            vec = np.random.randint(0, 100, size, dtype=dtype)
        else:
            vec = np.random.randn(size).astype(dtype)

        start_time = time.time()
        result = py_gpu_algos.vector_cumsum_parallel(vec)
        gpu_time = time.time() - start_time

        bytes_per_element = vec.itemsize
        bandwidth = (vec.nbytes * 2) / gpu_time / 1e9  # Read + write in GB/s

        print(f"  {dtype.__name__:>8}: {gpu_time*1000:6.2f}ms "
              f"({bytes_per_element} bytes/elem, {bandwidth:5.1f} GB/s)")

    print()

def main():
    """Run all examples."""
    print("Vector Cumulative Sum Parallel Kernel Examples")
    print("=" * 60)

    try:
        basic_usage()
        compare_serial_vs_parallel()
        different_data_types()
        scalability_analysis()
        algorithm_demonstration()
        low_level_functions()
        performance_optimization_tips()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
