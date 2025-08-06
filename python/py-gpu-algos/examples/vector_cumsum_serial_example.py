#!/usr/bin/env python3
"""
Example: Vector Cumulative Sum Serial Kernel

This example demonstrates how to use the vector_cumsum_serial kernel
for computing cumulative sums using a serial algorithm.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with cumulative sum."""
    print("=== Basic Usage: vector_cumsum_serial ===")

    # Create example vector
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print(f"Vector shape: {vec.shape}, dtype: {vec.dtype}")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.vector_cumsum_serial(vec)
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

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # Test different dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint32]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        # Create vector
        if dtype in [np.int32, np.int64, np.uint32]:
            vec = np.array([1, 3, 2, 4, 5, 1], dtype=dtype)
        else:
            vec = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 1.0], dtype=dtype)

        # Compute cumulative sum
        result = py_gpu_algos.vector_cumsum_serial(vec)
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

def different_vector_sizes():
    """Example with various vector sizes."""
    print("=== Different Vector Sizes ===")

    sizes = [10, 100, 1000, 10000, 100000]

    for size in sizes:
        print(f"Vector size {size}:")

        # Create random vector
        np.random.seed(42)  # For reproducibility
        vec = np.random.randn(size).astype(np.float32)

        # GPU computation
        start_time = time.time()
        result_gpu = py_gpu_algos.vector_cumsum_serial(vec)
        gpu_time = time.time() - start_time

        # NumPy reference
        start_time = time.time()
        result_numpy = np.cumsum(vec)
        numpy_time = time.time() - start_time

        # Verify correctness
        max_error = np.max(np.abs(result_gpu - result_numpy))
        speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"  GPU: {gpu_time*1000:8.2f}ms, NumPy: {numpy_time*1000:8.2f}ms")
        print(f"  Speedup: {speedup:6.2f}x, Max error: {max_error:.2e}")
        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create int32 vector
    vec = np.array([10, 20, 30, 40, 50], dtype=np.int32)

    print(f"Input vector: {vec} ({vec.dtype})")

    # High-level function (automatic dispatch)
    result_high = py_gpu_algos.vector_cumsum_serial(vec)

    # Low-level function (explicit type)
    result_low = py_gpu_algos.vector_cumsum_serial_int32(vec)

    # Compare
    print(f"High-level result: {result_high}")
    print(f"Low-level result:  {result_low}")
    print(f"Results match: {np.array_equal(result_high, result_low)}")
    print()

def mathematical_properties():
    """Demonstrate mathematical properties of cumulative sum."""
    print("=== Mathematical Properties ===")

    # Create test vector
    vec = np.array([2.0, -1.0, 3.0, -2.0, 4.0], dtype=np.float32)
    cumsum_result = py_gpu_algos.vector_cumsum_serial(vec)

    print(f"Original vector:   {vec}")
    print(f"Cumulative sum:    {cumsum_result}")

    # Property 1: Last element equals sum of all elements
    total_sum = np.sum(vec)
    last_cumsum = cumsum_result[-1]
    print(f"Total sum:         {total_sum}")
    print(f"Last cumsum:       {last_cumsum}")
    print(f"Property 1 (sum): {np.isclose(total_sum, last_cumsum)}")

    # Property 2: Differences give original vector
    if len(cumsum_result) > 1:
        diff = np.diff(cumsum_result)
        original_tail = vec[1:]

        print(f"Differences:       {diff}")
        print(f"Original tail:     {original_tail}")
        print(f"Property 2 (diff): {np.allclose(diff, original_tail)}")

    # Property 3: Each element is sum of prefix
    for i in range(len(vec)):
        expected = np.sum(vec[:i+1])
        actual = cumsum_result[i]
        print(f"  cumsum[{i}] = {actual:.1f}, sum(vec[:{i+1}]) = {expected:.1f}")

    print()

def edge_cases():
    """Test edge cases and boundary conditions."""
    print("=== Edge Cases ===")

    # Single element
    single = np.array([42.0], dtype=np.float32)
    result_single = py_gpu_algos.vector_cumsum_serial(single)
    print(f"Single element: {single} -> {result_single}")

    # Two elements
    two = np.array([10.0, 20.0], dtype=np.float32)
    result_two = py_gpu_algos.vector_cumsum_serial(two)
    print(f"Two elements: {two} -> {result_two}")

    # All zeros
    zeros = np.zeros(5, dtype=np.float32)
    result_zeros = py_gpu_algos.vector_cumsum_serial(zeros)
    print(f"All zeros: {zeros} -> {result_zeros}")

    # All ones
    ones = np.ones(5, dtype=np.float32)
    result_ones = py_gpu_algos.vector_cumsum_serial(ones)
    print(f"All ones: {ones} -> {result_ones}")

    # Alternating signs
    alternating = np.array([1, -1, 1, -1, 1], dtype=np.int32)
    result_alt = py_gpu_algos.vector_cumsum_serial(alternating)
    print(f"Alternating: {alternating} -> {result_alt}")

    print()

def performance_comparison():
    """Compare serial algorithm performance characteristics."""
    print("=== Performance Characteristics ===")

    # Test with different sizes to see scaling
    sizes = [1000, 10000, 100000, 1000000]

    print("Serial cumsum scaling analysis:")
    prev_gpu_time = None
    prev_size = None

    for size in sizes:
        # Create vector
        np.random.seed(42)
        vec = np.random.randn(size).astype(np.float32)

        # Time GPU
        start_time = time.time()
        result = py_gpu_algos.vector_cumsum_serial(vec)
        gpu_time = time.time() - start_time

        # Time NumPy
        start_time = time.time()
        np_result = np.cumsum(vec)
        numpy_time = time.time() - start_time

        # Calculate scaling
        if prev_gpu_time is not None:
            size_ratio = size / prev_size
            time_ratio = gpu_time / prev_gpu_time
            print(f"  Size {size:>7}: {gpu_time*1000:6.2f}ms (vs NumPy: {numpy_time/gpu_time:5.2f}x) "
                  f"[scaling: {size_ratio:.1f}x size -> {time_ratio:.2f}x time]")
        else:
            print(f"  Size {size:>7}: {gpu_time*1000:6.2f}ms (vs NumPy: {numpy_time/gpu_time:5.2f}x)")

        prev_gpu_time = gpu_time
        prev_size = size

    print("\nNote: Serial algorithm has O(n) complexity but limited parallelism")
    print()

def main():
    """Run all examples."""
    print("Vector Cumulative Sum Serial Kernel Examples")
    print("=" * 60)

    try:
        basic_usage()
        different_data_types()
        different_vector_sizes()
        low_level_functions()
        mathematical_properties()
        edge_cases()
        performance_comparison()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
