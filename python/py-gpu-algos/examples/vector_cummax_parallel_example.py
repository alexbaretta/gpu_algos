#!/usr/bin/env python3
"""
Example: Vector Cumulative Maximum Parallel Kernel

This example demonstrates how to use the vector_cummax_parallel kernel
for computing cumulative maximums using a parallel algorithm.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with parallel cumulative maximum."""
    print("=== Basic Usage: vector_cummax_parallel ===")

    # Create example vector with interesting pattern
    vec = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)

    print(f"Input vector: {vec}")
    print(f"Vector shape: {vec.shape}, dtype: {vec.dtype}")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.vector_cummax_parallel(vec)
    gpu_time = time.time() - start_time

    # NumPy reference
    start_time = time.time()
    result_numpy = np.maximum.accumulate(vec)
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

        # Create vector with clear maximum progression
        if dtype in [np.int32, np.int64, np.uint32]:
            vec = np.array([5, 2, 8, 1, 9, 3, 7, 4], dtype=dtype)
        else:
            vec = np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0], dtype=dtype)

        # Compute cumulative maximum
        result = py_gpu_algos.vector_cummax_parallel(vec)
        reference = np.maximum.accumulate(vec)

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

def mathematical_properties():
    """Demonstrate mathematical properties of cumulative maximum."""
    print("=== Mathematical Properties ===")

    # Create test vector with known pattern
    vec = np.array([2.0, 5.0, 1.0, 8.0, 3.0, 6.0, 4.0], dtype=np.float32)
    cummax_result = py_gpu_algos.vector_cummax_parallel(vec)

    print(f"Original vector:    {vec}")
    print(f"Cumulative maximum: {cummax_result}")

    # Property 1: Non-decreasing sequence
    is_non_decreasing = np.all(cummax_result[1:] >= cummax_result[:-1])
    print(f"Property 1 (non-decreasing): {is_non_decreasing}")

    # Property 2: Each element is max of prefix
    print("Property 2 (prefix maximums):")
    for i in range(len(vec)):
        expected = np.max(vec[:i+1])
        actual = cummax_result[i]
        match = np.isclose(expected, actual)
        print(f"  Position {i}: max(vec[:{i+1}]) = {expected:.1f}, cummax[{i}] = {actual:.1f} ✓" if match else f"  Position {i}: MISMATCH")

    # Property 3: Last element equals global maximum
    global_max = np.max(vec)
    last_cummax = cummax_result[-1]
    print(f"Property 3 (global max): {global_max:.1f} == {last_cummax:.1f} -> {np.isclose(global_max, last_cummax)}")

    # Property 4: Idempotency - cummax of cummax equals cummax
    cummax_twice = py_gpu_algos.vector_cummax_parallel(cummax_result)
    idempotent = np.allclose(cummax_result, cummax_twice)
    print(f"Property 4 (idempotent): {idempotent}")

    print()

def edge_cases():
    """Test edge cases and boundary conditions."""
    print("=== Edge Cases ===")

    # Single element
    single = np.array([42.0], dtype=np.float32)
    result_single = py_gpu_algos.vector_cummax_parallel(single)
    print(f"Single element: {single} -> {result_single}")

    # Two elements
    two_inc = np.array([10.0, 20.0], dtype=np.float32)
    result_two_inc = py_gpu_algos.vector_cummax_parallel(two_inc)
    print(f"Two increasing: {two_inc} -> {result_two_inc}")

    two_dec = np.array([20.0, 10.0], dtype=np.float32)
    result_two_dec = py_gpu_algos.vector_cummax_parallel(two_dec)
    print(f"Two decreasing: {two_dec} -> {result_two_dec}")

    # All same values
    same = np.array([7.0, 7.0, 7.0, 7.0], dtype=np.float32)
    result_same = py_gpu_algos.vector_cummax_parallel(same)
    print(f"All same: {same} -> {result_same}")

    # Strictly increasing
    increasing = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result_inc = py_gpu_algos.vector_cummax_parallel(increasing)
    print(f"Increasing: {increasing} -> {result_inc}")

    # Strictly decreasing
    decreasing = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    result_dec = py_gpu_algos.vector_cummax_parallel(decreasing)
    print(f"Decreasing: {decreasing} -> {result_dec}")

    # With negative values
    with_negatives = np.array([-5.0, 2.0, -1.0, 4.0, -3.0], dtype=np.float32)
    result_neg = py_gpu_algos.vector_cummax_parallel(with_negatives)
    print(f"With negatives: {with_negatives} -> {result_neg}")

    print()

def performance_comparison():
    """Compare performance across different sizes."""
    print("=== Performance Comparison ===")

    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        print(f"Vector size {size}:")

        # Create random vector
        np.random.seed(42)
        vec = np.random.randn(size).astype(np.float32)

        # GPU computation
        start_time = time.time()
        result_gpu = py_gpu_algos.vector_cummax_parallel(vec)
        gpu_time = time.time() - start_time

        # NumPy reference
        start_time = time.time()
        result_numpy = np.maximum.accumulate(vec)
        numpy_time = time.time() - start_time

        # Verify correctness
        max_error = np.max(np.abs(result_gpu - result_numpy))
        speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"  GPU: {gpu_time*1000:8.2f}ms, NumPy: {numpy_time*1000:8.2f}ms")
        print(f"  Speedup: {speedup:6.2f}x, Max error: {max_error:.2e}")
        print()

def comparison_with_cumsum():
    """Compare cummax with cumsum behavior."""
    print("=== Comparison with Cumulative Sum ===")

    # Create test vector
    vec = np.array([3.0, -1.0, 4.0, -2.0, 5.0], dtype=np.float32)

    # Compute both cumulative operations
    cumsum = py_gpu_algos.vector_cumsum_parallel(vec)
    cummax = py_gpu_algos.vector_cummax_parallel(vec)

    print(f"Vector:  {vec}")
    print(f"Cumsum:  {cumsum}")
    print(f"Cummax:  {cummax}")

    print("\nStep-by-step comparison:")
    for i in range(len(vec)):
        prefix = vec[:i+1]
        sum_val = np.sum(prefix)
        max_val = np.max(prefix)

        print(f"  pos {i}: prefix={prefix}, sum={sum_val:5.1f}, max={max_val:5.1f}")
        print(f"         cumsum[{i}]={cumsum[i]:5.1f}, cummax[{i}]={cummax[i]:5.1f}")

    print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create uint32 vector
    vec = np.array([10, 50, 30, 80, 20], dtype=np.uint32)

    print(f"Input vector: {vec} ({vec.dtype})")

    # High-level function (automatic dispatch)
    result_high = py_gpu_algos.vector_cummax_parallel(vec)

    # Low-level function (explicit type)
    result_low = py_gpu_algos.vector_cummax_parallel_uint32(vec)

    # Compare
    print(f"High-level result: {result_high}")
    print(f"Low-level result:  {result_low}")
    print(f"Results match: {np.array_equal(result_high, result_low)}")
    print()

def real_world_applications():
    """Demonstrate real-world applications of cumulative maximum."""
    print("=== Real-World Applications ===")

    print("1. Stock Price High-Water Mark:")
    # Simulate daily stock prices
    np.random.seed(42)
    prices = np.array([100.0, 105.0, 98.0, 110.0, 95.0, 115.0, 108.0], dtype=np.float32)
    high_water_mark = py_gpu_algos.vector_cummax_parallel(prices)

    print("  Day | Price | High-Water Mark")
    print("  ----|-------|----------------")
    for i, (price, hwm) in enumerate(zip(prices, high_water_mark)):
        print(f"  {i+1:>3} | {price:>5.1f} | {hwm:>5.1f}")

    print("\n2. Running Record Detection:")
    # Simulate race times (lower is better, so we'll use negative values)
    times = np.array([12.5, 11.8, 12.1, 11.3, 12.0, 10.9, 11.5], dtype=np.float32)
    neg_times = -times
    neg_cummax = py_gpu_algos.vector_cummax_parallel(neg_times)
    best_times = -neg_cummax

    print("  Race | Time | Best So Far | Record?")
    print("  -----|------|-------------|--------")
    for i, (time, best) in enumerate(zip(times, best_times)):
        is_record = i == 0 or time < best_times[i-1]
        record_marker = "YES" if is_record else "no"
        print(f"  {i+1:>4} | {time:>4.1f} | {best:>7.1f}   | {record_marker}")

    print("\n3. Peak Detection in Signal:")
    # Simulate a signal with peaks
    signal = np.array([2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 1.0, 6.0], dtype=np.float32)
    running_max = py_gpu_algos.vector_cummax_parallel(signal)

    print("  Time | Signal | Running Max | New Peak?")
    print("  -----|--------|-------------|----------")
    for i, (sig, rmax) in enumerate(zip(signal, running_max)):
        is_peak = i == 0 or sig > running_max[i-1]
        peak_marker = "YES" if is_peak else "no"
        print(f"  {i+1:>4} | {sig:>6.1f} | {rmax:>7.1f}     | {peak_marker}")

    print()

def main():
    """Run all examples."""
    print("Vector Cumulative Maximum Parallel Kernel Examples")
    print("=" * 60)

    try:
        basic_usage()
        different_data_types()
        mathematical_properties()
        edge_cases()
        performance_comparison()
        comparison_with_cumsum()
        low_level_functions()
        real_world_applications()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
