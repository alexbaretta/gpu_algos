#!/usr/bin/env python3
"""
Example: Matrix Product Tiled Kernel

This example demonstrates how to use the matrix_product_tiled kernel,
which uses tiled memory access patterns for better performance.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with tiled matrix multiplication."""
    print("=== Basic Usage: matrix_product_tiled ===")

    # Create example matrices (use tile-friendly sizes)
    m, k, n = 128, 96, 160
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    print(f"Matrix A: {a.shape} ({a.dtype})")
    print(f"Matrix B: {b.shape} ({b.dtype})")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.matrix_product_tiled(a, b)
    gpu_time = time.time() - start_time

    # NumPy reference
    start_time = time.time()
    result_numpy = np.dot(a, b)
    numpy_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(result_gpu - result_numpy))
    speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"Result shape: {result_gpu.shape}")
    print(f"Max error vs NumPy: {max_error:.2e}")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()

def compare_algorithms():
    """Compare naive vs tiled algorithms."""
    print("=== Algorithm Comparison: Naive vs Tiled ===")

    # Test with different sizes
    sizes = [64, 128, 256]

    for size in sizes:
        print(f"Size {size}x{size}:")

        # Create matrices
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        # Naive algorithm
        start_time = time.time()
        result_naive = py_gpu_algos.matrix_product_naive(a, b)
        naive_time = time.time() - start_time

        # Tiled algorithm
        start_time = time.time()
        result_tiled = py_gpu_algos.matrix_product_tiled(a, b)
        tiled_time = time.time() - start_time

        # NumPy reference
        start_time = time.time()
        result_numpy = np.dot(a, b)
        numpy_time = time.time() - start_time

        # Compare results
        naive_error = np.max(np.abs(result_naive - result_numpy))
        tiled_error = np.max(np.abs(result_tiled - result_numpy))
        algorithms_match = np.allclose(result_naive, result_tiled, rtol=1e-5)

        # Calculate speedups
        naive_speedup = naive_time / tiled_time if tiled_time > 0 else float('inf')
        vs_numpy_naive = numpy_time / naive_time if naive_time > 0 else float('inf')
        vs_numpy_tiled = numpy_time / tiled_time if tiled_time > 0 else float('inf')

        print(f"  Naive:  {naive_time*1000:6.2f}ms (speedup vs NumPy: {vs_numpy_naive:5.2f}x, error: {naive_error:.2e})")
        print(f"  Tiled:  {tiled_time*1000:6.2f}ms (speedup vs NumPy: {vs_numpy_tiled:5.2f}x, error: {tiled_error:.2e})")
        print(f"  Tiled vs Naive speedup: {naive_speedup:.2f}x")
        print(f"  Algorithms match: {algorithms_match}")
        print()

def supported_data_types():
    """Test supported data types for tiled algorithm."""
    print("=== Supported Data Types ===")

    # Tiled algorithm has limited type support
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        try:
            # Create matrices
            if dtype in [np.int32, np.int64]:
                a = np.random.randint(-5, 5, (32, 32), dtype=dtype)
                b = np.random.randint(-5, 5, (32, 32), dtype=dtype)
            else:
                a = np.random.randn(32, 32).astype(dtype)
                b = np.random.randn(32, 32).astype(dtype)

            # Test computation
            result = py_gpu_algos.matrix_product_tiled(a, b)
            reference = np.dot(a, b)

            # Check accuracy
            if dtype in [np.int32, np.int64]:
                accuracy_ok = np.array_equal(result, reference)
                print(f"  ✅ Supported - Exact match: {accuracy_ok}")
            else:
                max_error = np.max(np.abs(result - reference))
                print(f"  ✅ Supported - Max error: {max_error:.2e}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Test float32 specifically
    a = np.random.randn(64, 48).astype(np.float32)
    b = np.random.randn(48, 80).astype(np.float32)

    print("Testing float32 specifically:")

    # High-level function
    result_high = py_gpu_algos.matrix_product_tiled(a, b)

    # Low-level function
    result_low = py_gpu_algos.matrix_product_tiled_float32(a, b)

    # Compare
    match = np.allclose(result_high, result_low)
    print(f"  High-level vs low-level match: {match}")
    print(f"  Result shape: {result_high.shape}")
    print()

def memory_access_patterns():
    """Demonstrate tiled algorithm benefits with different access patterns."""
    print("=== Memory Access Pattern Analysis ===")

    # Test with matrices that should benefit from tiling
    shapes = [
        (64, 64, 64),    # Small, cache-friendly
        (128, 128, 128), # Medium
        (256, 256, 256), # Large
        (512, 128, 256), # Rectangular
    ]

    for m, k, n in shapes:
        print(f"Shape ({m}, {k}) @ ({k}, {n}):")

        # Create matrices
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        # Time both algorithms
        times = {}

        for algorithm_name, algorithm_func in [
            ("naive", py_gpu_algos.matrix_product_naive),
            ("tiled", py_gpu_algos.matrix_product_tiled)
        ]:
            try:
                start_time = time.time()
                result = algorithm_func(a, b)
                elapsed = time.time() - start_time
                times[algorithm_name] = elapsed

                # Verify result
                reference = np.dot(a, b)
                error = np.max(np.abs(result - reference))

                print(f"  {algorithm_name:>5}: {elapsed*1000:6.2f}ms (error: {error:.2e})")

            except Exception as e:
                print(f"  {algorithm_name:>5}: Failed - {e}")

        # Calculate improvement
        if "naive" in times and "tiled" in times:
            improvement = times["naive"] / times["tiled"]
            print(f"  Tiled improvement: {improvement:.2f}x")

        print()

def main():
    """Run all examples."""
    print("Matrix Product Tiled Kernel Examples")
    print("=" * 50)

    try:
        basic_usage()
        compare_algorithms()
        supported_data_types()
        low_level_functions()
        memory_access_patterns()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
