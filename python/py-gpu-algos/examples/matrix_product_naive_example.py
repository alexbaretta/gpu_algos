#!/usr/bin/env python3
"""
Example: Matrix Product Naive Kernel

This example demonstrates how to use the matrix_product_naive kernel
with different data types and sizes.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with float32 matrices."""
    print("=== Basic Usage: matrix_product_naive ===")

    # Create example matrices
    m, k, n = 100, 80, 120
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    print(f"Matrix A: {a.shape} ({a.dtype})")
    print(f"Matrix B: {b.shape} ({b.dtype})")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.matrix_product_naive(a, b)
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

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # Test different dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        # Create matrices
        if dtype in [np.int32, np.int64]:
            # For integers, use smaller values to avoid overflow
            a = np.random.randint(-10, 10, (32, 24), dtype=dtype)
            b = np.random.randint(-10, 10, (24, 40), dtype=dtype)
        else:
            a = np.random.randn(32, 24).astype(dtype)
            b = np.random.randn(24, 40).astype(dtype)

        # Compute
        result = py_gpu_algos.matrix_product_naive(a, b)
        reference = np.dot(a, b)

        # Check accuracy
        if dtype in [np.int32, np.int64]:
            accuracy_ok = np.array_equal(result, reference)
            print(f"  Exact match: {accuracy_ok}")
        else:
            max_error = np.max(np.abs(result - reference))
            rel_error = max_error / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0
            print(f"  Max error: {max_error:.2e}, Relative error: {rel_error:.2e}")

        print(f"  Shape: {result.shape}")
        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create float32 matrices
    a = np.random.randn(50, 30).astype(np.float32)
    b = np.random.randn(30, 70).astype(np.float32)

    # High-level function (automatic dispatch)
    result_high = py_gpu_algos.matrix_product_naive(a, b)

    # Low-level function (explicit type)
    result_low = py_gpu_algos.matrix_product_naive_float32(a, b)

    # Compare
    print(f"High-level result shape: {result_high.shape}")
    print(f"Low-level result shape: {result_low.shape}")
    print(f"Results match: {np.allclose(result_high, result_low)}")
    print()

def different_matrix_shapes():
    """Example with various matrix shapes."""
    print("=== Different Matrix Shapes ===")

    shapes = [
        (64, 64, 64),    # Square
        (100, 1, 50),    # Very thin middle
        (1, 100, 1),     # Very thin outer
        (200, 80, 120),  # Rectangular
    ]

    for m, k, n in shapes:
        print(f"Testing shape ({m}, {k}) @ ({k}, {n}):")

        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        start_time = time.time()
        result = py_gpu_algos.matrix_product_naive(a, b)
        elapsed = time.time() - start_time

        # Verify correctness
        reference = np.dot(a, b)
        max_error = np.max(np.abs(result - reference))

        print(f"  Result: {result.shape}, Time: {elapsed*1000:.2f}ms, Error: {max_error:.2e}")
        print()

def performance_comparison():
    """Performance comparison across different sizes."""
    print("=== Performance Comparison ===")

    sizes = [64, 128, 256, 512]

    for size in sizes:
        print(f"Square matrices ({size}x{size}):")

        # Create matrices
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        # GPU timing
        start_time = time.time()
        result_gpu = py_gpu_algos.matrix_product_naive(a, b)
        gpu_time = time.time() - start_time

        # CPU timing
        start_time = time.time()
        result_cpu = np.dot(a, b)
        cpu_time = time.time() - start_time

        # Analysis
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        accuracy = np.allclose(result_gpu, result_cpu, rtol=1e-5)

        print(f"  GPU: {gpu_time*1000:.2f}ms, CPU: {cpu_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x, Accurate: {accuracy}")
        print()

def main():
    """Run all examples."""
    print("Matrix Product Naive Kernel Examples")
    print("=" * 50)

    try:
        basic_usage()
        different_data_types()
        low_level_functions()
        different_matrix_shapes()
        performance_comparison()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
