#!/usr/bin/env python3
"""
Example: Matrix Transpose Striped Kernel

This example demonstrates how to use the matrix_transpose_striped kernel
for efficient matrix transposition on GPU.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with matrix transpose."""
    print("=== Basic Usage: matrix_transpose_striped ===")

    # Create example matrix
    m, n = 100, 80
    matrix = np.random.randn(m, n).astype(np.float32)

    print(f"Original matrix: {matrix.shape} ({matrix.dtype})")

    # GPU computation
    start_time = time.time()
    result_gpu = py_gpu_algos.matrix_transpose_striped(matrix)
    gpu_time = time.time() - start_time

    # NumPy reference
    start_time = time.time()
    result_numpy = matrix.T
    numpy_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(result_gpu - result_numpy))
    speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"Transposed shape: {result_gpu.shape}")
    print(f"Max error vs NumPy: {max_error:.2e}")
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

        # Create matrix
        if dtype in [np.int32, np.int64, np.uint32]:
            matrix = np.random.randint(0, 100, (32, 48), dtype=dtype)
        else:
            matrix = np.random.randn(32, 48).astype(dtype)

        # Compute transpose
        result = py_gpu_algos.matrix_transpose_striped(matrix)
        reference = matrix.T

        # Check accuracy
        if np.issubdtype(dtype, np.integer):
            accuracy_ok = np.array_equal(result, reference)
            print(f"  Exact match: {accuracy_ok}")
        else:
            max_error = np.max(np.abs(result - reference))
            print(f"  Max error: {max_error:.2e}")

        print(f"  Shape: {matrix.shape} -> {result.shape}")
        print()

def different_matrix_shapes():
    """Example with various matrix shapes."""
    print("=== Different Matrix Shapes ===")

    shapes = [
        (64, 64),     # Square
        (100, 50),    # Rectangular wide
        (50, 100),    # Rectangular tall
        (1, 1000),    # Very wide
        (1000, 1),    # Very tall
        (256, 128),   # Large rectangular
    ]

    for m, n in shapes:
        print(f"Testing shape ({m}, {n}):")

        matrix = np.random.randn(m, n).astype(np.float32)

        start_time = time.time()
        result = py_gpu_algos.matrix_transpose_striped(matrix)
        elapsed = time.time() - start_time

        # Verify correctness
        reference = matrix.T
        max_error = np.max(np.abs(result - reference))

        print(f"  Result: {result.shape}, Time: {elapsed*1000:.2f}ms, Error: {max_error:.2e}")
        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create float64 matrix
    matrix = np.random.randn(60, 80).astype(np.float64)

    # High-level function (automatic dispatch)
    result_high = py_gpu_algos.matrix_transpose_striped(matrix)

    # Low-level function (explicit type)
    result_low = py_gpu_algos.matrix_transpose_striped_float64(matrix)

    # Compare
    print(f"Original shape: {matrix.shape}")
    print(f"High-level result shape: {result_high.shape}")
    print(f"Low-level result shape: {result_low.shape}")
    print(f"Results match: {np.allclose(result_high, result_low)}")
    print()

def performance_analysis():
    """Performance analysis for different matrix sizes."""
    print("=== Performance Analysis ===")

    sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 512),
        (512, 1024)
    ]

    for m, n in sizes:
        print(f"Matrix ({m}x{n}):")

        # Create matrix
        matrix = np.random.randn(m, n).astype(np.float32)

        # GPU timing
        start_time = time.time()
        result_gpu = py_gpu_algos.matrix_transpose_striped(matrix)
        gpu_time = time.time() - start_time

        # CPU timing
        start_time = time.time()
        result_cpu = matrix.T
        cpu_time = time.time() - start_time

        # Analysis
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        accuracy = np.allclose(result_gpu, result_cpu)

        # Memory bandwidth estimation
        bytes_transferred = matrix.nbytes * 2  # Read + write
        gpu_bandwidth = bytes_transferred / gpu_time / 1e9  # GB/s
        cpu_bandwidth = bytes_transferred / cpu_time / 1e9  # GB/s

        print(f"  GPU: {gpu_time*1000:6.2f}ms ({gpu_bandwidth:5.1f} GB/s)")
        print(f"  CPU: {cpu_time*1000:6.2f}ms ({cpu_bandwidth:5.1f} GB/s)")
        print(f"  Speedup: {speedup:.2f}x, Accurate: {accuracy}")
        print()

def memory_layout_demonstration():
    """Demonstrate memory layout considerations."""
    print("=== Memory Layout Considerations ===")

    # Create matrices with different layouts
    m, n = 128, 256

    # Contiguous C-order matrix
    matrix_c = np.random.randn(m, n).astype(np.float32)

    # Non-contiguous view (every other row)
    matrix_nc = matrix_c[::2, :]

    print(f"Original matrix: {matrix_c.shape}, contiguous: {matrix_c.flags.c_contiguous}")
    print(f"Non-contiguous view: {matrix_nc.shape}, contiguous: {matrix_nc.flags.c_contiguous}")

    # Time both
    for name, matrix in [("Contiguous", matrix_c), ("Non-contiguous", matrix_nc)]:
        start_time = time.time()
        result = py_gpu_algos.matrix_transpose_striped(matrix)
        elapsed = time.time() - start_time

        # Verify against NumPy
        reference = matrix.T
        error = np.max(np.abs(result - reference))

        print(f"  {name}: {elapsed*1000:6.2f}ms, error: {error:.2e}")

    print()

def transpose_chain_operations():
    """Example chaining transpose with other operations."""
    print("=== Transpose Chain Operations ===")

    # Create matrices for A^T @ B computation
    m, n, k = 64, 80, 96
    A = np.random.randn(n, m).astype(np.float32)  # Note: transposed dimensions
    B = np.random.randn(n, k).astype(np.float32)

    print(f"A: {A.shape}, B: {B.shape}")
    print("Computing (A^T @ B) using GPU operations:")

    # Method 1: Transpose then multiply
    start_time = time.time()
    A_T = py_gpu_algos.matrix_transpose_striped(A)
    result1 = py_gpu_algos.matrix_product_naive(A_T, B)
    method1_time = time.time() - start_time

    print(f"  Method 1 (transpose + multiply): {method1_time*1000:.2f}ms")
    print(f"  A^T shape: {A_T.shape}, Result shape: {result1.shape}")

    # NumPy reference
    start_time = time.time()
    result_numpy = np.dot(A.T, B)
    numpy_time = time.time() - start_time

    print(f"  NumPy reference: {numpy_time*1000:.2f}ms")

    # Compare accuracy
    error = np.max(np.abs(result1 - result_numpy))
    speedup = numpy_time / method1_time if method1_time > 0 else float('inf')

    print(f"  Max error: {error:.2e}")
    print(f"  Speedup vs NumPy: {speedup:.2f}x")
    print()

def main():
    """Run all examples."""
    print("Matrix Transpose Striped Kernel Examples")
    print("=" * 50)

    try:
        basic_usage()
        different_data_types()
        different_matrix_shapes()
        low_level_functions()
        performance_analysis()
        memory_layout_demonstration()
        transpose_chain_operations()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
