#!/usr/bin/env python3
"""
Example: GLM Predict Naive Kernel

This example demonstrates how to use the glm_predict_naive kernel
for linear model prediction with 3D tensors for multitask learning.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with GLM prediction."""
    print("=== Basic Usage: glm_predict_naive ===")

    # Create example tensors for multitask learning
    nfeatures, ntasks, nobs = 5, 3, 10
    ntargets = 2

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Features tensor X: {X.shape} ({X.dtype})")
    print(f"Model tensor M: {M.shape} ({M.dtype})")

    # GPU computation
    start_time = time.time()
    Y_pred = py_gpu_algos.glm_predict_naive(X, M)
    gpu_time = time.time() - start_time

    # NumPy reference computation
    start_time = time.time()
    Y_ref = np.zeros((ntargets, ntasks, nobs), dtype=np.float32)
    for task in range(ntasks):
        Y_ref[:, task, :] = M[:, :, task].T @ X[:, task, :]
    numpy_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(Y_pred - Y_ref))
    speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"Prediction shape: {Y_pred.shape}")
    print(f"Max error vs NumPy: {max_error:.2e}")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()

def different_data_types():
    """Example with different data types."""
    print("=== Different Data Types ===")

    # GLM operations support limited types
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        try:
            # Create tensors
            if dtype in [np.int32, np.int64]:
                X = np.random.randint(-5, 5, (3, 2, 8), dtype=dtype)
                M = np.random.randint(-3, 3, (3, 2, 2), dtype=dtype)
            else:
                X = np.random.randn(3, 2, 8).astype(dtype)
                M = np.random.randn(3, 2, 2).astype(dtype)

            # Compute prediction
            Y_pred = py_gpu_algos.glm_predict_naive(X, M)

            # NumPy reference
            ntargets, ntasks, nobs = Y_pred.shape
            Y_ref = np.zeros((ntargets, ntasks, nobs), dtype=dtype)
            for task in range(ntasks):
                Y_ref[:, task, :] = M[:, :, task].T @ X[:, task, :]

            # Check accuracy
            if dtype in [np.int32, np.int64]:
                accuracy_ok = np.array_equal(Y_pred, Y_ref)
                print(f"  ✅ Supported - Exact match: {accuracy_ok}")
            else:
                max_error = np.max(np.abs(Y_pred - Y_ref))
                print(f"  ✅ Supported - Max error: {max_error:.2e}")

            print(f"  Shapes: X{X.shape} @ M{M.shape} -> Y{Y_pred.shape}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def multitask_learning_example():
    """Real-world multitask learning example."""
    print("=== Multitask Learning Example ===")

    print("Scenario: Predicting house prices across different cities")
    print("- Features: size, bedrooms, age, location_score")
    print("- Targets: price, rent_estimate")
    print("- Tasks: 3 different cities")
    print()

    nfeatures = 4  # size, bedrooms, age, location_score
    ntargets = 2   # price, rent_estimate
    ntasks = 3     # 3 cities
    nobs = 50      # 50 houses per city

    # Create realistic feature data
    np.random.seed(42)
    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)

    # Create model parameters (different for each city)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Feature tensor X: {X.shape}")
    print(f"Model tensor M: {M.shape}")

    # Make predictions
    Y_pred = py_gpu_algos.glm_predict_naive(X, M)

    print(f"Predictions Y: {Y_pred.shape}")
    print()

    # Show sample predictions for first city
    city = 0
    print(f"Sample predictions for city {city + 1}:")
    print("House | Price Pred | Rent Pred")
    print("------|------------|----------")
    for house in range(min(5, nobs)):
        price_pred = Y_pred[0, city, house]
        rent_pred = Y_pred[1, city, house]
        print(f"  {house+1:>3} | {price_pred:>8.1f}  | {rent_pred:>7.1f}")

    print()

def performance_scaling():
    """Analyze performance scaling with problem size."""
    print("=== Performance Scaling Analysis ===")

    # Test different problem sizes
    configs = [
        (10, 3, 5, 100),    # Small
        (50, 5, 10, 500),   # Medium
        (100, 10, 20, 1000), # Large
    ]

    for nfeatures, ntargets, ntasks, nobs in configs:
        print(f"Problem size: {nfeatures} features, {ntargets} targets, {ntasks} tasks, {nobs} obs")

        # Create tensors
        X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

        # GPU timing
        start_time = time.time()
        Y_gpu = py_gpu_algos.glm_predict_naive(X, M)
        gpu_time = time.time() - start_time

        # CPU timing (task-by-task)
        start_time = time.time()
        Y_cpu = np.zeros((ntargets, ntasks, nobs), dtype=np.float32)
        for task in range(ntasks):
            Y_cpu[:, task, :] = M[:, :, task].T @ X[:, task, :]
        cpu_time = time.time() - start_time

        # Analysis
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        accuracy = np.allclose(Y_gpu, Y_cpu, rtol=1e-5)

        print(f"  GPU: {gpu_time*1000:6.2f}ms, CPU: {cpu_time*1000:6.2f}ms")
        print(f"  Speedup: {speedup:.2f}x, Accurate: {accuracy}")
        print()

def low_level_functions():
    """Example using low-level type-specific functions."""
    print("=== Low-Level Type-Specific Functions ===")

    # Create float64 tensors
    X = np.random.randn(4, 2, 6).astype(np.float64)
    M = np.random.randn(4, 3, 2).astype(np.float64)

    print(f"Input X: {X.shape} ({X.dtype})")
    print(f"Input M: {M.shape} ({M.dtype})")

    # High-level function (automatic dispatch)
    Y_high = py_gpu_algos.glm_predict_naive(X, M)

    # Low-level function (explicit type)
    Y_low = py_gpu_algos.glm_predict_naive_float64(X, M)

    # Compare
    print(f"High-level result shape: {Y_high.shape}")
    print(f"Low-level result shape: {Y_low.shape}")
    print(f"Results match: {np.allclose(Y_high, Y_low)}")
    print()

def main():
    """Run all examples."""
    print("GLM Predict Naive Kernel Examples")
    print("=" * 40)

    try:
        basic_usage()
        different_data_types()
        multitask_learning_example()
        performance_scaling()
        low_level_functions()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
