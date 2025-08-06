#!/usr/bin/env python3
"""
Example: GLM Gradient Naive Kernel

This example demonstrates how to use the glm_gradient_naive kernel
for computing gradients in linear regression with 3D tensors.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with GLM gradient computation."""
    print("=== Basic Usage: glm_gradient_naive ===")

    # Create example tensors for gradient computation
    nfeatures, ntargets, ntasks, nobs = 4, 2, 3, 10

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Features X: {X.shape} ({X.dtype})")
    print(f"Targets Y: {Y.shape} ({Y.dtype})")
    print(f"Model M: {M.shape} ({M.dtype})")

    # GPU computation
    start_time = time.time()
    grad_gpu = py_gpu_algos.glm_gradient_naive(X, Y, M)
    gpu_time = time.time() - start_time

    # NumPy reference computation
    start_time = time.time()
    grad_ref = np.zeros_like(M)
    for task in range(ntasks):
        # Y_pred = M[:, :, task].T @ X[:, task, :]
        Y_pred = M[:, :, task].T @ X[:, task, :]
        # Residual = Y_pred - Y[:, task, :]
        residual = Y_pred - Y[:, task, :]
        # Gradient = X[:, task, :] @ residual.T
        grad_ref[:, :, task] = X[:, task, :] @ residual.T
    numpy_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(grad_gpu - grad_ref))
    speedup = numpy_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"Gradient shape: {grad_gpu.shape}")
    print(f"Max error vs NumPy: {max_error:.2e}")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()

def regression_example():
    """Linear regression gradient descent example."""
    print("=== Linear Regression Example ===")

    print("Simulating gradient descent for house price prediction")
    print("Features: size, bedrooms, location_score")
    print("Target: price")
    print()

    nfeatures, ntargets, ntasks, nobs = 3, 1, 2, 20

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)

    # True model parameters
    M_true = np.array([[[2.0], [1.5]],
                      [[3.0], [2.5]],
                      [[1.0], [0.8]]], dtype=np.float32)

    # Generate true targets with noise
    Y_true = np.zeros((ntargets, ntasks, nobs), dtype=np.float32)
    for task in range(ntasks):
        Y_true[:, task, :] = M_true[:, :, task].T @ X[:, task, :]
    Y_true += 0.1 * np.random.randn(*Y_true.shape).astype(np.float32)

    # Initialize model with random parameters
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32) * 0.1

    # Gradient descent
    learning_rate = 0.01
    num_iterations = 100

    print("Iteration | Loss (Task 0) | Loss (Task 1)")
    print("----------|---------------|-------------")

    for i in range(0, num_iterations, 10):
        # Compute gradient
        grad = py_gpu_algos.glm_gradient_naive(X, Y_true, M)

        # Update parameters
        M -= learning_rate * grad

        # Compute loss for monitoring
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)
        loss_task0 = np.mean((Y_pred[0, 0, :] - Y_true[0, 0, :]) ** 2)
        loss_task1 = np.mean((Y_pred[0, 1, :] - Y_true[0, 1, :]) ** 2)

        print(f"    {i:>3}   |     {loss_task0:>6.3f}    |    {loss_task1:>6.3f}")

    print()
    print("Final learned parameters vs true parameters:")
    print("Feature | True (Task 0) | Learned (Task 0) | True (Task 1) | Learned (Task 1)")
    print("--------|---------------|------------------|---------------|------------------")
    for f in range(nfeatures):
        print(f"   {f}    |     {M_true[f, 0, 0]:>6.2f}    |      {M[f, 0, 0]:>6.2f}       |     {M_true[f, 0, 1]:>6.2f}    |      {M[f, 0, 1]:>6.2f}")

    print()

def compare_with_gradient_xyyhat():
    """Compare naive gradient with optimized xyyhat gradient."""
    print("=== Comparison: gradient_naive vs gradient_xyyhat ===")

    # Create test tensors
    nfeatures, ntargets, ntasks, nobs = 10, 3, 5, 50

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Problem size: {nfeatures} features, {ntargets} targets, {ntasks} tasks, {nobs} obs")

    # Naive algorithm
    start_time = time.time()
    grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
    naive_time = time.time() - start_time

    # Optimized algorithm
    start_time = time.time()
    grad_xyyhat = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
    xyyhat_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(grad_naive - grad_xyyhat))
    speedup = naive_time / xyyhat_time if xyyhat_time > 0 else float('inf')

    print(f"Naive algorithm:     {naive_time*1000:6.2f} ms")
    print(f"XYYhat algorithm:    {xyyhat_time*1000:6.2f} ms")
    print(f"Speedup (xyyhat):    {speedup:.2f}x")
    print(f"Max difference:      {max_error:.2e}")
    print(f"Algorithms match:    {np.allclose(grad_naive, grad_xyyhat, rtol=1e-5)}")
    print()

def different_data_types():
    """Example with different supported data types."""
    print("=== Different Data Types ===")

    # GLM operations support limited types
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        try:
            # Create small tensors
            if dtype in [np.int32, np.int64]:
                X = np.random.randint(-3, 3, (3, 2, 5), dtype=dtype)
                Y = np.random.randint(-5, 5, (2, 2, 5), dtype=dtype)
                M = np.random.randint(-2, 2, (3, 2, 2), dtype=dtype)
            else:
                X = np.random.randn(3, 2, 5).astype(dtype)
                Y = np.random.randn(2, 2, 5).astype(dtype)
                M = np.random.randn(3, 2, 2).astype(dtype)

            # Compute gradient
            grad = py_gpu_algos.glm_gradient_naive(X, Y, M)

            print(f"  ✅ Supported - Gradient shape: {grad.shape}")

            # Show sample gradient values
            print(f"  Sample gradient: {grad[0, 0, 0]:.3f}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def performance_scaling():
    """Analyze performance scaling with problem size."""
    print("=== Performance Scaling Analysis ===")

    # Test different problem sizes
    configs = [
        (5, 2, 3, 50),     # Small
        (20, 5, 5, 200),   # Medium
        (50, 10, 10, 500), # Large
    ]

    for nfeatures, ntargets, ntasks, nobs in configs:
        print(f"Size: {nfeatures}×{ntargets}×{ntasks}, {nobs} obs")

        # Create tensors
        X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

        # Time GPU gradient computation
        start_time = time.time()
        grad_gpu = py_gpu_algos.glm_gradient_naive(X, Y, M)
        gpu_time = time.time() - start_time

        # Time CPU computation
        start_time = time.time()
        grad_cpu = np.zeros_like(M)
        for task in range(ntasks):
            Y_pred = M[:, :, task].T @ X[:, task, :]
            residual = Y_pred - Y[:, task, :]
            grad_cpu[:, :, task] = X[:, task, :] @ residual.T
        cpu_time = time.time() - start_time

        # Analysis
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        accuracy = np.allclose(grad_gpu, grad_cpu, rtol=1e-5)

        # Compute FLOPs estimate
        flops = ntasks * (2 * nfeatures * ntargets * nobs + 2 * nfeatures * nobs * ntargets)
        gpu_gflops = flops / gpu_time / 1e9

        print(f"  GPU: {gpu_time*1000:6.2f}ms ({gpu_gflops:5.2f} GFLOPS)")
        print(f"  CPU: {cpu_time*1000:6.2f}ms")
        print(f"  Speedup: {speedup:.2f}x, Accurate: {accuracy}")
        print()

def main():
    """Run all examples."""
    print("GLM Gradient Naive Kernel Examples")
    print("=" * 40)

    try:
        basic_usage()
        regression_example()
        compare_with_gradient_xyyhat()
        different_data_types()
        performance_scaling()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
