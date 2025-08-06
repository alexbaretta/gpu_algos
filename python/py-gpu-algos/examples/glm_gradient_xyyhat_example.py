#!/usr/bin/env python3
"""
Example: GLM Gradient XYYhat Kernel

This example demonstrates how to use the glm_gradient_xyyhat kernel,
an optimized implementation for gradient computation in linear regression.
"""

import numpy as np
import time
import py_gpu_algos

def basic_usage():
    """Basic usage example with optimized GLM gradient computation."""
    print("=== Basic Usage: glm_gradient_xyyhat ===")

    # Create example tensors
    nfeatures, ntargets, ntasks, nobs = 6, 3, 4, 15

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Features X: {X.shape} ({X.dtype})")
    print(f"Targets Y: {Y.shape} ({Y.dtype})")
    print(f"Model M: {M.shape} ({M.dtype})")

    # GPU computation (optimized)
    start_time = time.time()
    grad_opt = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
    opt_time = time.time() - start_time

    # GPU computation (naive for comparison)
    start_time = time.time()
    grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
    naive_time = time.time() - start_time

    # Compare results
    max_error = np.max(np.abs(grad_opt - grad_naive))
    speedup = naive_time / opt_time if opt_time > 0 else float('inf')

    print(f"Gradient shape: {grad_opt.shape}")
    print(f"XYYhat time: {opt_time*1000:.2f} ms")
    print(f"Naive time: {naive_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max difference: {max_error:.2e}")
    print(f"Results match: {np.allclose(grad_opt, grad_naive, rtol=1e-6)}")
    print()

def algorithm_optimization_analysis():
    """Analyze the optimization benefits of XYYhat algorithm."""
    print("=== Algorithm Optimization Analysis ===")

    print("XYYhat algorithm optimizes memory access patterns and reduces")
    print("redundant computations compared to the naive implementation.")
    print()

    # Test different problem sizes to show scaling benefits
    sizes = [
        (10, 3, 5, 100),
        (25, 5, 8, 250),
        (50, 10, 10, 500),
        (100, 15, 15, 1000),
    ]

    print("Problem Size Analysis:")
    print("Features×Targets×Tasks, Obs | Naive (ms) | XYYhat (ms) | Speedup")
    print("----------------------------|------------|-------------|--------")

    for nfeatures, ntargets, ntasks, nobs in sizes:
        # Create test data
        X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
        Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
        M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

        # Time naive algorithm
        start_time = time.time()
        grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
        naive_time = time.time() - start_time

        # Time optimized algorithm
        start_time = time.time()
        grad_opt = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
        opt_time = time.time() - start_time

        # Calculate speedup
        speedup = naive_time / opt_time if opt_time > 0 else float('inf')

        # Verify correctness
        correct = np.allclose(grad_naive, grad_opt, rtol=1e-5)
        status = "✓" if correct else "✗"

        print(f"    {nfeatures:>2}×{ntargets:>2}×{ntasks:>2}, {nobs:>4}     |   {naive_time*1000:>6.2f}   |   {opt_time*1000:>6.2f}    |  {speedup:>5.2f} {status}")

    print()

def memory_efficiency_demo():
    """Demonstrate memory efficiency characteristics."""
    print("=== Memory Efficiency Demonstration ===")

    print("XYYhat algorithm reduces memory allocations and improves")
    print("cache efficiency through optimized computation order.")
    print()

    # Test with larger problem to show memory benefits
    nfeatures, ntargets, ntasks, nobs = 50, 8, 12, 800

    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)
    Y = np.random.randn(ntargets, ntasks, nobs).astype(np.float32)
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32)

    print(f"Large problem: {nfeatures} features, {ntargets} targets, {ntasks} tasks, {nobs} obs")
    print(f"Input memory: {(X.nbytes + Y.nbytes + M.nbytes) / 1024**2:.1f} MB")

    # Multiple runs to test consistency
    naive_times = []
    opt_times = []

    for run in range(5):
        # Naive algorithm
        start_time = time.time()
        grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
        naive_times.append(time.time() - start_time)

        # Optimized algorithm
        start_time = time.time()
        grad_opt = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
        opt_times.append(time.time() - start_time)

    # Statistics
    naive_avg = np.mean(naive_times)
    naive_std = np.std(naive_times)
    opt_avg = np.mean(opt_times)
    opt_std = np.std(opt_times)

    print(f"\nTiming statistics (5 runs):")
    print(f"Naive:   {naive_avg*1000:6.2f} ± {naive_std*1000:4.2f} ms")
    print(f"XYYhat:  {opt_avg*1000:6.2f} ± {opt_std*1000:4.2f} ms")
    print(f"Speedup: {naive_avg/opt_avg:6.2f}x")
    print(f"Accuracy: {np.allclose(grad_naive, grad_opt, rtol=1e-5)}")
    print()

def batch_gradient_descent():
    """Example using XYYhat in batch gradient descent."""
    print("=== Batch Gradient Descent Example ===")

    print("Training multiple regression models simultaneously")
    print("using optimized gradient computation.")
    print()

    # Setup multitask regression problem
    nfeatures, ntargets, ntasks, nobs = 8, 2, 6, 100

    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.randn(nfeatures, ntasks, nobs).astype(np.float32)

    # Create different true models for each task
    M_true = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32) * 2

    # Generate targets with noise
    Y = np.zeros((ntargets, ntasks, nobs), dtype=np.float32)
    for task in range(ntasks):
        Y[:, task, :] = M_true[:, :, task].T @ X[:, task, :]
    Y += 0.2 * np.random.randn(*Y.shape).astype(np.float32)

    # Initialize model parameters
    M = np.random.randn(nfeatures, ntargets, ntasks).astype(np.float32) * 0.1

    # Training parameters
    learning_rate = 0.01
    num_epochs = 200

    # Track training progress
    losses = []
    gradient_times = []

    print("Training progress:")
    print("Epoch | Avg Loss | Grad Time (ms)")
    print("------|----------|---------------")

    for epoch in range(0, num_epochs, 20):
        # Compute gradient using optimized algorithm
        start_time = time.time()
        grad = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)
        grad_time = time.time() - start_time
        gradient_times.append(grad_time)

        # Update parameters
        M -= learning_rate * grad

        # Compute loss
        Y_pred = py_gpu_algos.glm_predict_naive(X, M)
        loss = np.mean((Y_pred - Y) ** 2)
        losses.append(loss)

        print(f" {epoch:>4} |  {loss:>6.4f}  |     {grad_time*1000:>6.2f}")

    print()

    # Final results
    avg_grad_time = np.mean(gradient_times)
    final_loss = losses[-1]

    print("Training completed:")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Average gradient time: {avg_grad_time*1000:.2f} ms")
    print(f"Total gradient computations: {len(gradient_times)}")

    # Compare learned vs true parameters for one task
    task_id = 0
    print(f"\nLearned vs True parameters (Task {task_id}):")
    print("Feature | Target 0      | Target 1")
    print("        | True | Learn  | True | Learn")
    print("--------|------|--------|------|-------")
    for f in range(min(5, nfeatures)):
        true_0, learn_0 = M_true[f, 0, task_id], M[f, 0, task_id]
        true_1, learn_1 = M_true[f, 1, task_id], M[f, 1, task_id]
        print(f"   {f}    |{true_0:>5.2f} |{learn_0:>6.2f}  |{true_1:>5.2f} |{learn_1:>6.2f}")

    print()

def different_data_types():
    """Test with different supported data types."""
    print("=== Different Data Types ===")

    # Test supported types
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        print(f"Testing {dtype.__name__}:")

        try:
            # Create appropriate test data
            if dtype in [np.int32, np.int64]:
                X = np.random.randint(-2, 3, (4, 2, 6), dtype=dtype)
                Y = np.random.randint(-3, 4, (2, 2, 6), dtype=dtype)
                M = np.random.randint(-2, 3, (4, 2, 2), dtype=dtype)
            else:
                X = np.random.randn(4, 2, 6).astype(dtype)
                Y = np.random.randn(2, 2, 6).astype(dtype)
                M = np.random.randn(4, 2, 2).astype(dtype)

            # Test both algorithms
            grad_naive = py_gpu_algos.glm_gradient_naive(X, Y, M)
            grad_opt = py_gpu_algos.glm_gradient_xyyhat(X, Y, M)

            # Compare
            if dtype in [np.int32, np.int64]:
                match = np.array_equal(grad_naive, grad_opt)
            else:
                match = np.allclose(grad_naive, grad_opt, rtol=1e-6)

            print(f"  ✅ Supported - Algorithms match: {match}")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

        print()

def main():
    """Run all examples."""
    print("GLM Gradient XYYhat Kernel Examples")
    print("=" * 45)

    try:
        basic_usage()
        algorithm_optimization_analysis()
        memory_efficiency_demo()
        batch_gradient_descent()
        different_data_types()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
